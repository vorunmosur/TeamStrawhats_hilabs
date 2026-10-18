"""Offline mapping command: one legal ICD-10-CM and SNOMED CT code per input."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .input_adapter import input_rows,labelled_rows
from .normalize import level1,context_window
from .encode_retrieval import token_windows
from .retrieval import HybridRetriever,ALL_FEATURES
from .rank_retrieval import TrainingMemory,assemble,ordering,SYSTEMS
from .deployment_memory import DeploymentMemory,input_key
from .prepare_retrieval import file_sha,SAPBERT_MODEL,SAPBERT_REVISION

class QueryEncoder:
    def __init__(self,path,device):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.device=device
        source=json.loads((Path(path)/'MODEL_SOURCE.json').read_text())
        if source['model_id']!=SAPBERT_MODEL or source['revision']!=SAPBERT_REVISION:raise ValueError('Wrong pinned encoder')
        for name,digest in source['files'].items():
            if file_sha(Path(path)/name)!=digest:raise ValueError(f'Encoder file changed: {name}')
        torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.float16 if device=='cuda' else torch.float32,
                                             attn_implementation='sdpa').to(device).eval()
    def encode(self,text):
        torch=self.torch;tokenizer=self.tokenizer
        tokens=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
        windows=list(enumerate(token_windows(tokens)))
        windows.sort(key=lambda pair:(len(pair[1]),pair[0]))
        sums=np.zeros(self.model.config.hidden_size,dtype=np.float32)
        # Batch one matches the complete query encoding job; long inputs retain all tokens.
        with torch.inference_mode():
            for _,window in windows:
                ids=torch.tensor([[tokenizer.cls_token_id]+window+[tokenizer.sep_token_id]],device=self.device)
                vector=self.model(input_ids=ids,attention_mask=torch.ones_like(ids)).last_hidden_state[0,0].float().cpu().numpy()
                sums+=vector
        means=sums[None,:]/np.asarray([[len(windows)]],dtype=np.int32)
        norms=np.linalg.norm(means,axis=1,keepdims=True)
        if not np.isfinite(means).all() or np.any(norms==0):raise ValueError('Invalid query embedding')
        return np.asarray((means/norms)[0],dtype=np.float32)

class MappingEngine:
    def __init__(self,root,device='auto',candidate_cache=None,force_reencode=False):
        self.root=Path(root);self.force_reencode=force_reencode
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        import torch
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        manifest=json.loads((self.root/'RUNTIME_MANIFEST.json').read_text())
        if manifest['status']!='complete':raise ValueError('Incomplete runtime bundle')
        for name,digest in manifest['files'].items():
            resource=(self.root/name).resolve()
            if not resource.is_relative_to(self.root.resolve()) or file_sha(resource)!=digest:
                raise ValueError(f'Runtime artifact changed: {name}')
        for name,digest in manifest['source_sha256'].items():
            if file_sha(Path(__file__).with_name(name))!=digest:raise ValueError(f'Runtime code changed: {name}')
        result=json.loads((self.root/'models/RANKER_RESULT.json').read_text())
        if result['status']!='complete' or result['rows']!=9644:raise ValueError('Completed all-label final models required')
        lock=json.loads((self.root/'models/RANKER_LOCK.json').read_text())
        if lock['features']!=ALL_FEATURES:raise ValueError('Model feature schema mismatch')
        self.index=HybridRetriever(self.root/'prepared',self.root/'embeddings',self.root/'lexical',device=self.device)
        self.queries=pd.read_parquet(self.root/'prepared/queries.parquet')
        self.rows=labelled_rows(self.root/'prepared/rows.parquet')
        self.vectors=np.load(self.root/'embeddings/queries.npy',mmap_mode='r').reshape(-1,2,768)
        self.hashed_memory='context_key_sha256' in self.queries.columns
        self.known={r.context_key_sha256:int(r.query_id) for r in self.queries.itertuples(index=False)} if self.hashed_memory else {(r.term_norm,r.context_norm):int(r.query_id) for r in self.queries.itertuples(index=False)}
        memory_class=DeploymentMemory if self.hashed_memory else TrainingMemory
        self.memories={s:memory_class(self.rows,self.queries,self.vectors,s,self.index.details) for s in SYSTEMS}
        self.models={}
        for s in SYSTEMS:
            path=self.root/f'models/{s}_final_ranker.joblib'
            if file_sha(path)!=result['models'][s]:raise ValueError('Changed final ranker')
            self.models[s]=joblib.load(path)
        self.encoder=None;self.output_cache={};self.candidate_cache=None
        if candidate_cache:
            from .rank_retrieval import CandidateCache
            self.candidate_cache=CandidateCache(candidate_cache)
    def predict(self,entity,chunk):
        term=level1(entity);context=context_window(entity,chunk);key=(term,context)
        if key in self.output_cache:return self.output_cache[key]
        query_id=self.known.get(input_key(term,context) if self.hashed_memory else key)
        query=pd.Series(dict(term_norm=term,context_norm=context))
        if query_id is None or self.force_reencode:
            if self.encoder is None:self.encoder=QueryEncoder(self.root/'encoder',self.device)
            vectors=np.stack([self.encoder.encode(term),self.encoder.encode(term if not context else term+' [SEP] '+context)])
        else:vectors=np.asarray(self.vectors[query_id],dtype=np.float32)
        vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
        output={}
        for system in SYSTEMS:
            base=self.candidate_cache.get(query_id,system) if self.candidate_cache is not None and query_id is not None else self.index.candidates(system,term,context,*vectors)
            codes,matrix,aliases=assemble(self.index,self.memories[system],query,vectors,base)
            scores=self.models[system].predict_proba(matrix)[:,1];order=ordering(codes,scores)
            top=[]
            for i in order[:20]:
                i=int(i);code=codes[i]
                if (system,code) not in self.index.details:raise ValueError('Illegal output code')
                top.append(dict(code=code,name=self.index.details[(system,code)]['name'],matched_alias=self.index.names[int(aliases[i])],ranking_score=float(scores[i])))
            output[system]=dict(code=top[0]['code'],ranking_score=top[0]['ranking_score'],
                score_margin=top[0]['ranking_score']-top[1]['ranking_score'],top20=top)
        self.output_cache[key]=output
        return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['predict'])
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunks',type=Path)
    parser.add_argument('--contextless',action='store_true',help='Explicitly map entity-only input when chart context is unavailable')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--candidate-cache',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--reencode',action='store_true',help='Independently encode every input instead of reusing frozen known-query vectors')
    parser.add_argument('--deadline',help=argparse.SUPPRESS)
    parser.add_argument('--usable-seconds',type=float,default=3600,help=argparse.SUPPRESS)
    args=parser.parse_args();rows,entity=input_rows(args.input,args.chunks,args.contextless)
    if args.output.exists():raise ValueError('Choose a new output file')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    engine=MappingEngine(args.artifacts,args.device,args.candidate_cache,args.reencode)
    if args.reencode:engine.encoder=QueryEncoder(args.artifacts/'encoder',engine.device)
    from datetime import datetime
    deadline=datetime.fromisoformat(args.deadline) if args.deadline else None
    job_start=time.monotonic()
    columns=list(dict.fromkeys([c for c in rows.columns if c not in ('icd10cm_code','snomed_code')]+['icd10cm_code','snomed_code']))
    latencies=[];debug=args.output.with_name(args.output.stem+'_debug.jsonl');temporary=args.output.with_suffix('.partial.csv')
    with temporary.open('w',encoding='utf-8',newline='') as f,debug.open('w',encoding='utf-8') as d:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for number,(_,row) in enumerate(rows.iterrows()):
            start=time.monotonic();result=engine.predict(row[entity],row.chunk_text);latencies.append(time.monotonic()-start)
            record={c:row[c] for c in rows.columns if c in columns}
            record.update(icd10cm_code=result['ICD10CM']['code'],snomed_code=result['SNOMEDCT_US']['code'])
            writer.writerow(record);d.write(json.dumps(dict(input_row=number,systems=result))+'\n')
            if number==1 or (number+1)%250==0:
                f.flush();d.flush();elapsed=time.monotonic()-job_start
                forecast=elapsed/(number+1)*(len(rows)-number-1)*1.25
                available=max(0,min(args.usable_seconds-elapsed,(deadline-datetime.now(deadline.tzinfo)).total_seconds())) if deadline else float('inf')
                plan=dict(mapped=number+1,scope=len(rows),measured_seconds=elapsed,forecast_remaining_seconds=forecast,available_seconds=available,fits=forecast<=available)
                if deadline:args.output.with_suffix('.compute.json').write_text(json.dumps(plan,indent=2)+'\n')
                print(json.dumps(plan),flush=True)
                if deadline and not plan['fits']:raise SystemExit('Prediction compute gate stopped; partial predictions retained')
    temporary.replace(args.output)
    metrics=dict(rows=len(rows),both_systems_predicted=True,valid_dictionary_codes=True,output_sha256=file_sha(args.output),
        median_seconds=float(np.median(latencies)) if latencies else 0,p95_seconds=float(np.percentile(latencies,95)) if latencies else 0,
        ranking_scores='Uncalibrated candidate ranking scores; not probabilities of correctness.',
        latencies_seconds=latencies,reencoded_inputs=args.reencode,
        benchmark='Includes normalization, retrieval and ranking; excludes model/index startup. Repeated inputs may use in-process cache.')
    args.output.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
