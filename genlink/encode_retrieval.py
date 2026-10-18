"""Checkpointed full-catalogue SapBERT encoding on two private Kaggle T4s."""
import argparse
from datetime import datetime
import json
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .prepare_retrieval import file_sha,SAPBERT_MODEL,SAPBERT_REVISION


def token_windows(tokens,capacity=510,stride=384):
    if capacity<1 or not 0<stride<=capacity:raise ValueError('Invalid window geometry')
    tokens=list(tokens)
    begin=0
    while True:
        yield tokens[begin:begin+capacity]
        if begin+capacity>=len(tokens):break
        begin+=stride


def token_work(lengths,batch_size):
    ordered=sorted(lengths)
    return sum(max(ordered[i:i+batch_size])*len(ordered[i:i+batch_size])
               for i in range(0,len(ordered),batch_size))


def prepare_tokens(prepared,model,output):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    source=json.loads((model/'MODEL_SOURCE.json').read_text())
    manifest=json.loads((prepared/'CATALOG_MANIFEST.json').read_text())
    if source['model_id']!=SAPBERT_MODEL or source['revision']!=SAPBERT_REVISION:
        raise ValueError('Unexpected biomedical encoder snapshot')
    for name,digest in source['files'].items():
        if file_sha(model/name)!=digest:raise ValueError(f'Changed model file: {name}')
    for name,digest in manifest['files'].items():
        if file_sha(prepared/name)!=digest:raise ValueError(f'Changed catalogue file: {name}')
    identity=dict(catalogue_sha256=file_sha(prepared/'CATALOG_MANIFEST.json'),model_source=source,
                  encoder_source_sha256=file_sha(Path(__file__)),name_batch_size=128,query_batch_size=1,
                  capacity=510,stride=384,normalization='L2-normalized mean of window CLS vectors',name_dtype='float16',query_dtype='float32',
                  python=sys.version,packages={name:importlib.metadata.version(name) for name in
                  ('torch','transformers','numpy','pandas','pyarrow','scikit-learn','scipy','huggingface-hub')})
    lock=output/'ENCODER_LOCK.json'
    if lock.exists() and json.loads(lock.read_text())!=identity:
        raise ValueError('Encoder output belongs to another configuration')
    lock.write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    plan_path=output/'TOKEN_PLAN.json'
    if plan_path.exists():return json.loads(plan_path.read_text())
    tables={'names':pd.read_parquet(prepared/'names.parquet').name.tolist()}
    queries=pd.read_parquet(prepared/'queries.parquet')
    tables['queries']=[text for pair in zip(queries.term_norm,queries.context_query) for text in pair]
    plan=dict(kinds={},special_tokens=dict(cls=tokenizer.cls_token_id,sep=tokenizer.sep_token_id,pad=tokenizer.pad_token_id))
    for kind,texts in tables.items():
        arrays=[];offsets=[0]
        for begin in range(0,len(texts),8192):
            encoded=tokenizer(texts[begin:begin+8192],add_special_tokens=False,truncation=False)['input_ids']
            for tokens in encoded:
                arrays.append(np.asarray(tokens,dtype=np.int32));offsets.append(offsets[-1]+len(tokens))
        flat=np.concatenate(arrays) if arrays else np.asarray([],dtype=np.int32)
        np.save(output/f'{kind}-tokens.npy',flat)
        np.save(output/f'{kind}-offsets.npy',np.asarray(offsets,dtype=np.int64))
        chunk_size=8192 if kind=='names' else 128
        batch=128 if kind=='names' else 1
        chunks=[]
        for begin in range(0,len(texts),chunk_size):
            end=min(begin+chunk_size,len(texts));window_lengths=[]
            for i in range(begin,end):
                length=offsets[i+1]-offsets[i]
                window_lengths.extend(len(window)+2 for window in token_windows(range(length)))
            chunks.append(dict(begin=begin,end=end,items=end-begin,windows=len(window_lengths),
                               input_tokens=sum(window_lengths),padded_tokens=token_work(window_lengths,batch)))
        plan['kinds'][kind]=dict(items=len(texts),raw_tokens=len(flat),input_tokens=sum(c['input_tokens'] for c in chunks),
                                 padded_tokens=sum(c['padded_tokens'] for c in chunks),chunks=chunks)
        print(json.dumps(dict(kind=kind,**{k:v for k,v in plan['kinds'][kind].items() if k!='chunks'})),flush=True)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def worker(args):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name():
        raise ValueError('One visible T4 per encoder worker required')
    model=AutoModel.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    plan=json.loads((args.output/'TOKEN_PLAN.json').read_text())
    jobs={kind:[(i,chunk) for i,chunk in enumerate(value['chunks']) if i%2==args.rank]
          for kind,value in plan['kinds'].items()}
    tasks=[]
    for kind in ('names','queries'):
        if jobs[kind]:tasks.append((kind,*jobs[kind][0]))
    for kind in ('names','queries'):
        tasks.extend((kind,i,chunk) for i,chunk in jobs[kind][1:])
    profiles={};completed_work={'names':0,'queries':0};start=time.monotonic()
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:raise ValueError('Explicit deadline timezone required')
    tokens={kind:np.load(args.output/f'{kind}-tokens.npy',mmap_mode='r') for kind in jobs}
    offsets={kind:np.load(args.output/f'{kind}-offsets.npy',mmap_mode='r') for kind in jobs}
    special=plan['special_tokens']
    shard=args.output/f'rank-{args.rank}'
    shard.mkdir(exist_ok=True)
    for kind,index,chunk in tasks:
        path=shard/f'{kind}-{index:05d}.npy'
        metadata=path.with_suffix('.json')
        if metadata.exists():
            info=json.loads(metadata.read_text())
            if file_sha(path)!=info['sha256']:raise ValueError('Changed embedding checkpoint')
            profiles[kind]=dict(seconds=info['seconds'],work=chunk['padded_tokens'])
            completed_work[kind]+=chunk['padded_tokens']
            continue
        tick=time.monotonic();windows=[]
        for item in range(chunk['begin'],chunk['end']):
            body=tokens[kind][offsets[kind][item]:offsets[kind][item+1]]
            for order,window in enumerate(token_windows(body)):
                windows.append((item-chunk['begin'],order,[special['cls']]+window+[special['sep']]))
        windows.sort(key=lambda r:(len(r[2]),r[0],r[1]))
        sums=np.zeros((chunk['items'],model.config.hidden_size),dtype=np.float32)
        counts=np.zeros(chunk['items'],dtype=np.int32)
        batch_size=128 if kind=='names' else 1
        with torch.inference_mode():
            for begin in range(0,len(windows),batch_size):
                block=windows[begin:begin+batch_size];width=max(len(r[2]) for r in block)
                ids=torch.tensor([r[2]+[special['pad']]*(width-len(r[2])) for r in block],device='cuda')
                attention=torch.tensor([[1]*len(r[2])+[0]*(width-len(r[2])) for r in block],device='cuda')
                vectors=model(input_ids=ids,attention_mask=attention).last_hidden_state[:,0,:].float().cpu().numpy()
                for row,vector in zip(block,vectors):sums[row[0]]+=vector;counts[row[0]]+=1
        if not np.all(counts>0):raise ValueError('Encoder dropped input windows')
        vectors=sums/counts[:,None]
        norms=np.linalg.norm(vectors,axis=1,keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms==0):raise ValueError('Invalid biomedical embeddings')
        vectors=(vectors/norms).astype(np.float16 if kind=='names' else np.float32)
        np.save(path,vectors)
        elapsed=time.monotonic()-tick
        metadata.write_text(json.dumps(dict(**chunk,seconds=elapsed,sha256=file_sha(path)),indent=2)+'\n')
        profiles[kind]=dict(seconds=elapsed,work=chunk['padded_tokens'])
        completed_work[kind]+=chunk['padded_tokens']
        print(json.dumps(dict(rank=args.rank,kind=kind,completed_items=chunk['end'],seconds=elapsed)),flush=True)
        if len(profiles)==2:
            remaining={k:sum(c['padded_tokens'] for _,c in jobs[k])-completed_work[k] for k in jobs}
            forecast=sum(profiles[k]['seconds']/profiles[k]['work']*remaining[k] for k in jobs)*1.25
            available=max(0,min(args.usable_seconds-(time.monotonic()-start),(deadline-datetime.now(deadline.tzinfo)).total_seconds()))
            gate=dict(rank=args.rank,projected_remaining_seconds=forecast,available_seconds=available,fits=forecast<=available,
                      remaining_padded_tokens=remaining,measured=profiles)
            (shard/'COMPUTE_PLAN.json').write_text(json.dumps(gate,indent=2)+'\n')
            if not gate['fits']:
                print(json.dumps(gate),flush=True)
                (shard/'RESULT.json').write_text(json.dumps(dict(status='stopped_at_compute_gate'))+'\n')
                return
    (shard/'RESULT.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-start))+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline',required=True)
    parser.add_argument('--usable-seconds',type=float,required=True)
    parser.add_argument('--rank',type=int,choices=[0,1])
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.rank is not None:return worker(args)
    plan=prepare_tokens(args.prepared,args.model,args.output)
    processes=[];logs=[]
    for rank in (0,1):
        stream=(args.output/f'worker-{rank}.log').open('a');logs.append(stream)
        command=[sys.executable,'-u','-m','genlink.encode_retrieval','--prepared',str(args.prepared),
                 '--model',str(args.model),'--output',str(args.output),'--deadline',args.deadline,
                 '--usable-seconds',str(args.usable_seconds),'--rank',str(rank)]
        processes.append(subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank)),stdout=stream,stderr=subprocess.STDOUT))
    try:
        while any(process.poll() is None for process in processes):
            print(json.dumps([dict(rank=i,status=p.poll(),last=(args.output/f'worker-{i}.log').read_text(errors='replace')[-1200:])
                              for i,p in enumerate(processes)]),flush=True)
            time.sleep(20)
    finally:
        for stream in logs:stream.close()
    if any(process.returncode!=0 for process in processes):raise RuntimeError('Encoder worker failed; inspect preserved logs')
    if any(json.loads((args.output/f'rank-{rank}/RESULT.json').read_text())['status']!='complete' for rank in (0,1)):
        raise SystemExit('Encoding stopped at its measured full-scope compute gate; checkpoints retained')
    for kind,value in plan['kinds'].items():
        combined=np.lib.format.open_memmap(args.output/f'{kind}.npy',mode='w+',dtype=np.float16 if kind=='names' else np.float32,
                                          shape=(value['items'],768))
        for index,chunk in enumerate(value['chunks']):
            path=args.output/f'rank-{index%2}/{kind}-{index:05d}.npy'
            info=json.loads(path.with_suffix('.json').read_text())
            if file_sha(path)!=info['sha256']:raise ValueError('Changed checkpoint before collation')
            combined[chunk['begin']:chunk['end']]=np.load(path,mmap_mode='r')
        combined.flush()
        del combined
    result=dict(status='complete',names=plan['kinds']['names']['items'],query_vectors=plan['kinds']['queries']['items'],
                dimensions=768,files={f'{kind}.npy':file_sha(args.output/f'{kind}.npy') for kind in plan['kinds']})
    (args.output/'ENCODER_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
