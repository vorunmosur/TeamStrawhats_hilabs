"""Fold-isolated learned example memory and lightweight context ranking."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .retrieval import HybridRetriever,FEATURES,ALL_FEATURES
from .build_candidates import CandidateStore
from .prepare_retrieval import file_sha
from .input_adapter import labelled_rows

SYSTEMS=('ICD10CM','SNOMEDCT_US')
PARAMETERS=dict(max_iter=120,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=20,
                l2_regularization=5.,max_bins=127,early_stopping=False,random_state=42)

def chunk_key(text):return re.sub('[^a-z0-9]+','',text.lower())

class TrainingMemory:
    def __init__(self,rows,queries,vectors,system,details):
        self.system=system
        self.rows=rows.loc[rows.system.eq(system)].reset_index(drop=True)
        self.codes=sorted(set(self.rows.code));self.code_lookup={c:i for i,c in enumerate(self.codes)}
        self.labels=self.rows.code.map(self.code_lookup).to_numpy()
        self.query_ids=self.rows.query_id.to_numpy(dtype=int)
        self.vectors=np.asarray(vectors[self.query_ids],dtype=np.float32)
        self.vectors/=np.maximum(np.linalg.norm(self.vectors,axis=2,keepdims=True),1e-12)
        q=queries.set_index('query_id')
        self.terms=q.loc[self.query_ids].term_norm.to_numpy()
        self.groups=np.asarray([chunk_key(t) for t in q.loc[self.query_ids].chunk_text])
        self.code_groups=[c[:3] if system=='ICD10CM' else details[(system,c)]['tag'] for c in self.codes]
        self.details=details

    def search(self,term,context,term_vector,context_vector,exclude_chunk=None):
        eligible=np.ones(len(self.rows),dtype=bool)
        if exclude_chunk is not None:eligible=self.groups!=chunk_key(exclude_chunk)
        labels=self.labels[eligible]
        frequency=np.bincount(labels,minlength=len(self.codes)).astype(float)
        exact=np.bincount(self.labels[eligible & (self.terms==term)],minlength=len(self.codes)).astype(float)
        exact/=max(1,float(exact.sum()))
        group_frequency=Counter()
        for i,count in enumerate(frequency):group_frequency[self.code_groups[i]]+=count
        similarity=np.einsum('nd,d->n',self.vectors[:,0],term_vector)
        context_similarity=np.einsum('nd,d->n',self.vectors[:,1],context_vector)
        best=np.full(len(self.codes),-1.,dtype=np.float32);best_context=best.copy()
        np.maximum.at(best,labels,similarity[eligible]);np.maximum.at(best_context,labels,context_similarity[eligible])
        eligible_codes=np.flatnonzero(frequency>0)
        term_order=eligible_codes[np.lexsort((np.asarray(self.codes)[eligible_codes],-best[eligible_codes]))]
        ctx_order=eligible_codes[np.lexsort((np.asarray(self.codes)[eligible_codes],-best_context[eligible_codes]))]
        term_ranks={int(i):1/(60+r) for r,i in enumerate(term_order,1)}
        ctx_ranks={int(i):1/(60+r) for r,i in enumerate(ctx_order,1)}
        extra=sorted(set(self.codes[i] for i in term_order[:32])|set(self.codes[i] for i in ctx_order[:16]))
        def features(code):
            i=self.code_lookup.get(code)
            group=code[:3] if self.system=='ICD10CM' else self.details[(self.system,code)]['tag']
            if i is None:return [0.,np.log1p(group_frequency[group]),-1.,-1.,0.,0.,0.]
            return [np.log1p(frequency[i]),np.log1p(group_frequency[group]),best[i],best_context[i],exact[i],term_ranks.get(i,0.),ctx_ranks.get(i,0.)]
        return extra,features

class CandidateCache:
    def __init__(self,path):self.stores=[CandidateStore(Path(path)/f'candidates-rank-{r}.sqlite') for r in (0,1)]
    def get(self,query_id,system):
        result=self.stores[int(query_id)%2].get(query_id,system)
        if result is None:raise ValueError('Missing full-scope candidates')
        return result


def assemble(index,memory,query,vectors,base,exclude_chunk=None,gold=None):
    term_vector,context_vector=vectors
    extras,memory_features=memory.search(query.term_norm,query.context_norm,term_vector,context_vector,exclude_chunk)
    codes,features,aliases=base
    base_lookup={c:i for i,c in enumerate(codes)}
    candidates=sorted(set(codes)|set(extras)|({gold} if gold is not None else set()))
    matrix=[];alias_ids=[]
    for code in candidates:
        if code in base_lookup:
            i=base_lookup[code];f=features[i];alias=int(aliases[i])
        else:
            f,alias=index.code_feature(memory.system,code,query.term_norm,query.context_norm,term_vector,context_vector)
        matrix.append(np.r_[f,memory_features(code)]);alias_ids.append(alias)
    result=np.asarray(matrix,dtype=np.float32)
    if result.shape!=(len(candidates),len(ALL_FEATURES)) or not np.isfinite(result).all():raise ValueError('Bad rank features')
    return candidates,result,np.asarray(alias_ids,dtype=np.int64)


def ordering(codes,scores):return np.lexsort((np.asarray(codes),-np.asarray(scores)))


def score_report(predictions):
    report={}
    for system in SYSTEMS:
        rows=predictions.loc[predictions.system.eq(system)]
        report[system]=dict(rows=len(rows),accuracy=float(rows.correct.mean()),
            top20_accuracy=float(rows.top20_correct.mean()),hybrid_candidate_recall=float(rows.base_recall.mean()),augmented_candidate_recall=float(rows.memory_recall.mean()),
            rare_seen_rows=int(rows.training_code_count.between(1,5).sum()),
            rare_seen_accuracy=float(rows.loc[rows.training_code_count.between(1,5),'correct'].mean()),unseen_code_rows=int(rows.unseen.sum()),unseen_code_accuracy=float(rows.loc[rows.unseen,'correct'].mean()),
            dense_accuracy=float(rows.dense_correct.mean()),lexical_accuracy=float(rows.lexical_correct.mean()),
            rrf_accuracy=float(rows.rrf_correct.mean()),fold_accuracy={str(f):float(g.correct.mean()) for f,g in rows.groupby('fold')})
    report['headline_min_accuracy']=min(report[s]['accuracy'] for s in SYSTEMS)
    report['meets_95_percent']=report['headline_min_accuracy']>=.95
    report['evaluation']='Supplied fixed-fold out-of-fold validation; no official hidden evaluation file was available.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','embeddings','lexical','candidates','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',required=True)
    parser.add_argument('--usable-seconds',type=float,default=5400)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if json.loads((args.candidates/'CANDIDATE_RESULT.json').read_text())['status']!='complete':raise ValueError('Complete candidates required')
    identity=dict(features=ALL_FEATURES,parameters=PARAMETERS,source_sha256=file_sha(Path(__file__)),
                  retrieval_sha256=file_sha(Path(__file__).with_name('retrieval.py')),
                  candidate_result_sha256=file_sha(args.candidates/'CANDIDATE_RESULT.json'))
    lock=args.output/'RANKER_LOCK.json'
    if lock.exists() and json.loads(lock.read_text())!=identity:raise ValueError('Ranker configuration changed')
    lock.write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    index=HybridRetriever(args.prepared,args.embeddings,args.lexical)
    queries=pd.read_parquet(args.prepared/'queries.parquet').set_index('query_id',drop=False)
    rows=labelled_rows(args.prepared/'rows.parquet')
    if len(rows)!=9644:raise ValueError('All 9644 retained labels required')
    vectors=np.load(args.embeddings/'queries.npy',mmap_mode='r').reshape(-1,2,768)
    vectors=np.array(vectors,dtype=np.float32,copy=True);vectors/=np.maximum(np.linalg.norm(vectors,axis=2,keepdims=True),1e-12)
    cache=CandidateCache(args.candidates)
    deadline=datetime.fromisoformat(args.deadline);start=time.monotonic();completed_jobs=0
    for fold in (0,1,2,'final'):
        train=rows if fold=='final' else rows.loc[rows.fold.ne(fold)]
        test=rows.iloc[:0] if fold=='final' else rows.loc[rows.fold.eq(fold)]
        for system in SYSTEMS:
            tag=f'{system}_{fold}';finish=args.output/f'{tag}_DONE.json'
            if finish.exists():completed_jobs+=1;continue
            job_start=time.monotonic()
            memory=TrainingMemory(train,queries.reset_index(drop=True),vectors,system,index.details)
            train_system=train.loc[train.system.eq(system)]
            training_counts=Counter(train_system.code)
            data=[];targets=[];weights=[];offsets=[];position=0;injected=0
            for number,row in enumerate(train_system.itertuples(index=False),1):
                query=queries.loc[row.query_id]
                base=cache.get(row.query_id,system)
                codes,matrix,_=assemble(index,memory,query,vectors[row.query_id],base,exclude_chunk=row.chunk_text,gold=row.code)
                injected+=row.code not in base[0]
                y=np.asarray([code==row.code for code in codes],dtype=np.int8)
                if y.sum()!=1 or len(codes)<2:raise ValueError('Training gold or negatives missing')
                weight=np.where(y,.5,.5/(len(codes)-1)).astype(np.float32)
                data.append(matrix);targets.append(y);weights.append(weight);offsets.append((position,position+len(codes)))
                position+=len(codes)
                if number%1000==0:print(json.dumps(dict(stage='ranker_features',job=tag,rows=number,total=len(train_system))),flush=True)
                if datetime.now(deadline.tzinfo)>=deadline:raise SystemExit('Deadline reached; completed checkpoints retained')
            x=np.concatenate(data);y=np.concatenate(targets);w=np.concatenate(weights)
            del data,targets,weights
            model=HistGradientBoostingClassifier(**PARAMETERS)
            model.fit(x,y,sample_weight=w)
            del x,y,w
            joblib.dump(model,args.output/f'{tag}_ranker.joblib')
            records=[]
            for row in test.loc[test.system.eq(system)].itertuples(index=False):
                query=queries.loc[row.query_id];base=cache.get(row.query_id,system)
                codes,matrix,aliases=assemble(index,memory,query,vectors[row.query_id],base)
                scores=model.predict_proba(matrix)[:,1];order=ordering(codes,scores);top=int(order[0]);second=int(order[1])
                base_codes,base_features,_=base
                dense=base_codes[int(ordering(base_codes,base_features[:,0])[0])]
                lexical=base_codes[int(ordering(base_codes,base_features[:,2])[0])]
                rrf=base_codes[int(ordering(base_codes,base_features[:,22:25].sum(axis=1))[0])]
                records.append(dict(row=row.row,fold=fold,query_id=row.query_id,system=system,gold=row.code,prediction=codes[top],
                    correct=codes[top]==row.code,base_recall=row.code in base_codes,memory_recall=row.code in codes,
                    unseen=row.code not in memory.code_lookup,training_code_count=int(training_counts[row.code]),dense_correct=dense==row.code,lexical_correct=lexical==row.code,rrf_correct=rrf==row.code,
                    ranking_score=float(scores[top]),score_margin=float(scores[top]-scores[second]),matched_alias=index.names[int(aliases[top])],
                    top20_correct=row.code in {codes[int(i)] for i in order[:20]},
                    top20=json.dumps([dict(code=codes[int(i)],score=float(scores[int(i)])) for i in order[:20]])))
            pd.DataFrame(records).to_parquet(args.output/f'{tag}_predictions.parquet',index=False)
            completed_jobs+=1;elapsed=time.monotonic()-start;job_seconds=time.monotonic()-job_start
            # Forecast the complete remaining labelled work, using the slowest observed per-row job rate.
            previous=[json.loads(p.read_text()) for p in args.output.glob('*_DONE.json')]
            workload=len(train_system)+len(test.loc[test.system.eq(system)])
            rate=max([job_seconds/workload]+[r['seconds']/r['workload_rows'] for r in previous])
            completed_by_system=Counter([r['system'] for r in previous]+[system])
            remaining_rows=sum(int(rows.system.eq(s).sum())*(4-completed_by_system[s]) for s in SYSTEMS)
            remaining=rate*remaining_rows*1.25
            available=max(0,min(args.usable_seconds-elapsed,(deadline-datetime.now(deadline.tzinfo)).total_seconds()))
            report=dict(job=tag,system=system,workload_rows=workload,seconds=job_seconds,train_rows=len(train_system),training_only_gold_injections=injected,
                        candidate_training_pairs=position,completed_jobs=completed_jobs,forecast_remaining_seconds=remaining,
                        available_seconds=available,fits=remaining<=available,
                        validation_accuracy=float(np.mean([r['correct'] for r in records])) if records else None)
            finish.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
            if fold!='final':
                partial=pd.concat([pd.read_parquet(p) for p in args.output.glob('*_predictions.parquet') if 'final' not in p.name],ignore_index=True)
                partial.to_csv(args.output/'oof_progress.csv',index=False)
                if len(partial)==len(rows):
                    summary=score_report(partial)
                    (args.output/'VALIDATION_REPORT.json').write_text(json.dumps(summary,indent=2)+'\n')
                    print(json.dumps(summary,indent=2),flush=True)
            if not report['fits'] and completed_jobs<8:raise SystemExit('Ranker stopped at compute gate; completed folds retained')
    predictions=pd.concat([pd.read_parquet(args.output/f'{s}_{f}_predictions.parquet') for f in (0,1,2) for s in SYSTEMS],ignore_index=True).sort_values('row')
    if len(predictions)!=len(rows) or predictions.row.nunique()!=len(rows):raise ValueError('Incomplete OOF row coverage')
    predictions.to_csv(args.output/'oof_predictions.csv',index=False)
    for system in SYSTEMS:
        ranked=[]
        for record in predictions.loc[predictions.system.eq(system)].itertuples(index=False):
            for rank,candidate in enumerate(json.loads(record.top20),1):
                ranked.append(dict(row=record.row,code=candidate['code'],rank=rank,score=candidate['score'],
                    name=index.details[(system,candidate['code'])]['name'],fold=record.fold))
        pd.DataFrame(ranked).to_parquet(args.output/f'retrieval_oof_{system}.parquet',index=False)
    predictions.loc[~predictions.correct].merge(rows[['row','entity','chunk_text']],on='row').to_csv(args.output/'error_analysis.csv',index=False)
    report=score_report(predictions)
    (args.output/'VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'RANKER_RESULT.json').write_text(json.dumps(dict(status='complete',rows=len(rows),headline_min_accuracy=report['headline_min_accuracy'],
        models={s:file_sha(args.output/f'{s}_final_ranker.joblib') for s in SYSTEMS}),indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
