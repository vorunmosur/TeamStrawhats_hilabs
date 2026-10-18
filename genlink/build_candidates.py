"""Full label-free hybrid retrieval for every input instance and both systems."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .prepare_retrieval import file_sha
from .retrieval import HybridRetriever,FEATURES,build_lexical


class CandidateStore:
    def __init__(self,path):
        self.connection=sqlite3.connect(path)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS candidates (query_id INTEGER, system TEXT, codes TEXT NOT NULL, features BLOB NOT NULL, alias_ids BLOB NOT NULL, PRIMARY KEY(query_id,system))')

    def get(self,query_id,system):
        row=self.connection.execute('SELECT codes,features,alias_ids FROM candidates WHERE query_id=? AND system=?',(int(query_id),system)).fetchone()
        if row is None:return None
        codes=json.loads(row[0]);matrix=np.frombuffer(row[1],dtype='<f4').reshape(len(codes),len(FEATURES)).copy()
        aliases=np.frombuffer(row[2],dtype='<i4').copy()
        if len(aliases)!=len(codes):raise ValueError('Corrupt candidate checkpoint')
        return codes,matrix,aliases

    def put(self,query_id,system,codes,features,aliases):
        if len(codes)!=len(set(codes)) or features.shape!=(len(codes),len(FEATURES)):
            raise ValueError('Invalid candidate batch')
        with self.connection:
            self.connection.execute('INSERT INTO candidates VALUES (?,?,?,?,?)',
                (int(query_id),system,json.dumps(codes),np.asarray(features,dtype='<f4').tobytes(),np.asarray(aliases,dtype='<i4').tobytes()))

    def close(self):self.connection.close()


def worker(args):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    index=HybridRetriever(args.prepared,args.embeddings,args.lexical,device='cuda')
    queries=pd.read_parquet(args.prepared/'queries.parquet')
    queries=queries.loc[queries.query_id.mod(2).eq(args.rank)]
    vectors=np.load(args.embeddings/'queries.npy',mmap_mode='r').reshape(-1,2,768)
    store=CandidateStore(args.output/f'candidates-rank-{args.rank}.sqlite')
    start=time.monotonic();measured=0;done=0;status='running'
    deadline=datetime.fromisoformat(args.deadline)
    try:
        for query in queries.itertuples(index=False):
            existing=[store.get(query.query_id,system) is not None for system in ('ICD10CM','SNOMEDCT_US')]
            if all(existing):done+=1;continue
            for system,present in zip(('ICD10CM','SNOMEDCT_US'),existing):
                if present:continue
                codes,features,aliases=index.candidates(system,query.term_norm,query.context_norm,
                                                       vectors[query.query_id,0],vectors[query.query_id,1])
                store.put(query.query_id,system,codes,features,aliases)
            done+=1;measured+=1
            if measured==2 or measured%250==0 or done==len(queries):
                elapsed=time.monotonic()-start
                available=max(0,min(args.usable_seconds-elapsed,(deadline-datetime.now(deadline.tzinfo)).total_seconds()))
                forecast=elapsed/measured*(len(queries)-done)*1.25
                plan=dict(rank=args.rank,completed_queries=done,scope_queries=len(queries),both_systems=True,
                          measured_seconds=elapsed,projected_remaining_seconds=forecast,available_seconds=available,fits=forecast<=available)
                (args.output/f'COMPUTE_PLAN_rank_{args.rank}.json').write_text(json.dumps(plan,indent=2)+'\n')
                print(json.dumps(plan),flush=True)
                if not plan['fits']:status='stopped_at_compute_gate';break
        if done==len(queries):status='complete'
        (args.output/f'RESULT_rank_{args.rank}.json').write_text(json.dumps(dict(status=status,queries=done))+'\n')
    finally:store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--embeddings',type=Path,required=True)
    parser.add_argument('--lexical',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline',required=True)
    parser.add_argument('--usable-seconds',type=float,required=True)
    parser.add_argument('--rank',type=int,choices=[0,1])
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    args.output.mkdir(exist_ok=True,parents=True)
    if args.rank is not None:return worker(args)
    result=json.loads((args.embeddings/'ENCODER_RESULT.json').read_text())
    if result['status']!='complete':raise ValueError('Complete dictionary and query embeddings required')
    identity=dict(catalogue_sha256=file_sha(args.prepared/'CATALOG_MANIFEST.json'),
                  encoder_result_sha256=file_sha(args.embeddings/'ENCODER_RESULT.json'),features=FEATURES,
                  retrieval_source_sha256=file_sha(Path(__file__).with_name('retrieval.py')),
                  worker_source_sha256=file_sha(Path(__file__)))
    lock=args.output/'CANDIDATE_LOCK.json'
    if lock.exists() and json.loads(lock.read_text())!=identity:raise ValueError('Candidate output configuration changed')
    lock.write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    build_lexical(args.prepared,args.lexical)
    processes=[];logs=[]
    for rank in (0,1):
        log=(args.output/f'worker-{rank}.log').open('a');logs.append(log)
        command=[sys.executable,'-u','-m','genlink.build_candidates','--prepared',str(args.prepared),
                 '--embeddings',str(args.embeddings),'--lexical',str(args.lexical),'--output',str(args.output),
                 '--deadline',args.deadline,'--usable-seconds',str(args.usable_seconds),'--rank',str(rank)]
        processes.append(subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT))
    try:
        while any(p.poll() is None for p in processes):
            print(json.dumps([dict(rank=i,status=p.poll(),last=(args.output/f'worker-{i}.log').read_text(errors='replace')[-1300:]) for i,p in enumerate(processes)]),flush=True)
            time.sleep(20)
    finally:
        for log in logs:log.close()
    if any(p.returncode!=0 for p in processes):raise RuntimeError('Candidate worker failed; logs retained')
    if any(json.loads((args.output/f'RESULT_rank_{rank}.json').read_text())['status']!='complete' for rank in (0,1)):
        raise SystemExit('Full candidate retrieval stopped at its measured budget gate')
    expected=len(pd.read_parquet(args.prepared/'queries.parquet'))*2
    total=0
    for rank in (0,1):
        store=CandidateStore(args.output/f'candidates-rank-{rank}.sqlite')
        if store.connection.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Corrupt candidate store')
        total+=store.connection.execute('SELECT count(*) FROM candidates').fetchone()[0]
        store.close()
    if total!=expected:raise ValueError('Missing query/system candidate results')
    (args.output/'CANDIDATE_RESULT.json').write_text(json.dumps(dict(status='complete',query_system_pairs=total,
        features=FEATURES,files={f'candidates-rank-{rank}.sqlite':file_sha(args.output/f'candidates-rank-{rank}.sqlite') for rank in (0,1)}),indent=2)+'\n')
    print(f'Complete: {total} input-instance/system pairs retrieved without labels.',flush=True)


if __name__=='__main__':main()
