"""Complete label-free retrieval accuracy audit on every retained label."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from .rank_retrieval import CandidateCache,ordering,SYSTEMS
from .input_adapter import labelled_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=labelled_rows(a.prepared/'rows.parquet');cache=CandidateCache(a.candidates);records=[]
    for row in rows.itertuples(index=False):
        codes,f,_=cache.get(row.query_id,row.system)
        record=dict(row=row.row,fold=row.fold,system=row.system,gold=row.code,candidate_recall=row.code in codes,candidates=len(codes))
        for name,scores in [('dense',f[:,0]),('lexical',f[:,2]),('rrf',f[:,22:25].sum(axis=1)),('hybrid',f[:,0]+f[:,2])]:
            order=ordering(codes,scores);record[name]=codes[int(order[0])]==row.code;record[name+'_top20']=row.code in {codes[int(i)] for i in order[:20]}
        records.append(record)
    frame=pd.DataFrame(records);a.output.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output/'retrieval_baselines.csv',index=False)
    report={s:{k:float(g[k].mean()) for k in ['candidate_recall','dense','lexical','rrf','hybrid','dense_top20','lexical_top20','rrf_top20','hybrid_top20']} for s in SYSTEMS for g in [frame.loc[frame.system.eq(s)]]}
    report['rows']=len(frame);report['scope']='All retained labels, dictionary-only candidate search. No ranker or training-label memory.'
    (a.output/'RETRIEVAL_BASELINES.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
