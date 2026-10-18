"""Choose dense versus learned ranking per system from complete OOF comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .rank_retrieval import CandidateCache,ordering,score_report,SYSTEMS

def select(models,candidates,prepared):
    models=Path(models);frame=pd.read_csv(models/'oof_predictions.csv',dtype={'gold':str,'prediction':str})
    if len(frame)!=9644 or frame.row.nunique()!=9644:raise ValueError('Complete OOF required for method selection')
    concepts=pd.read_parquet(Path(prepared)/'concepts.parquet')
    names={(r.system,r.code):r.retrieval_preferred for r in concepts.itertuples(index=False)}
    cache=CandidateCache(candidates);policy={}
    for system in SYSTEMS:
        mask=frame.system.eq(system);subset=frame.loc[mask]
        mode='dense' if float(subset.dense_correct.mean())>float(subset.correct.mean()) else 'ranker'
        policy[system]=dict(mode=mode,ranker_accuracy=float(subset.correct.mean()),dense_accuracy=float(subset.dense_correct.mean()))
        if mode=='dense':
            for i,row in subset.iterrows():
                codes,features,_=cache.get(row.query_id,system);scores=features[:,0];order=ordering(codes,scores);top=int(order[0])
                frame.at[i,'prediction']=codes[top];frame.at[i,'correct']=codes[top]==row.gold
                frame.at[i,'matched_alias']=names[(system,codes[top])]
                frame.at[i,'top20_correct']=row.gold in {codes[int(j)] for j in order[:20]}
                frame.at[i,'memory_recall']=row.base_recall
                frame.at[i,'ranking_score']=float(scores[top]);frame.at[i,'score_margin']=float(scores[top]-scores[int(order[1])])
                frame.at[i,'top20']=json.dumps([dict(code=codes[int(j)],score=float(scores[int(j)])) for j in order[:20]])
    report=score_report(frame);report['method_selection']=policy
    report['selection_caveat']='The same complete OOF comparison selected dense versus learned ranking per system. This is model-selection validation, not an untouched hidden test.'
    (models/'SELECTED_VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    (models/'DEPLOYMENT_SELECTION.json').write_text(json.dumps(policy,indent=2)+'\n')
    frame.to_csv(models/'deployment_oof_predictions.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report
