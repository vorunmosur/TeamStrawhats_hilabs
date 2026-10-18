"""Rebuild the complete retrieval catalogue directly from the four original files."""
import argparse,hashlib,json,sys
from pathlib import Path
import pandas as pd
from .dictionaries import load_dictionary
from .prepare_retrieval import file_sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'SOURCE_MANIFEST.json')
    a=p.parse_args();a.project.mkdir(parents=True,exist_ok=True);expected=json.loads(a.manifest.read_text())
    for name,info in expected['files'].items():
        if file_sha(a.source/name)!=info['sha256']:raise ValueError(f'Original input changed: {name}')
    folder=a.project/'artifacts/g1';folder.mkdir(parents=True,exist_ok=True);legal={}
    for system in ('ICD10CM','SNOMEDCT_US'):
        names,preferred,report=load_dictionary(a.source,system)
        names.to_parquet(folder/f'{system}_names.parquet',index=False)
        preferred.to_parquet(folder/f'{system}_preferred.parquet',index=False)
        legal[system]=set(preferred.code);print(json.dumps(dict(system=system,**report)),flush=True)
    rows=pd.read_csv(a.source/'hackathon_train.csv',dtype=str,keep_default_na=False).drop(columns=['subconcept'],errors='ignore')
    rows['code']=rows.code.str.strip()
    keep=pd.Series([c in legal.get(s,set()) for s,c in zip(rows.coding_system,rows.code)],index=rows.index)
    invalid=rows.loc[~keep,['coding_system','code']].to_dict('records')
    if invalid!=[dict(coding_system='ICD10CM',code='L6.0')]*2:raise ValueError('Unexpected invalid source labels')
    rows=rows.loc[keep].reset_index(drop=True)
    if len(rows)!=9644 or hashlib.sha256('\n'.join(rows.entity).encode()).hexdigest()[:16]!='845350f70cad58ab':raise ValueError('Retained label alignment changed')
    folds=pd.read_csv(a.source/'termmap_folds.csv',dtype={'coding_system':str})
    if folds.row.tolist()!=list(range(len(rows))) or folds.coding_system.tolist()!=rows.coding_system.tolist():raise ValueError('Supplied folds do not align')
    rows['row']=range(len(rows));rows['fold']=folds.fold
    for system,counts in [('ICD10CM',[817,749,738]),('SNOMEDCT_US',[2398,2466,2476])]:
        if rows.loc[rows.coding_system.eq(system)].groupby('fold').size().tolist()!=counts:raise ValueError('Fixed fold counts changed')
    if rows.groupby(['entity','chunk_text']).fold.nunique().max()!=1:raise ValueError('Input instance crosses folds')
    rows.to_csv(a.project/'aligned_rows.csv',index=False)
    from .prepare_retrieval import main as prepare_main
    prepare_main(['--project',str(a.project),'--output',str(a.project/'artifacts/retrieval/prepared')])
    print('All original dictionaries and retained labels rebuilt with supplied folds.',flush=True)
if __name__=='__main__':main()
