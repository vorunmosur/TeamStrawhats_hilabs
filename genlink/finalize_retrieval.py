"""Assemble the private offline runtime and verify complete repeated CLI predictions."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from .prepare_retrieval import file_sha
from .deployment_memory import text_digest,input_key
from .rank_retrieval import chunk_key


def copy_file(source,target):
    target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():
        if file_sha(source)!=file_sha(target):raise ValueError(f'Changed runtime source: {target.name}')
        return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=Path('/kaggle/working'))
    p.add_argument('--deadline',required=True);p.add_argument('--passes',type=int,default=2)
    a=p.parse_args();base=a.base;project=base/'termmap-genlink';runtime=base/'termmap-runtime';models=base/'retrieval-models-v2'
    if json.loads((models/'RANKER_RESULT.json').read_text())['status']!='complete':raise ValueError('All fold and final models required')
    from .select_deployment import select
    selected_report=select(models,base/'retrieval-candidates',project/'artifacts/retrieval/prepared')
    early=base/'termmap_validation_private.zip'
    with zipfile.ZipFile(early,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in models.iterdir():
            if f.is_file() and f.suffix in {'.json','.csv','.parquet','.joblib'}:z.write(f,Path('models')/f.name)
    print(json.dumps(dict(validation_file=early.name,bytes=early.stat().st_size,sha256=file_sha(early))),flush=True)
    prepared=project/'artifacts/retrieval/prepared'
    original_queries=pd.read_parquet(prepared/'queries.parquet')
    (runtime/'prepared').mkdir(parents=True,exist_ok=True)
    for f in prepared.iterdir():
        if f.is_file() and f.name not in {'queries.parquet','rows.parquet','CATALOG_MANIFEST.json'}:copy_file(f,runtime/'prepared'/f.name)
    safe_queries=pd.DataFrame(dict(query_id=original_queries.query_id,
        term_sha256=original_queries.term_norm.map(text_digest),
        context_key_sha256=[input_key(r.term_norm,r.context_norm) for r in original_queries.itertuples(index=False)],
        chunk_sha256=[text_digest(chunk_key(t)) for t in original_queries.chunk_text]))
    safe_queries.to_parquet(runtime/'prepared/queries.parquet',index=False)
    pd.read_parquet(prepared/'rows.parquet')[['row','fold','coding_system','code','query_id']].to_parquet(runtime/'prepared/rows.parquet',index=False)
    catalogue=json.loads((prepared/'CATALOG_MANIFEST.json').read_text())
    catalogue['source_catalogue_sha256']=file_sha(prepared/'CATALOG_MANIFEST.json')
    catalogue['deployment_privacy']='No clinical text: SHA-256 input keys, labels and learned support vectors only.'
    catalogue['files']={f.name:file_sha(f) for f in (runtime/'prepared').glob('*.parquet')}
    (runtime/'prepared/CATALOG_MANIFEST.json').write_text(json.dumps(catalogue,indent=2,sort_keys=True)+'\n')
    for name in ['names.npy','queries.npy','ENCODER_LOCK.json','ENCODER_RESULT.json','TOKEN_PLAN.json']:
        copy_file(base/'retrieval-embeddings'/name,runtime/'embeddings'/name)
    for name in ['char_tfidf.npz','char_vectorizer.joblib','LEXICAL_MANIFEST.json']:
        copy_file(base/'retrieval-lexical'/name,runtime/'lexical'/name)
    for name in ['ICD10CM_final_ranker.joblib','SNOMEDCT_US_final_ranker.joblib','RANKER_LOCK.json']:
        copy_file(models/name,runtime/'models'/name)
    # Reports are separate copies so finalized training provenance remains intact.
    (runtime/'models/VALIDATION_REPORT.json').write_text(json.dumps(selected_report,indent=2)+'\n')
    training_result=json.loads((models/'RANKER_RESULT.json').read_text())
    training_result['ranker_headline_min_accuracy']=training_result['headline_min_accuracy']
    training_result['headline_min_accuracy']=selected_report['headline_min_accuracy']
    training_result['method_selection']=selected_report['method_selection']
    (runtime/'models/RANKER_RESULT.json').write_text(json.dumps(training_result,indent=2)+'\n')
    copy_file(models/'VALIDATION_REPORT.json',runtime/'models/RANKER_VALIDATION_REPORT.json')
    copy_file(models/'DEPLOYMENT_SELECTION.json',runtime/'models/DEPLOYMENT_SELECTION.json')
    encoder=Path('/kaggle/temp/termmap-sapbert');source=json.loads((encoder/'MODEL_SOURCE.json').read_text())
    for name in list(source['files'])+['MODEL_SOURCE.json']:copy_file(encoder/name,runtime/'encoder'/name)
    packages=['torch','transformers','numpy','pandas','scipy','scikit-learn','pyarrow','huggingface-hub','safetensors','joblib','threadpoolctl']
    package_versions={name:importlib.metadata.version(name) for name in packages}
    (runtime/'requirements-resolved.txt').write_text(''.join(f'{name}=={version}\n' for name,version in package_versions.items()))
    files={f.relative_to(runtime).as_posix():file_sha(f) for f in sorted(runtime.rglob('*')) if f.is_file() and f.name!='RUNTIME_MANIFEST.json'}
    core=['normalize.py','retrieval.py','rank_retrieval.py','retrieval_cli.py','encode_retrieval.py','prepare_retrieval.py','input_adapter.py','build_candidates.py','deployment_memory.py']
    manifest=dict(status='complete',files=files,source_sha256={n:file_sha(project/'genlink'/n) for n in core},
        python=sys.version,packages=package_versions,bytes=sum((runtime/name).stat().st_size for name in files),
        data_policy='Private organizer resources and training memory. No external inference calls.',
        validation='Fixed supplied folds, 9644 labelled rows. Deployment rankers fitted on all own-system labels.')
    (runtime/'RUNTIME_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    archive=base/'termmap_runtime_private.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for f in sorted(runtime.rglob('*')):
                if f.is_file():z.write(f,Path('runtime')/f.relative_to(runtime))
    artifact=dict(file=archive.name,bytes=archive.stat().st_size,sha256=file_sha(archive),runtime_manifest_sha256=file_sha(runtime/'RUNTIME_MANIFEST.json'))
    (base/'RUNTIME_DOWNLOAD.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps(artifact),flush=True)
    public=base/'termmap_model_weights.zip'
    excluded={'prepared/names.parquet','prepared/aliases.parquet','prepared/concepts.parquet'}
    with zipfile.ZipFile(public,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in sorted(runtime.rglob('*')):
            if f.is_file() and f.relative_to(runtime).as_posix() not in excluded:z.write(f,Path('runtime')/f.relative_to(runtime))
    public_info=dict(file=public.name,bytes=public.stat().st_size,sha256=file_sha(public),
        scope='Ranker weights, frozen encoder/index weights, hashed example keys and support vectors; no raw clinical text or raw organizer dictionary tables.',
        setup='Rebuild the three excluded dictionary tables from the supplied original XLSX files with genlink.setup_runtime.')
    (base/'MODEL_WEIGHTS_DOWNLOAD.json').write_text(json.dumps(public_info,indent=2)+'\n')
    print(json.dumps(public_info),flush=True)
    output=base/'retrieval-verification';output.mkdir(exist_ok=True)
    queries=original_queries.sort_values('query_id')
    inputs=queries[['entity','chunk_text']].copy();inputs.insert(0,'input_id',[f'{i:08d}' for i in range(len(inputs))])
    inputs.to_csv(output/'all_inputs.csv',index=False)
    for rank in (0,1):inputs.iloc[rank::2].to_csv(output/f'inputs-{rank}.csv',index=False)
    env=dict(os.environ,PYTHONPATH=str(project),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from datetime import datetime
    deadline=datetime.fromisoformat(a.deadline)
    checks=[];durations=[]
    for run in range(1,a.passes+1):
        start=time.monotonic();processes=[];logs=[]
        for rank in (0,1):
            target=output/f'run-{run}-rank-{rank}.csv'
            if target.exists():continue
            log=(output/f'run-{run}-rank-{rank}.log').open('a');logs.append(log)
            cmd=[sys.executable,'-u','-m','genlink.retrieval_cli','predict','--artifacts',str(runtime),'--input',str(output/f'inputs-{rank}.csv'),
                 '--output',str(target),'--device','cuda','--candidate-cache',str(base/'retrieval-candidates'),'--deadline',a.deadline,'--usable-seconds','5400']
            processes.append(subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT))
        try:
            while any(process.poll() is None for process in processes):
                print(json.dumps([dict(run=run,rank=r,last=(output/f'run-{run}-rank-{r}.log').read_text(errors='replace')[-900:]) for r in (0,1)]),flush=True)
                time.sleep(20)
        finally:
            for log in logs:log.close()
        if any(process.returncode!=0 for process in processes):raise RuntimeError('Offline CLI verification incomplete; logs and runtime saved')
        frames=[pd.read_csv(output/f'run-{run}-rank-{r}.csv',dtype=str,keep_default_na=False) for r in (0,1)]
        combined=pd.concat(frames,ignore_index=True).sort_values('input_id').reset_index(drop=True)
        if len(combined)!=len(inputs) or not combined[inputs.columns].equals(inputs):raise ValueError('Input order or coverage mismatch')
        concepts=pd.read_parquet(runtime/'prepared/concepts.parquet')
        for s,c in [('ICD10CM','icd10cm_code'),('SNOMEDCT_US','snomed_code')]:
            if not combined[c].isin(set(concepts.loc[concepts.system.eq(s),'code'])).all():raise ValueError('Illegal code in complete CLI output')
        destination=output/f'predictions-run-{run}.csv';combined.to_csv(destination,index=False)
        latencies=[]
        for r in (0,1):latencies+=json.loads((output/f'run-{run}-rank-{r}.metrics.json').read_text())['latencies_seconds']
        checks.append(dict(run=run,rows=len(combined),both_systems=True,legal_codes=True,sha256=file_sha(destination),
            median_seconds=float(np.median(latencies)),p95_seconds=float(np.percentile(latencies,95)),
            independently_reencoded_inputs=False,independently_retrieved_candidates=False,full_frozen_candidate_cache=True))
        durations.append(time.monotonic()-start);print(json.dumps(checks[-1]),flush=True)
        (output/'VERIFICATION_PROGRESS.json').write_text(json.dumps(checks,indent=2)+'\n')
        remaining=max(0,(deadline-datetime.now(deadline.tzinfo)).total_seconds())
        if run<a.passes and durations[-1]*1.25>remaining:raise SystemExit('Repeat pass does not fit deadline; first complete predictions and runtime preserved')
    if len(checks)!=2 or checks[0]['sha256']!=checks[1]['sha256']:raise ValueError('Complete final-ranking predictions are not byte-identical')
    known={(r.term_norm,r.context_norm) for r in original_queries.itertuples(index=False)}
    new_cases=original_queries.loc[[bool(r.context_norm) and (r.term_norm,'') not in known for r in original_queries.itertuples(index=False)],['entity']].head(10).copy()
    new_cases.insert(0,'input_id',[f'fresh-{i}' for i in range(len(new_cases))]);new_cases['chunk_text']=''
    new_cases.to_csv(output/'fresh_inputs.csv',index=False)
    fresh=[]
    for run in (1,2):
        target=output/f'fresh_predictions_{run}.csv'
        if not target.exists():
            subprocess.run([sys.executable,'-m','genlink.retrieval_cli','predict','--artifacts',str(runtime),
                '--input',str(output/'fresh_inputs.csv'),'--output',str(target),'--device','cuda','--reencode'],
                env=dict(env,CUDA_VISIBLE_DEVICES='0'),check=True)
        fresh.append(file_sha(target))
    if fresh[0]!=fresh[1]:raise ValueError('Fresh-input encoder/retrieval/ranker integration is not deterministic')
    fresh_test=dict(inputs=len(new_cases),independent_fresh_runs=2,sha256=fresh[0],deterministic=True,
        scope='Contextless variants excluded from the known-query cache; integration test, not an accuracy benchmark.')

    report=dict(status='complete',deterministic=True,fresh_input_integration=fresh_test,full_passes=checks,all_inputs=len(inputs),runtime=artifact,
        caveat='All-input final-ranking repeatability with complete frozen label-free candidates. Not independent retrieval/encoder repeatability, not validation accuracy. OOF report is separate. Cached latency is not fresh-input latency.')
    (output/'VERIFICATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    reports=base/'termmap_reports_private.zip'
    with zipfile.ZipFile(reports,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for folder,prefix in [(models,'models'),(output,'verification')]:
            for f in folder.iterdir():
                if f.is_file() and f.suffix in {'.json','.csv','.parquet','.log','.jsonl','.joblib'}:z.write(f,Path(prefix)/f.name)
        for name in ['RUNTIME_DOWNLOAD.json','MODEL_WEIGHTS_DOWNLOAD.json']:z.write(base/name,name)
    print(json.dumps(dict(reports_file=reports.name,bytes=reports.stat().st_size,sha256=file_sha(reports))),flush=True)

if __name__=='__main__':main()
