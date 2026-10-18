"""Full label-free concept catalogue and instance queries for biomedical retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .normalize import context_window,level1
from .prompts import SYSTEMS

SAPBERT_MODEL='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
SAPBERT_REVISION='090663c3ae57bf35ffe4d0d468a2a88d03051a4d'

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    parts=[]
    preferred=[]
    for system in SYSTEMS:
        names=pd.read_parquet(args.project/f'artifacts/g1/{system}_names.parquet')
        names=names.loc[names.name_norm.ne('')].copy()
        names['system']=system
        parts.append(names)
        table=pd.read_parquet(args.project/f'artifacts/g1/{system}_preferred.parquet')
        table['system']=system
        # Every concept has a readable alias, including the symbol-only preferred names.
        fallback=names.sort_values(['preferred','name_norm'],ascending=[False,True]).drop_duplicates('code').set_index('code').name_norm
        table['retrieval_preferred']=table.name_norm.where(table.name_norm.ne(''),table.code.map(fallback))
        assert table.retrieval_preferred.ne('').all()
        preferred.append(table)
    aliases=pd.concat(parts,ignore_index=True)
    all_names=sorted(set(aliases.name_norm))
    lookup={name:i for i,name in enumerate(all_names)}
    name_table=pd.DataFrame(dict(name_id=range(len(all_names)),name=all_names))
    aliases['name_id']=aliases.name_norm.map(lookup)
    aliases=aliases.drop(columns='name_norm').sort_values(['system','name_id','code']).reset_index(drop=True)
    concepts=pd.concat(preferred,ignore_index=True).sort_values(['system','code']).reset_index(drop=True)
    concepts['preferred_name_id']=concepts.retrieval_preferred.map(lookup)
    assert len(concepts)==480306 and not aliases.duplicated(['system','name_id','code']).any()
    rows=pd.read_csv(args.project/'aligned_rows.csv',dtype=str,keep_default_na=False)
    rows['row']=rows.row.astype(int);rows['fold']=rows.fold.astype(int)
    instances=rows[['entity','chunk_text']].drop_duplicates().reset_index(drop=True)
    instances['query_id']=instances.index
    instances['term_norm']=instances.entity.map(level1)
    instances['context_norm']=[context_window(r.entity,r.chunk_text) for r in instances.itertuples(index=False)]
    instances['context_query']=[term if not context else term+' [SEP] '+context
                                for term,context in zip(instances.term_norm,instances.context_norm)]
    rows=rows.merge(instances[['entity','chunk_text','query_id']],on=['entity','chunk_text'],how='left',validate='many_to_one')
    assert len(rows)==9644 and len(instances)==7522
    assert rows.groupby('query_id').fold.nunique().eq(1).all()
    for filename,table in [('names.parquet',name_table),('aliases.parquet',aliases),
                            ('concepts.parquet',concepts),('queries.parquet',instances),('rows.parquet',rows)]:
        table.to_parquet(args.output/filename,index=False)
    manifest=dict(model_id=SAPBERT_MODEL,model_revision=SAPBERT_REVISION,license='apache-2.0',
        names=len(name_table),alias_code_pairs=len(aliases),concepts=len(concepts),queries=len(instances),labelled_rows=len(rows),
        input_features=['entity','chunk_text'],synthetic_examples=0,
        systems=concepts.system.value_counts().to_dict(),
        files={p.name:file_sha(p) for p in sorted(args.output.glob('*.parquet'))},
        scope='Every supplied dictionary concept with all nonempty normalized aliases. Empty aliases have readable alternatives.',
        long_input_policy='Tokenize without truncation; encode overlapping windows and average CLS vectors for inputs exceeding model context.')
    (args.output/'CATALOG_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
