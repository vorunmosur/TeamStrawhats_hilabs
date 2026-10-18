"""Assemble the public model state with the organizer's original dictionary tables."""
import argparse,json,zipfile
from pathlib import Path
import pandas as pd
from .dictionaries import load_dictionary
from .prepare_retrieval import file_sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.weights) as z:
        for member in z.infolist():
            name=Path(member.filename)
            if not name.parts or name.parts[0]!='runtime':raise ValueError('Unexpected model archive path')
            target=(a.output/Path(*name.parts[1:])).resolve()
            if not target.is_relative_to(a.output.resolve()):raise ValueError('Unsafe model archive path')
            if not member.is_dir():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
    folder=a.output/'prepared';parts=[];preferred=[]
    for system in ('ICD10CM','SNOMEDCT_US'):
        names,table,_=load_dictionary(a.source,system)
        names=names.loc[names.name_norm.ne('')].copy();names['system']=system;parts.append(names)
        fallback=names.sort_values(['preferred','name_norm'],ascending=[False,True]).drop_duplicates('code').set_index('code').name_norm
        table['system']=system;table['retrieval_preferred']=table.name_norm.where(table.name_norm.ne(''),table.code.map(fallback));preferred.append(table)
    aliases=pd.concat(parts,ignore_index=True);all_names=sorted(set(aliases.name_norm));lookup={n:i for i,n in enumerate(all_names)}
    pd.DataFrame(dict(name_id=range(len(all_names)),name=all_names)).to_parquet(folder/'names.parquet',index=False)
    aliases['name_id']=aliases.name_norm.map(lookup)
    aliases.drop(columns='name_norm').sort_values(['system','name_id','code']).reset_index(drop=True).to_parquet(folder/'aliases.parquet',index=False)
    concepts=pd.concat(preferred,ignore_index=True).sort_values(['system','code']).reset_index(drop=True);concepts['preferred_name_id']=concepts.retrieval_preferred.map(lookup)
    concepts.to_parquet(folder/'concepts.parquet',index=False)
    manifest=json.loads((a.output/'RUNTIME_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if file_sha(a.output/name)!=digest:raise ValueError(f'Runtime file differs from pinned model resource: {name}')
    print('Complete offline runtime verified. No raw clinical training text is required.',flush=True)
if __name__=='__main__':main()
