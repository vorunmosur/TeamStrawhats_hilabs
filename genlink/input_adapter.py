"""Validated CSV input contract, preserving identifiers and explicit context."""
def input_rows(path, chunks=None):
    import pandas as pd
    rows=pd.read_csv(path,dtype=str,keep_default_na=False)
    entity='entity' if 'entity' in rows.columns else 'clinical_entity'
    if entity not in rows.columns:
        raise ValueError('Input needs entity or clinical_entity')
    if 'chunk_text' not in rows.columns:
        if chunks is None or 'chunk_id' not in rows.columns:
            raise ValueError('Provide chunk_text, or chunk_id with --chunks; use explicit empty chunk_text for contextless input')
        lookup=pd.read_csv(chunks,dtype=str,keep_default_na=False)
        if not {'chunk_id','chunk_text'}.issubset(lookup.columns) or lookup.chunk_id.duplicated().any():
            raise ValueError('Chunk lookup must contain one chunk_text per chunk_id')
        mapping=lookup.set_index('chunk_id').chunk_text.to_dict()
        missing=set(rows.chunk_id)-mapping.keys()
        if missing:
            raise ValueError(f'Missing context for {len(missing)} chunk identifiers')
        rows['chunk_text']=rows.chunk_id.map(mapping)
    return rows,entity


def labelled_rows(path):
    """Adapt the original coding_system column without changing its values/folds."""
    import pandas as pd
    rows=pd.read_parquet(path)
    if 'system' not in rows.columns and 'coding_system' in rows.columns:
        rows=rows.rename(columns={'coding_system':'system'})
    required={'row','fold','query_id','system','code'}
    if not required.issubset(rows.columns):raise ValueError('Label table schema is incomplete')
    if rows.row.duplicated().any() or not rows.fold.isin([0,1,2]).all():raise ValueError('Invalid fixed folds/row ids')
    if not rows.system.isin(['ICD10CM','SNOMEDCT_US']).all():raise ValueError('Unexpected coding system')
    return rows
