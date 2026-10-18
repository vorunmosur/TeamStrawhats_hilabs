"""Importable team interface backed by the completed offline runtime."""
from functools import lru_cache
import os

@lru_cache(maxsize=1)
def _engine(artifacts):
    from .retrieval_cli import MappingEngine
    return MappingEngine(artifacts)

def gen_topk(entity,chunk_text,system,k=20,*,artifacts=None):
    """Historical interface name; scores come from retrieval/context ranking."""
    if system not in ('ICD10CM','SNOMEDCT_US'):raise ValueError('Unknown coding system')
    if not 1<=k<=20:raise ValueError('k must be between 1 and 20')
    root=str(artifacts or os.environ.get('TERMMAP_ARTIFACTS','artifacts/runtime'))
    result=_engine(root).predict(entity,chunk_text)[system]['top20'][:k]
    return [(row['code'],row['ranking_score'],row['name']) for row in result]

termmap_topk=gen_topk
