"""Deployment memory preserves learned features using hashes instead of clinical text."""
import hashlib,json
from .rank_retrieval import TrainingMemory

def text_digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def input_key(term,context):return text_digest(json.dumps([term,context],ensure_ascii=False,separators=(',',':')))

class DeploymentMemory(TrainingMemory):
    def __init__(self,rows,queries,vectors,system,details):
        adapted=queries[['query_id','term_sha256','chunk_sha256']].rename(columns={'term_sha256':'term_norm','chunk_sha256':'chunk_text'})
        super().__init__(rows,adapted,vectors,system,details)
    def search(self,term,context,term_vector,context_vector,exclude_chunk=None):
        if exclude_chunk is not None:raise ValueError('Hashed deployment memory is for prediction, not fitting')
        return super().search(text_digest(term),context,term_vector,context_vector)
