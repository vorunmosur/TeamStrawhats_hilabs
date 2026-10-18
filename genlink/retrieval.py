"""Label-free hybrid candidates and transparent clinical/context features."""
from collections import defaultdict
from functools import lru_cache
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .prepare_retrieval import file_sha

FEATURES=['dense_term','dense_context','lexical','exact_name','preferred_alias','term_token_coverage',
          'name_token_coverage','context_token_coverage','length_ratio','term_words','name_words',
          'left_right_conflict','bilateral_mismatch','acute_chronic_conflict','missing_numbers',
          'extra_numbers','without_mismatch','history_mismatch','context_laterality_conflict',
          'context_acuity_conflict','context_number_coverage','semantic_tag','dense_rank','context_rank','lexical_rank']
MEMORY_FEATURES=['log_code_frequency','log_group_frequency','nearest_term_for_code','nearest_context_for_code',
                 'exact_training_term_probability','nearest_term_rank','nearest_context_rank']
ALL_FEATURES=FEATURES+MEMORY_FEATURES


def stable_top(scores,ids,k,positive_only=False):
    scores=np.asarray(scores);ids=np.asarray(ids)
    if len(scores)!=len(ids) or not np.isfinite(scores).all():raise ValueError('Invalid retrieval scores')
    if not len(ids):return np.asarray([],dtype=np.int64)
    if positive_only and float(np.max(scores))<=0:return np.asarray([],dtype=np.int64)
    k=min(k,len(scores))
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    positions=np.flatnonzero(scores>=threshold)
    order=np.lexsort((ids[positions],-scores[positions]))[:k]
    return ids[positions[order]]


@lru_cache(maxsize=250000)
def words(text):return frozenset(re.findall(r'[a-z]+|\d+(?:\.\d+)?',text.lower()))


def text_features(term,context,name):
    t,c,n=words(term),words(context),words(name)
    tn={w for w in t if w[0].isdigit()};nn={w for w in n if w[0].isdigit()}
    conflict=('left' in t and 'right' in n) or ('right' in t and 'left' in n)
    acute=('acute' in t and 'chronic' in n) or ('chronic' in t and 'acute' in n)
    ctx_side=(('left' in c)!=('right' in c)) and (('left' in c and 'right' in n) or ('right' in c and 'left' in n))
    ctx_acuity=(('acute' in c)!=('chronic' in c)) and (('acute' in c and 'chronic' in n) or ('chronic' in c and 'acute' in n))
    cn={w for w in c if w[0].isdigit()}
    return [len(t&n)/max(1,len(t)),len(t&n)/max(1,len(n)),len(c&n)/max(1,len(n)),
            min(len(term),len(name))/max(1,len(term),len(name)),len(t),len(n),float(conflict),
            float(('bilateral' in t)!=('bilateral' in n)),float(acute),len(tn-nn),len(nn-tn),
            float(('without' in t)!=('without' in n)),float(('history' in t)!=('history' in n)),
            float(ctx_side),float(ctx_acuity),len(cn&nn)/max(1,len(nn))]


def build_lexical(prepared,output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    identity=file_sha(Path(prepared)/'CATALOG_MANIFEST.json')
    metadata=output/'LEXICAL_MANIFEST.json'
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        if saved['catalogue_sha256']!=identity:raise ValueError('Lexical index belongs to another catalogue')
        return
    names=pd.read_parquet(Path(prepared)/'names.parquet').name.tolist()
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,4),lowercase=False,dtype=np.float32)
    matrix=vectorizer.fit_transform(names).tocsr()
    sparse.save_npz(output/'char_tfidf.npz',matrix)
    joblib.dump(vectorizer,output/'char_vectorizer.joblib')
    metadata.write_text(json.dumps(dict(catalogue_sha256=identity,names=len(names),shape=matrix.shape,
        features='Character 3-4 gram L2 TF-IDF, fitted on static dictionary names only',
        files={p.name:file_sha(p) for p in (output/'char_tfidf.npz',output/'char_vectorizer.joblib')}),indent=2)+'\n')


class HybridRetriever:
    def __init__(self,prepared,embeddings,lexical,device='cpu'):
        self.prepared=Path(prepared)
        self.names=pd.read_parquet(self.prepared/'names.parquet').name.tolist()
        self.aliases=pd.read_parquet(self.prepared/'aliases.parquet')
        self.concepts=pd.read_parquet(self.prepared/'concepts.parquet')
        self.lookup={name:i for i,name in enumerate(self.names)}
        self.vectors=np.load(Path(embeddings)/'names.npy',mmap_mode='r')
        if self.vectors.shape!=(len(self.names),768):raise ValueError('Wrong catalogue embedding shape')
        self.vectorizer=joblib.load(Path(lexical)/'char_vectorizer.joblib')
        self.lexical=sparse.load_npz(Path(lexical)/'char_tfidf.npz').tocsr()
        if self.lexical.shape[0]!=len(self.names):raise ValueError('Wrong lexical index shape')
        self.system_ids={system:np.sort(group.name_id.unique()) for system,group in self.aliases.groupby('system')}
        self.code_names={};self.name_codes={system:defaultdict(list) for system in self.system_ids}
        grouped={}
        for row in self.aliases.itertuples(index=False):
            pair=grouped.setdefault((row.system,row.code),([],[]))
            pair[0].append(row.name_id);pair[1].append(row.preferred)
            self.name_codes[row.system][int(row.name_id)].append(row.code)
        self.code_names={key:(np.asarray(ids,dtype=np.int64),np.asarray(pref,dtype=bool)) for key,(ids,pref) in grouped.items()}
        del grouped
        self.tags={tag:i for i,tag in enumerate(sorted(set(self.concepts.semantic_tag)))}
        self.details={(r.system,r.code):dict(name=r.retrieval_preferred,tag=r.semantic_tag,
                      tag_id=self.tags[r.semantic_tag]) for r in self.concepts.itertuples(index=False)}
        self.device=device;self.gpu_vectors=None
        if device!='cpu':
            import torch
            raw=torch.from_numpy(np.array(self.vectors,copy=True)).to(device=device,dtype=torch.float32)
            self.gpu_vectors=torch.nn.functional.normalize(raw,dim=1)

    def score_arrays(self,term_vector,context_vector,term):
        query=np.stack([term_vector,context_vector]).astype(np.float32)
        query/=np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
        if self.gpu_vectors is not None:
            import torch
            with torch.inference_mode():scores=(torch.from_numpy(query).to(self.device)@self.gpu_vectors.T).cpu().numpy()
        else:
            scores=np.empty((2,len(self.names)),dtype=np.float32)
            for begin in range(0,len(self.names),8192):
                vectors=np.asarray(self.vectors[begin:begin+8192],dtype=np.float32)
                vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
                scores[:,begin:begin+len(vectors)]=query@vectors.T
        lexical=(self.lexical@self.vectorizer.transform([term]).T).toarray().ravel()
        return scores[0],scores[1],lexical

    def code_feature(self,system,code,term,context,term_vector,context_vector,arrays=None,ranks=None):
        ids,preferred=self.code_names[(system,code)]
        if arrays is None:
            vectors=np.asarray(self.vectors[ids],dtype=np.float32)
            vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            dense=vectors@term_vector;ctx=vectors@context_vector
            lexical=(self.lexical[ids]@self.vectorizer.transform([term]).T).toarray().ravel()
        else:
            dense,ctx,lexical=(values[ids] for values in arrays)
        hybrid=dense+lexical
        best=int(np.lexsort((ids,-preferred.astype(int),-hybrid))[0])
        name_id=int(ids[best]);name=self.names[name_id]
        rank_values=ranks.get(code,[0.,0.,0.]) if ranks else [0.,0.,0.]
        vector=[float(np.max(dense)),float(np.max(ctx)),float(np.max(lexical)),float(self.lookup.get(term,-1) in ids),
                float(preferred[best])]+text_features(term,context,name)+[self.details[(system,code)]['tag_id']]+rank_values
        if len(vector)!=len(FEATURES) or not np.isfinite(vector).all():raise ValueError('Invalid candidate features')
        return np.asarray(vector,dtype=np.float32),name_id

    def candidates(self,system,term,context,term_vector,context_vector):
        arrays=self.score_arrays(term_vector,context_vector,term)
        ids=self.system_ids[system]
        ranks={};codes=set()
        for source,(scores,k) in enumerate(zip(arrays,(200,100,200))):
            selected=stable_top(scores[ids],ids,k,positive_only=source==2)
            for rank,name_id in enumerate(selected,1):
                for code in self.name_codes[system][int(name_id)]:
                    codes.add(code)
                    values=ranks.setdefault(code,[0.,0.,0.])
                    values[source]=max(values[source],1/(60+rank))
        ordered=sorted(codes)
        features=[];aliases=[]
        for code in ordered:
            vector,name_id=self.code_feature(system,code,term,context,term_vector,context_vector,arrays,ranks)
            features.append(vector);aliases.append(name_id)
        return ordered,np.asarray(features,dtype=np.float32),np.asarray(aliases,dtype=np.int64)
