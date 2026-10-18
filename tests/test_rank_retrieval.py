import numpy as np
import pandas as pd
from genlink.retrieval import stable_top,text_features,FEATURES,ALL_FEATURES
from genlink.rank_retrieval import TrainingMemory,assemble,ordering
from genlink.build_candidates import CandidateStore


def test_deterministic_cutoff_ties():
    assert stable_top(np.array([.2,.8,.8,.8]),np.array([4,3,2,1]),2).tolist()==[1,2]
    assert stable_top(np.zeros(3),np.arange(3),2,True).tolist()==[]
    assert ordering(['B','A'],[.5,.5]).tolist()==[1,0]


def test_qualifier_features_and_schema():
    features=text_features('acute left pain 2','chronic right pain','chronic right pain 3')
    assert len(features)==16
    assert features[6]==1 and features[8]==1 and features[9:11]==[1,1]
    assert len(FEATURES)==25 and len(ALL_FEATURES)==32


def test_memory_excludes_own_chunk_and_has_no_held_labels():
    rows=pd.DataFrame(dict(system=['ICD10CM']*3,code=['A01','B02','B02'],query_id=[0,1,2]))
    queries=pd.DataFrame(dict(query_id=[0,1,2],term_norm=['pain','pain','other'],chunk_text=['SAME chunk','same chunk!','different']))
    vectors=np.zeros((3,2,768),dtype=np.float32)
    vectors[0,:,0]=1;vectors[1,:,0]=1;vectors[2,:,1]=1
    details={('ICD10CM',c):{'tag':''} for c in ['A01','B02','C03']}
    memory=TrainingMemory(rows,queries,vectors,'ICD10CM',details)
    extra,feature=memory.search('pain','',vectors[0,0],vectors[0,1],exclude_chunk='same chunk')
    assert extra==['B02']
    assert feature('A01')[0]==0 and feature('A01')[2]==-1 and feature('A01')[4]==0
    assert feature('B02')[0]==np.log(2) and feature('B02')[4]==0
    assert feature('C03')[0]==0 and feature('C03')[2]==-1
    assert 'C03' not in memory.code_lookup


def test_candidate_store_roundtrip(tmp_path):
    store=CandidateStore(tmp_path/'cache.sqlite')
    matrix=np.arange(50,dtype=np.float32).reshape(2,25)
    store.put(7,'ICD10CM',['A','B'],matrix,np.array([4,8]))
    codes,features,aliases=store.get(7,'ICD10CM')
    assert codes==['A','B'] and np.array_equal(features,matrix) and aliases.tolist()==[4,8]
    assert store.get(8,'ICD10CM') is None
    store.close()

def test_training_gold_injection_is_explicit_and_not_in_held_candidates():
    class Index:
        names=['a','b']
        def code_feature(self,*args,**kwargs):return np.zeros(25,dtype=np.float32),1
    class Memory:
        system='ICD10CM'
        def search(self,*args,**kwargs):return [],lambda code:[0.]*7
    query=pd.Series(dict(term_norm='term',context_norm='context'))
    vectors=np.zeros((2,768),dtype=np.float32)
    base=(['A'],np.zeros((1,25),dtype=np.float32),np.array([0]))
    codes,features,_=assemble(Index(),Memory(),query,vectors,base)
    assert codes==['A'] and features.shape==(1,32)
    codes,features,_=assemble(Index(),Memory(),query,vectors,base,gold='B',exclude_chunk='own')
    assert codes==['A','B'] and features.shape==(2,32)


def test_input_adapter_preserves_ids_and_requires_real_context(tmp_path):
    import pytest
    from genlink.input_adapter import input_rows
    source=tmp_path/'in.csv';source.write_text('chunk_id,clinical_entity\n001,HTN\n')
    with pytest.raises(ValueError):input_rows(source)
    lookup=tmp_path/'chunks.csv';lookup.write_text('chunk_id,chunk_text\n001,history of HTN\n')
    rows,entity=input_rows(source,lookup)
    assert entity=='clinical_entity' and rows.chunk_id.tolist()==['001']
    assert rows.chunk_text.tolist()==['history of HTN']
    lookup.write_text('chunk_id,chunk_text\n001,a\n001,b\n')
    with pytest.raises(ValueError):input_rows(source,lookup)

def test_original_label_schema_is_adapted_without_changing_folds(tmp_path):
    from genlink.input_adapter import labelled_rows
    path=tmp_path/'rows.parquet'
    original=pd.DataFrame(dict(row=[1,2],fold=[0,2],query_id=[0,1],coding_system=['ICD10CM','SNOMEDCT_US'],code=['A01','123']))
    original.to_parquet(path,index=False)
    result=labelled_rows(path)
    assert result.system.tolist()==original.coding_system.tolist()
    assert result.fold.tolist()==[0,2] and result.row.tolist()==[1,2]
