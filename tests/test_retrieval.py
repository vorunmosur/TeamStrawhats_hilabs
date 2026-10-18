import pytest
from genlink.encode_retrieval import token_windows,token_work


@pytest.mark.parametrize('length',[0,1,510,511,1024,1444,4097])
def test_overlapping_encoder_windows_preserve_every_token(length):
    source=list(range(length))
    windows=list(token_windows(source))
    assert windows and all(len(window)<=510 for window in windows)
    assert set(token for window in windows for token in window)==set(source)
    assert windows==list(token_windows(source))


def test_padded_token_plan_counts_complete_real_batches():
    assert token_work([3,5,7,11,13],2)==5*2+11*2+13
    assert token_work([3,5,7],1)==15
