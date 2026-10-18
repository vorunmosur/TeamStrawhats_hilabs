"""Shared normalization, exactly matching the supplied data contract."""
import re
import unicodedata

_BR=re.compile(r'<br\s*/?>',re.I); _HEADER=re.compile(r'header_\d+\s*:'); _JUNK=re.compile(r'::|_{2,}')
_HCC=re.compile(r'\(\s*hcc\s*\)'); _SPACE_PUNCT=re.compile(r'\s+([.,;:!?)\]])'); _WS=re.compile(r'\s+')
_EDGE_L=re.compile(r'^(?![+-]\d)[\s\W_]+'); _EDGE_R=re.compile(r'[\s\W_]+$')

def level1(text):
    t=unicodedata.normalize('NFKC',str(text)).lower()
    t=_BR.sub(' ',t); t=_HEADER.sub(' ',t); t=_JUNK.sub(' ',t); t=_HCC.sub(' ',t)
    t=_SPACE_PUNCT.sub(r'\1',t); t=_WS.sub(' ',t).strip()
    s=_EDGE_L.sub('',t)
    m=_EDGE_R.search(s)
    if m:
        keep='+' if m.group().startswith('+') and m.start()>0 and s[m.start()-1].isdigit() else ''
        s=s[:m.start()]+keep
    if t.endswith(')') and s.count('(')>s.count(')'): s+=')'
    return s

def context_window(entity, chunk):
    term, text = level1(entity), level1(chunk)
    if not text:
        return ''
    pos = text.find(term) if term else -1
    if pos < 0:
        return text[:200]
    return f'{text[max(0,pos-100):pos]} [ENT] {term} [/ENT] {text[pos+len(term):pos+len(term)+100]}'.strip()
