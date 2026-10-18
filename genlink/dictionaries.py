"""Full dictionaries and fold-isolated ambiguity resolution."""
from collections import Counter
from pathlib import Path
import re
import pandas as pd
from .normalize import level1

# Official Editorial Guide, Semantic Tag table, retrieved 2026-10-18.
TAG_SOURCE = 'https://docs.snomed.org/snomed-ct-specifications/snomed-ct-editorial-guide/readme/authoring/general-naming-conventions/descriptions/fully-specified-name/index'
SEMANTIC_TAGS = frozenset('''body structure|cell|cell structure|morphologic abnormality|finding|disorder|environment|geographic location|event|observable entity|organism|clinical drug|medicinal product|medicinal product form|physical object|product|physical force|procedure|regime/therapy|qualifier value|administration method|basic dose form|calculation|disposition|dose form|intended site|product name|release characteristic|role|state of matter|transformation|supplier|unit of presentation|record artifact|situation|attribute|core metadata concept|foundation metadata concept|link assertion|linkage concept|namespace concept|owl metadata concept|social concept|ethnic group|occupation|person|racial group|religion/philosophy|navigational concept|specimen|staging scale|assessment scale|tumor staging|substance'''.split('|'))
_TAG = re.compile(r'\s*\(([^()]*)\)\s*$')

def split_tag(description):
    """Only official trailing tags are removed. Abbreviations stay intact."""
    match = _TAG.search(description)
    if match and match.group(1).lower() in SEMANTIC_TAGS:
        return description[:match.start()].strip(), match.group(1).lower()
    return description, ''

def normalize_name(description, system):
    return level1(split_tag(description)[0] if system == 'SNOMEDCT_US' else description)

def load_dictionary(source, system):
    source = Path(source)
    if system == 'ICD10CM':
        raw = pd.read_excel(source/'icd10cm_codes.xlsx', dtype=str, keep_default_na=False)
        raw = raw.apply(lambda c: c.str.strip())
        raw = raw.loc[raw.Code.ne('') & ~raw.Code.str.contains('-', regex=False)].copy()
        assert len(raw) == 159195 and raw.Code.nunique() == 98182
        raw['preferred'] = raw.groupby('Code', sort=False).cumcount().eq(0)
        names = raw.rename(columns={'Code':'code', 'Description':'description'})[['code','description','preferred']]
        names['semantic_tag'] = ''
    elif system == 'SNOMEDCT_US':
        raw = pd.read_excel(source/'SNOMEDCT_US_codes_and_descriptions.xlsx', dtype=str, keep_default_na=False)
        raw = raw.apply(lambda c: c.str.strip())
        assert len(raw) == 984585 and raw.snomed_code.nunique() == 382124
        names = raw.rename(columns={'snomed_code':'code'})[['code','description','description_order']].copy()
        names['preferred'] = names.description_order.eq('1')
        assert names.loc[names.preferred, 'code'].is_unique
        assert names.loc[names.preferred, 'code'].nunique() == names.code.nunique()
        tagged = names.description.map(split_tag)
        names['stripped'] = tagged.map(lambda pair: pair[0])
        names['suffix_tag'] = tagged.map(lambda pair: pair[1])
        # The export has no description-type field. Recognize FSNs only by an
        # official trailing semantic tag, as the supplied contract specifies.
        fsns = names.loc[names.suffix_tag.ne(''), ['code','suffix_tag']].drop_duplicates()
        conflicts = fsns.groupby('code').suffix_tag.nunique()
        if conflicts.gt(1).any():
            raise ValueError(f'{int(conflicts.gt(1).sum())} concepts have conflicting FSN tags')
        tags = fsns.set_index('code').suffix_tag
        names['semantic_tag'] = names.code.map(tags).fillna('')
    else:
        raise ValueError(system)
    names['name_norm'] = names.description.map(lambda n: normalize_name(n, system))
    # Exact Level-1 normalization erases ten symbol-only SNOMED aliases.
    # Preserve them, including five preferred aliases, rather than silently
    # dropping dictionary rows or changing the supplied normalization rule.
    empty_names = int(names.name_norm.eq('').sum())
    empty_preferred = int((names.name_norm.eq('') & names.preferred).sum())
    preferred = names.loc[names.preferred, ['code','name_norm','semantic_tag','description']].copy()
    preferred = preferred.rename(columns={'description':'name_raw'})
    assert preferred.code.is_unique
    # If preferred and non-preferred descriptions normalize identically,
    # preferred status wins. Keep one row for each normalized name/code pair.
    names = names.sort_values(['name_norm','code','preferred'], ascending=[True,True,False], kind='stable')
    names = names.drop_duplicates(['name_norm','code'])[['name_norm','code','preferred','semantic_tag']].reset_index(drop=True)
    return names, preferred, {'raw_rows':len(raw), 'codes':preferred.code.nunique(),
                              'name_code_pairs':len(names), 'unique_names':names.name_norm.nunique(),
                              'empty_normalized_description_rows':empty_names,
                              'empty_normalized_preferred_names':empty_preferred,
                              'tagged_concepts':int(preferred.semantic_tag.ne('').sum()),
                              'tag_counts':preferred.semantic_tag.value_counts().to_dict()}

class NameIndex:
    """Build independently for each fold. Validation labels cannot enter counts."""
    def __init__(self, names, labelled_rows, system, heldout_fold=None):
        rows = labelled_rows.loc[labelled_rows.coding_system.eq(system)]
        if heldout_fold is not None:
            if heldout_fold not in (0,1,2):
                raise ValueError('heldout_fold must be 0, 1, 2, or None for final')
            rows = rows.loc[rows.fold.ne(heldout_fold)]
        self.frequencies = Counter(rows.code)
        self.training_rows = len(rows)
        ordered = names.assign(frequency=names.code.map(self.frequencies).fillna(0))
        ordered = ordered.sort_values(['name_norm','preferred','frequency','code'], ascending=[True,False,False,True], kind='stable')
        self.mapping = {}
        for name,code in zip(ordered.name_norm,ordered.code):
            self.mapping.setdefault(name,[]).append(code)

    def codes(self, name):
        return self.mapping.get(name, [])
