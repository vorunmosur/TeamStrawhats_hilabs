"""One raw prompt builder and one target encoder for trie and future training."""
from .normalize import context_window, level1

SYSTEMS = ('ICD10CM', 'SNOMEDCT_US')

def build_prompt(entity, chunk_text, system):
    if system not in SYSTEMS:
        raise ValueError(f'Unknown coding system: {system}')
    return f'[SYSTEM] {system}\n[TERM] {level1(entity)}\n[CONTEXT] {context_window(entity, chunk_text)}\n[CONCEPT]\n'

def encode_targets(tokenizer, names):
    """Do not concatenate prompt text before tokenization: preserve this boundary."""
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires an EOS token')
    encoded = tokenizer([' ' + n for n in names], add_special_tokens=False)['input_ids']
    for tokens in encoded:
        if tokenizer.eos_token_id in tokens:
            raise ValueError('Dictionary name contains a literal EOS special token')
    return [tuple(tokens) + (tokenizer.eos_token_id,) for tokens in encoded]

def completion_example(tokenizer, entity, chunk_text, system, name):
    """Completion-only labels; also used by the shared-boundary unit test."""
    prompt = tokenizer(build_prompt(entity, chunk_text, system), add_special_tokens=False)['input_ids']
    target = list(encode_targets(tokenizer, [name])[0])
    return {'input_ids': prompt + target, 'labels': [-100] * len(prompt) + target,
            'attention_mask': [1] * (len(prompt) + len(target))}
