import os
import re
import pickle
from typing import List, Tuple, Set

from tqdm import tqdm
from pathlib import Path


def texts2tokens(input_list: List[str], output_path: str, ids: List[str] = None) -> None:
    """Tokenize texts and save as a pickle file

    Args:
        input_list (List[str]): list of texts
        output_path (str): output path
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = set()
    texts = []
    tokenized_texts = []
    for text in tqdm(input_list):
        tokenized_text, text_vocab = tokenize_with_spacy(text)
        vocab = vocab.union(text_vocab)
        texts.append(text)
        tokenized_texts.append(tokenized_text)
    with open(output_path, 'wb') as f:
        pickle.dump({'ids': ids, 'texts': texts, 'tokenized_texts': tokenized_texts, 'vocab': vocab}, f)


def tokenize_with_spacy(raw_text: str) -> Tuple[List[str], Set[str]]:
    """Tokenize text with SpaCy
    Filters special characters, digits, urls and emails
    start with smaller model, then to large: en_core_web_lg

    Args:
        raw_text (str): input text

    Returns:
        Tuple[List[str], Set[str]]: tokenized text, vocabulary
    """
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm")
    except Exception:
        os.system('python -m spacy download en')
        os.system('spacy download en_vectors_web_lg')
        raise ValueError('SpaCy is set. restart.')
    doc = NLP(raw_text)
    vocab = set()
    tokenized_text = []
    for token in doc:
        word = ''.join([i if ord(i) < 128 else ' ' for i in token.text])
        word = word.strip().lower()
        if not token.is_digit \
            and not token.like_url \
            and not token.like_email:
            vocab.add(word)
            tokenized_text.append(word)
    return tokenized_text, vocab


def tokenize_simple(text: str) -> List[str]:
    """Return the tokens of a sentence including punctuation.

    Example:
        tokenize_simple('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    
    Args:
        text (str): input text
    
    Returns:
        List(str): tokenized text
    """
    return [x.strip() for x in re.split('(\W+)', text) if x.strip()]

