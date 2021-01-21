'''Text-based utility functions

In case of spacy errors:
    os.system('python -m spacy download en')
    os.system('spacy download en_vectors_web_lg') 
'''
import os
import re
import pickle
from typing import List, Tuple, Set, Union, Optional

import spacy
import tensorflow as tf
import transformers as tfm


def texts2tokens(input_list: List[str], output_dir: str) -> None:
    """Tokenize texts and save as a pickle file

    Args:
        input_list (List[str]): list of texts
        output_dir (str): output path
    """
    try:
        NLP = spacy.load("en_core_web_sm")
    except:
        os.system('python -m spacy download en')
        os.system('spacy download en_vectors_web_lg') 
        NLP = spacy.load("en_core_web_sm")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for text in tqdm(input_list):
        tokenized_text, vocab = tokenize_with_spacy(text)
        with open(str(output_dir / 'transcript.pkl'), 'wb'):
            pickle.dump({'text': text, 'tokenized_text': tokenized_text, 'vocab': vocab})


def tokenize_with_spacy(raw_text: str) -> Tuple[List[str], Set[str]]:
    """Tokenize text with SpaCy
    Filters special characters, digits, urls and emails
    start with smaller model, then to large: en_core_web_lg

    Args:
        raw_text (str): input text

    Returns:
        Tuple[List[str], Set[str]]: tokenized text, vocabulary
    """
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


def extract_bert(text: Union[str, List[str], List[List[str]]],
                 padding: Union[bool, str] = True,
                 max_length: Optional[int] = None) -> tf.Tensor:
    """Extracts bert hidden states from text.
    https://huggingface.co/transformers/model_doc/bert.html

    Args:
        text (str or List[str]): Cleaned text as a single string or list of strings
        padding (bool): Pad the tokenized sequences to the same length. 
                        for predifined padding size: padding should be 'max_length', and max_length is set to an int

    Returns:
        tf.Tensor: Sequence of hidden-states at the output of the last layer of the model
                   Tensor of shape (batch_size, sequence_length, hidden_size)
    """
    tokenizer =  tfm.BertTokenizer.from_pretrained('bert-base-uncased')
    model =  tfm.TFBertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='tf', padding=padding, truncation=True, max_length=max_length)
    outputs = model(inputs)
    return outputs.last_hidden_state


def extract_albert(text: Union[str, List[str], List[List[str]]],
                   padding: Union[bool, str] = True,
                   max_length: Optional[int] = None) -> tf.Tensor:
    """Extracts albert hidden states from text.
    https://huggingface.co/transformers/model_doc/albert.html

    Args:
        text (str or List[str]): Cleaned text as a single string or list of strings
        padding (bool): Pad the tokenized sequences to the same length. 
                        for predifined padding size: padding should be 'max_length', and max_length is set to an int

    Returns:
        tf.Tensor: Sequence of hidden-states at the output of the last layer of the model
                   Tensor of shape (batch_size, sequence_length, hidden_size)
    """
    tokenizer =  tfm.AlbertTokenizer.from_pretrained('albert-base-v2')
    model =  tfm.TFAlbertModel.from_pretrained('albert-base-v2')
    inputs = tokenizer(text, return_tensors='tf', padding=padding, truncation=True, max_length=max_length)
    outputs = model(inputs)
    return outputs.last_hidden_state


if __name__ == "__main__":
    print(extract_albert('Welcome, this is an example').shape)
    print(extract_albert(['Welcome, this is an example', 'An another, longer example. I mean a lot longer.']).shape)

