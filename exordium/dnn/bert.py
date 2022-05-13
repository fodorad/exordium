from typing import Union, Optional, List
import transformers as tfm

BertTokenizer =  tfm.BertTokenizer.from_pretrained('bert-base-uncased')
BertModel =  tfm.BertModel.from_pretrained('bert-base-uncased')
BertModel.eval()


def extract_bert(text: Union[str, List[str], List[List[str]]],
                 padding: Union[bool, str] = True,
                 max_length: Optional[int] = None,
                 return_tensors='pt'):
    """Extracts bert hidden states from text.
    https://huggingface.co/transformers/model_doc/bert.html

    Args:
        text (str or List[str]): Cleaned text as a single string or list of strings
        padding (bool): Pad the tokenized sequences to the same length. 
                        for predifined padding size: padding should be 'max_length', and max_length is set to an int

    Returns:
        torch.Tensor: Sequence of hidden-states at the output of the last layer of the model
                      Tensor of shape (batch_size, sequence_length, hidden_size)
    """
    inputs = BertTokenizer(text, return_tensors=return_tensors, padding=padding, truncation=True, max_length=max_length)
    outputs = BertModel(**inputs)
    return outputs.last_hidden_state


if __name__ == "__main__":
    print(extract_bert('Welcome, this is an example').shape)
    print(extract_bert(['Welcome, this is an example', 'An another, longer example. I mean a lot longer.']).shape)
