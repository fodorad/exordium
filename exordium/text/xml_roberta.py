import transformers as tfm
import torch


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class XmlRobertaWrapper():

    def __init__(self) -> None:
        """XML RoBERTa wrapper class."""
        self.tokenizer = tfm.AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.model = tfm.AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.model.eval()

    def __call__(self, text: str | list[str] | list[list[str]],
                       padding: bool | str = True,
                       max_length: int | None = None):
        """Extracts XML RoBERTa embeddings.
        https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2

        Args:
            text (str | list[str] | list[list[str]]): Cleaned text as a single string or list of strings.
            padding (bool): Pad the tokenized sequences to the same length for predifined padding size:
                            padding should be 'max_length', and max_length is set to an int.

        Returns:
            torch.Tensor: embedding tensor of shape (batch_size, hidden_size).
        """
        inputs = self.tokenizer(text, padding=padding, truncation=True, return_tensors='pt', max_length=max_length)

        with torch.no_grad():
            outputs = self.model(**inputs)

        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        return sentence_embeddings # (B, C) == (B, 768)