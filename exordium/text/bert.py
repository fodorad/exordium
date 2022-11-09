import transformers as tfm


class BertWrapper():

    def __init__(self) -> None:
        self.tokenizer =  tfm.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model =  tfm.BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()


    def extract_features(self, text: str | list[str] | list[list[str]],
                         padding: bool | str = True,
                         max_length: int | None = None,
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
        inputs = self.tokenizer(text, return_tensors=return_tensors, padding=padding, truncation=True, max_length=max_length)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


if __name__ == "__main__":

    bert_wrapper = BertWrapper()
    print(bert_wrapper.extract_features('Welcome, this is an example').shape)
    print(bert_wrapper.extract_features(['Welcome, this is an example', 'An another, longer example. I mean a lot longer.']).shape)
