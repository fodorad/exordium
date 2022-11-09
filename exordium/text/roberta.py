import torch


class RobertaWrapper():

    def __init__(self) -> None:
        self.model = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.model.eval()


    def extract_features(self, text: str):
        tokens_text = self.model.encode(text)
        roberta_feature = self.model.extract_features(tokens_text).squeeze() # (T, C)
        return roberta_feature.detach().numpy()


if __name__ == "__main__":

    roberta_wrapper = RobertaWrapper()
    print(roberta_wrapper.extract_features('Welcome, this is an example').shape)
