import torch

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()


def extract_roberta(text: str):
    tokens_text = roberta.encode(text)
    roberta_feature = roberta.extract_features(tokens_text).squeeze() # (T, C)
    return roberta_feature


if __name__ == "__main__":
    print(extract_roberta('Welcome, this is an example').shape)