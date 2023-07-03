import torch


def fix_ckpt(weights: str | dict) -> dict:
    """The ckpt file saves the pytorch_lightning module which includes it's child members. The only child member we're interested in is the "_model".
    Loading the state_dict with _model creates an error as the model tries to find a child called _model within it that doesn't
    exist. Thus remove _model from the dictionary and all is well.
    """
    if isinstance(weights, str):
        weights = torch.load(weights)

    return {k[7:]: v for k, v in weights.items() if k.startswith("_model.")}
