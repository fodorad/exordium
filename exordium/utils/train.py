
def get_parameters(model):
    return filter(lambda p: p.requires_grad, model.parameters())
