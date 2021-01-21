from typing import Tuple
from classification_models.tfkeras import Classifiers


def get_resnet(name: str = 'resnet18',
               input_shape: Tuple[int, int, int] = (224, 224, 3),
               weights: str = 'imagenet',
               include_top: bool = False):
    ResNet18, preprocess_input = Classifiers.get(name)
    model = ResNet18(input_shape=input_shape, weights=weights, include_top=include_top)
    return model, preprocess_input

