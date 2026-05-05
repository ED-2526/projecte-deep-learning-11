import torch.nn as nn
from torchvision import models


def create_resnet_model(num_classes, model_name="resnet18", feature_extraction=True):
    """
    Crea una ResNet preentrenada i adapta l'última capa al nostre nombre de classes.

    feature_extraction=True:
        congela tota la ResNet i només entrena la capa final.

    feature_extraction=False:
        entrena tota la xarxa.
    """

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model no suportat: {model_name}")

    if feature_extraction:
        for param in model.parameters():
            param.requires_grad = False

    # Canviem la capa final perquè classifiqui els estils de WikiArt
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model