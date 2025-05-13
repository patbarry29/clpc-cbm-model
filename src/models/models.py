from src.models import inception_v3
from src.models.resnet_model import CustomResNet


def ModelXtoCInception(pretrained, freeze, n_classes, n_concepts, use_aux=False, expand_dim=0):
    return inception_v3(
            pretrained=pretrained,
            freeze=freeze,
            n_classes=n_classes,
            aux_logits=use_aux,
            n_concepts=n_concepts,
            bottleneck=True,
            expand_dim=expand_dim
        )

def ModelXtoCResNet(pretrained, freeze, n_concepts, expand_dim=0):
    return CustomResNet(pretrained=pretrained, freeze=freeze, n_concepts=n_concepts)