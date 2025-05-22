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

def ModelXtoCResNet(pretrained, freeze, n_concepts, expand_dim=0, n_classes=None, label_mode=False):
    return CustomResNet(pretrained=pretrained, freeze=freeze, n_concepts=n_concepts, n_classes=n_classes, label_mode=label_mode)