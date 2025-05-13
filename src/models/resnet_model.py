import torch
import torch.nn as nn
import torchvision.models as models

class FC(nn.Module):
    def __init__(self, in_features, out_features, expand_dim=0):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features + expand_dim, out_features)
    def forward(self, x):
        return self.fc(x)

class CustomResNet(nn.Module):
    def __init__(self, pretrained, freeze, n_concepts, expand_dim=0):
        super().__init__()
        # Load the base ResNet
        base_resnet = models.resnet50(pretrained=pretrained)

        if freeze:
            for param in base_resnet.parameters():
                param.requires_grad = False

        # Copy all layers from base_resnet except the original fc
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

        num_ftrs = base_resnet.fc.in_features

        # Your custom concept heads
        self.all_fc = nn.ModuleList()
        for _ in range(n_concepts):
            self.all_fc.append(FC(num_ftrs, 1, expand_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs_list = [fc(x) for fc in self.all_fc]
        return outputs_list

# model = CustomResNet(pretrained=True, freeze=True, n_concepts=N_TRIMMED_CONCEPTS)