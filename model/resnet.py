import torch.nn as nn
from torchvision import models

resnet_dict = {
  "ResNet18": models.resnet18,
  "ResNet34": models.resnet34,
  "ResNet50": models.resnet50,
  "ResNet101": models.resnet101,
  "ResNet152": models.resnet152
}

class ResNet(nn.Module):
  def __init__(self, bit, res_model="ResNet50"):
    super(ResNet, self).__init__()
    model_resnet = resnet_dict[res_model](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    self.hash_layer = nn.Linear(model_resnet.fc.in_features, bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
  
  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0),-1)
    y = self.hash_layer(x)
    return y