import torchvision
import torch.nn as nn
import torch

class cnn_model(nn.Module):
  """
  基于vgg11
  """
  def __init__(self, bit):
    super(cnn_model, self).__init()
    self.model_name = "vgg11"
    self.original_model = torchvision.models.vgg11(pretrained=True)
    self.features = self.original_model.features
    cl1 = nn.Linear(25088, 4096)
    cl1.weight = self.original_model.classifier[0].weight
    cl1.bias = self.original_model.classifier[0].bias
    
    cl2 = nn.Linear(4096, 4096)
    cl2.weight = self.original_model.classifier[3].weight
    cl2.bias = self.original_model.classifier[3].bias
    
    self.classifier = nn.Sequential(
      cl1,
      nn.ReLU(inplace=True),
      nn.Dropout(),
      cl2,
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, bit)
    )
  def forward(self, x):
    f = self.features(x)
    # 展平操作
    f = f.view(f.size(0),-1)
    y = self.classifier(f)
    return y