from .alexnet import AlexNet
from .resnet import ResNet

def load_model(model="AlexNet", bit=12, device="cuda:0"):
  if model == 'AlexNet':
    return AlexNet(hash_bit=bit).to(device)
  elif model == 'ResNet':
    return ResNet(bit=bit).to(device)