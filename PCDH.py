import torch
from utils.util import *
import torch.nn as nn
from loguru import logger
from data.loadData import get_data
from model.alexnet import AlexNet


# epoch: 250; batch size: 64; lr: 1e-5; dataset: cifar10; bit: 48 ; alpha: 1; beta: 1, Best mAP: 0.1882  
class PCDHNet(nn.Module):
  def __init__(self, args,pretrained=True):
    super(PCDHNet, self).__init__()
    self.conv_layer = nn.Sequential(
      nn.Conv2d(3, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
    )
    
    self.feature_layer = nn.Linear(8 * 8 * 256, 1024)
    self.hash_like_layer = nn.Sequential(
      nn.Linear(1024, args.bit),
      # 将输出映射到[-1, 1]
      nn.Tanh()
    )
    self.discrete_hash_layer = nn.Linear(args.bit, args.bit)
    self.classfication_layer = nn.Linear(args.bit, args.n_class, bias=False)
  
  def forward(self, x, istraining=False):
    x = self.conv_layer(x)
    x = x.view(x.size(0), -1)
    feature = self.feature_layer(x)
    h = self.hash_like_layer(feature)
    # 将hash_like映射到离散哈希码{-1,1}
    b = self.discrete_hash_layer(h).add(1).mul(0.5).clamp(min=0,max=1)
    b = (b >= 0.5).float() * 2 - 1
    y_pre = self.classfication_layer(b)
    if istraining:
      # 如果训练中，那么需要以下这些构成目标函数
      return feature, h, y_pre
    else:
      return b

def PCDH_loss(feature, h, y_pre, cls, m, ind, args):
  dist_h = (h.unsqueeze(1) - h.unsqueeze(0)).pow(2).sum(dim=2)
  y = (cls @ cls.t() == 0).float()
  
  loss1 = (1-y) /2 * dist_h + y/2 * (m-dist_h).clamp(min=0).pow(2)
  loss1 = loss1.mean()
  
  dist_feature = (feature.unsqueeze(1) - feature.unsqueeze(0)).pow(2).sum(dim=2)
  loss2 = (1-y) /2 * dist_feature + y/2 * (m-dist_feature).clamp(min=0).pow(2)
  loss2 = loss2.mean()
  
  # 交叉熵损失
  Lc = (-y_pre.softmax(dim=1).log() * cls).sum(dim=1).mean()
  
  return loss1 + args.alpha * loss2 + args.beta * Lc

if __name__ == '__main__':
  logger.add(os.path.join('log','PCDH','{time}.log'), rotation='500 MB', level="INFO")
  args = load_config()
  seed_setting(args.seed)
  args.alpha = 1
  args.beta = 1
  args.crop_size = 128
  
  
  net = PCDHNet(args).to(args.device)
  optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
  train_loader, test_loader, database_loader, num_train, num_test, num_database =  get_data(args)
  Best_mAP = 0
  
  for epoch in range(1, args.epoch+1):
    net.train()
    train_loss = 0
    
    for img, label, ind in train_loader:
      img = img.to(args.device)
      label = label.to(args.device)
      optimizer.zero_grad()
      feature, h, y_pre = net(img, istraining=True)
      
      loss = PCDH_loss(feature, h, y_pre, label.float(), 2 * args.bit, ind, args)
      
      train_loss += loss.item()
      loss.backward()
      optimizer.step()
      
    train_loss = train_loss / len(train_loader)
    logger.info(f'[{epoch}] loss: {train_loss:.4f}')
    
    if epoch % args.checkpoint == 0:
      mAP = validate(args, test_loader, database_loader, net)
      if Best_mAP < mAP:
        Best_mAP = mAP
        torch.save(net.state_dict(), os.path.join('save', 'PCDH', f'{args.bit}.pth'))
      logger.info(f'[{epoch}] retrieval mAP: {mAP:.4f},Best mAP: {Best_mAP:.4f}')