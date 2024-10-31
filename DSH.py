import torch
from utils.util import *
import torch.nn as nn
from loguru import logger
from data.loadData import get_data
from model.load_model import load_model

# epoch: 250; batch size: 64; lr: 1e-5; dataset: cifar10; bit: 48 Best mAP: 0.7786  
def DSH_loss(y_hat, cls, m, alpha):
  """
  compute hashing loss
  y_hat: [batch_size, bit]
  cls: [batch_size, class_num]
  y: similarity matrix [batch_size, batch_size]
  dist: hamming distance matrix [batch_size, batch_size]
  """
  # 如果类别相同为0， 类别不同为1
  y = (cls @ cls.t() == 0).float()
  dist = (y_hat.unsqueeze(1) - y_hat.unsqueeze(0)).pow(2).sum(dim=2)
  loss = (1-y) /2 * dist + y/2 * (m-dist).clamp(min=0)
  loss = loss.mean() + alpha * (y_hat.abs() - 1).abs().mean()
  return loss

class DSH(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(3,32,kernel_size = 5, padding = 2),
      # 原地修改tensor，节省内存
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3,stride=2),
      
      nn.Conv2d(32,32,kernel_size=5, padding = 2),
      nn.ReLU(inplace=True),
      nn.AvgPool2d(kernel_size=3,stride=2),
      
      nn.Conv2d(32, 64, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.AvgPool2d(kernel_size=3, stride=2),
    )
    
    self.fc = nn.Sequential(
      nn.Linear(64 * 3 * 3, 500),
      nn.ReLU(inplace=True),
      nn.Linear(500, args.bit)
    )
    
    for m in self.modules():
      if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
  
  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    
    return x
  

if __name__ == '__main__':
  logger.add(os.path.join('log','DSH','{time}.log'), rotation='500 MB', level="INFO")
  args = load_config()
  seed_setting(args.seed)
  
  net = load_model(args.model, args.bit, args.device)
  optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
  train_loader, test_loader, database_loader =  get_data(args)
  Best_mAP = 0
  logger.info(f"[DSH] model: {args.model}, dataset: {args.dataset}, bit: {args.bit}, lr: {args.lr}, gpu: {args.device}, batch_size: {args.batch_size}, epoch: {args.epoch}, alpha: {args.alpha} training...")
  for epoch in range(1, args.epoch+1):
    net.train()
    train_loss = 0
    for img, label, ind in train_loader:
      img = img.to(args.device)
      label = label.to(args.device)
      optimizer.zero_grad()
      u = net(img)
      loss = DSH_loss(u, label.float(), 2*args.bit, args.alpha)
      train_loss += loss.item()
      loss.backward()
      optimizer.step()
    train_loss = train_loss / len(train_loader)
    logger.info(f'[{epoch}] loss: {train_loss:.4f}')
    
    if epoch % args.checkpoint == 0:
      mAP = validate(args, test_loader, database_loader, net)
      if Best_mAP < mAP:
        Best_mAP = mAP
        torch.save(net.state_dict(), os.path.join('save', 'DSH', f'{args.bit}.pth'))
      logger.info(f'[{epoch}] retrieval mAP: {mAP:.4f},Best mAP: {Best_mAP:.4f}')