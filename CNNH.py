import argparse
import torch
from utils.util import *
import torch.nn as nn
from itertools import product
from loguru import logger
from random import shuffle
from tqdm import tqdm
from data.loadData import get_data
from model.alexnet import AlexNet

class CNNHLoss(nn.Module):
  def __init__(self, args, train_labels, num_train):
    super(CNNHLoss, self).__init__()
    S = (train_labels @ train_labels.t() > 0).float() * 2 - 1
    self.save_full_path = "save/CNNH/CNNH_H.pt"
    if os.path.exists(self.save_full_path):
      logger.info("loading ", self.save_full_path)
      self.H = torch.load(self.save_full_path).to(args.device)
    else:
      self.H = self.stage_one(num_train, args.bit, args.T, S, args.device)
  
  def stage_one(self, n, q, T, S, device):
    # H(n,q)，n行图像，每行是q bit的哈希码
    H = 2 * torch.rand((n,q)).to(device) - 1
    L = H @ H.t() - q * S
    # 生成所有(i,j)对
    permutation = list(product(range(n),range(q)))
    for t in range(T):
      H_temp = H.clone()
      L_temp = L.clone()
      shuffle(permutation)
      # 显示进度条
      for i,j in tqdm(permutation):
        # formula 7
        g_prime_Hij = 4 * L[i,:] @ H[:,j]
        g_prime_prime_Hij = 4 * (H[:,j].t() @ H[:,j] + H[i,j].pow(2) + L[i,i])
        # formula 6
        d = (-g_prime_Hij / g_prime_prime_Hij).clamp(min=-1-H[i,j], max=1-H[i,j])
        # formula 8
        L[i,:] = L[i,:] + d * H[:,j].t()
        L[:,i] = L[:,i] + d * H[:,j]
        L[i,i] = L[i,i] + d * d
        
        H[i,j] = H[i,j] + d

        if L.pow(2).mean() >= L_temp.pow(2).mean():
          H = H_temp
          L = L_temp
        logger.info(f"[CNNH stage 1][{t+1}/{T}] reconstruction loss: {L.pow(2).mean().item():.7f}")
    # 正数为1，负数为-1，生成哈希码
    torch.save(H.sign().cpu(), self.save_full_path)
    return H.sign()
  
  def forward(self, u, ind):
    #均方误差
    loss = (u - self.H[ind]).pow(2).mean()
    return loss

def train_val(args):
  # 获取数据
  train_loader, test_loader, database_loader = get_data(args)
  net = AlexNet(args.bit).to(args.device)
  
  # 获取labels
  clses = []
  for _, cls, _ in tqdm(train_loader):
    clses.append(cls)
  train_labels = torch.cat(clses).to(args.device).float()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  # 阶段1：分解S为H H^T
  logger.info("Stage 1: learning approximate hash codes.")
  criterion = CNNHLoss(args, train_labels, args.num_train)
  
  #阶段2：训练网络
  logger.info("Stage 2: learning images feature representation and hash functions.")
  Best_mAP = 0
  for epoch in range(1,args.epoch+1):
    logger.info(f"{epoch}/{args.epoch} bit:{args.bit} training...")
    net.train()
    train_loss = 0
    for image, label, i in train_loader:
      image = image.to(args.device)
      label = label.to(args.device)
      
      optimizer.zero_grad()
      u = net(image)
      loss = criterion(u,i)
      train_loss += loss.item()
      
      loss.backward()
      optimizer.step()
      
    train_loss = train_loss / len(train_loader)
    logger.info(f"loss:{train_loss:.3f}")
    
    if epoch % 10 == 0:
      mAP = validate(args, test_loader, database_loader, net)
      if Best_mAP < mAP:
        Best_mAP = mAP
        torch.save(net.state_dict(), os.path.join('save', 'CNNH', f'{args.bit}.pth'))
      logger.info(f'[{epoch}] retrieval mAP: {mAP:.4f},Best mAP: {Best_mAP:.4f}')
      
  
if __name__ == '__main__':
  logger.add(os.path.join('log','CNNH','{time}.log'), rotation='500 MB', level="INFO")
  args = load_config()
  seed_setting(args.seed)
  train_val(args)