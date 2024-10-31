import argparse
import torch
from utils.util import *
import torch.nn as nn
import torchvision
from loguru import logger
from data.loadData import get_data
from torchvision.models import VGG16_Weights
import os

class DSDH_loss(torch.nn.Module):
    def __init__(self, eta, bit):
      super(DSDH_loss, self).__init__()
      self.eta = eta
      self.bit = bit

    def forward(self, U_batch, U, S, B):
      # print("U.shape",U.shape)
      # print("U_batch.shape",U_batch.shape)
      inner_product = U.t() @ U_batch * 0.5
      # prevent exp overflow
      inner_product = torch.clamp(inner_product, min=-100, max=50)
      # print("theta.shape",inner_product.shape)
      # print("S.shape",S.shape)
      likelihood_loss = torch.log(1 + torch.exp(inner_product)) - S * inner_product

      likelihood_loss = likelihood_loss.mean()
      # print("likelihood_loss",likelihood_loss)

      # Regularization loss
      reg_loss = (B - U_batch).pow(2).mean()
      

      loss = likelihood_loss + self.eta * reg_loss
      return loss

def load_model(bit):
  """
  get model
  
  Args
    bit: hash code bits length
  """
  model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
  hash_layer = nn.Sequential(
    nn.Linear(4096, bit),
    nn.Tanh()
  )
  # 删除classifier最后一层4096x1000
  model.classifier = model.classifier[:-1]
  model.classifier.add_module('hash_layer', hash_layer)
  return model

def solve_dcc(W, Y, U, B, eta, mu):
    """
    DCC.
    """
    for i in range(B.shape[0]):
        P = W @ Y + eta / mu * U

        p = P[i, :]
        w = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

    return B

def compute_result(dataloader, net):
  bs, clses = [], []
  net.eval()
  with torch.no_grad():
    for img, cls, _ in dataloader:
      clses.append(cls)
      bs.append(net(img.cuda()).data.cpu())
  return torch.sign(torch.cat(bs)), torch.cat(clses)

if __name__ == "__main__":
  logger.add(os.path.join('log','DSDH','{time}.log'), rotation='500 MB', level="INFO")
  args = load_config()
  seed_setting(args.seed)
  net = load_model(args.bit)
  net.to(args.device)
  optimizer = torch.optim.RMSprop(
    net.parameters(),
    lr=1e-5,
    weight_decay=1e-5,
  )
  criterion = DSDH_loss(args.eta, args.bit)
  # 获取数据
  train_loader, test_loader, dataset_loader = get_data(args)
  
  Best_mAP = 0
  
  # init
  N = len(train_loader.dataset)
  B = torch.randn(args.bit, N).sign().to(args.device)
  U = torch.zeros(args.bit, N).to(args.device)
  
  logger.info(f"[DSDH] model: {args.model}, dataset: {args.dataset}, bit: {args.bit}, lr: {args.lr}, gpu: {args.device}, batch_size: {args.batch_size}, epoch: {args.epoch}, alpha: {args.alpha} training...")
  
  for epoch in range(args.epoch):
    logger.info(f'[{epoch}] bit: {args.bit}, dataset: cifar10, training...')
    net.train()
    
    train_loss = 0
    for img, label, ind in train_loader:
      img = img.to(args.device)
      label = label.to(args.device).float()
      optimizer.zero_grad()
      u = net(img)
      U_batch = u.t()
      U[:, ind] = U_batch.data
      S = (label @ label.t() > 0).float()
      Y = S.t()
      loss = criterion(U_batch, U[:,ind], S, B[:,ind])
      train_loss += loss.item()
      
      loss.backward()
      optimizer.step()
  
    
    train_loss = train_loss / len(train_loader)
    logger.info(f'loss: {train_loss:.3f}')
    if (epoch + 1) % args.checkpoint == 0:
      train_binary, train_label = compute_result(train_loader, net)
      test_binary, test_label = compute_result(test_loader, net)
      mAP = compute_mAP(train_binary, test_binary, train_label, test_label)
      logger.info(f'[{epoch}] retrieval mAP: {mAP:.4f}')