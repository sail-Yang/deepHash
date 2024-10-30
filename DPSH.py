import torch
from data.loadData import get_data
from utils.util import *
import torch.nn as nn
from loguru import logger
from model.load_model import load_model

def DPSH_loss(y_hat, label, alpha):
  s = (label @ label.t() > 0).float()
  theta = y_hat @ y_hat.t() * 0.5
  # likelihood_loss = -(s*theta - (1 + theta.exp()).log()) + theta.clamp(min=0) - s * theta
  likelihood_loss = (1 + (-(theta.abs())).exp()).log() + theta.clamp(min=0) - s * theta
  likelihood_loss = likelihood_loss.mean()
  
  quantization_loss = alpha * (y_hat - y_hat.sign()).pow(2).mean()
  return likelihood_loss + quantization_loss


if __name__ == '__main__':
  logger.add(os.path.join('log','DPSH','{time}.log'), rotation='500 MB', level="INFO")
  args = load_config()
  seed_setting(args.seed)
  # 获取数据
  train_loader, test_loader, dataset_loader =  get_data(args)
  Best_mAP = 0
  # 构造模型
  net = load_model(args.model, args.bit, args.device)
    
  optimizer = torch.optim.RMSprop(
    net.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
  )

  logger.info(f"[DPSH] model: {args.model}, dataset: {args.dataset}, bit: {args.bit}, lr: {args.lr}, gpu: {args.device}, batch_size: {args.batch_size}, epoch: {args.epoch}, alpha: {args.alpha} training...")
  for epoch in range(1,args.epoch+1):
    net.train()
    train_loss = 0  
    for img, cls, ind in train_loader:
      img , cls = img.to(args.device), cls.to(args.device)
      optimizer.zero_grad()
      y_hat = net(img)
      loss = DPSH_loss(y_hat, cls.float(), args.alpha)
      train_loss += loss.item()
      
      loss.backward()
      optimizer.step()
  
    train_loss = train_loss / len(train_loader)
    logger.info(f'[{epoch}] loss: {train_loss:.4f}')
    
    if epoch % args.checkpoint == 0:
      mAP = validate(args, test_loader, dataset_loader, net)
      if Best_mAP < mAP:
        Best_mAP = mAP
        torch.save(net.state_dict(), os.path.join('save', 'DPSH', f'{args.bit}.pth'))
      logger.info(f'[{epoch}] retrieval mAP: {mAP:.4f},Best mAP: {Best_mAP:.4f}')