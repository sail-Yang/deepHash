import torch
from utils.util import *
import torch.nn as nn
from loguru import logger
from data.loadData import get_data
from model.alexnet import AlexNet

def get_database_labels(device, dataset_loader):
  clses = []
  for _,cls,_ in tqdm(dataset_loader):
    clses.append(cls)
  return torch.cat(clses).to(device).float()

def get_S_matrix(query_labels, database_labels):
  """
  获取相似矩阵
  query_labels: m x c
  databse_labels: n x c
  """
  query_labels = query_labels.float()
  database_labels = database_labels.float()
  S = (database_labels @ query_labels.t() > 0).float()
  # soft constraint
  r = S.sum() / (1 - S).sum()
  S = S * (1 + r) - r
  return S
  
def ADSH_loss(u, V, S, bit, selected_indexs, ind, num_train):
  """
  计算ADSH的损失函数
  u: 卷积网络输出, query data近似哈希码
  V: database哈希码
  S: 相似矩阵
  bit: 哈希码长度
  selected_indexs: query data在database中的索引
  ind: 索引
  num_train: 训练集数量
  """
  square_loss = (V @ u.t() - bit * S).pow(2)
  quantization_loss = args.gamma * (V[selected_indexs[ind]] - u).pow(2)
  return (square_loss.sum() + quantization_loss.sum()) / (num_train * u.size(0))

if __name__ == '__main__':
  logger.add(os.path.join('log','ADSH','{time}.log'), rotation='500 MB', level="INFO")
  args = load_config()
  seed_setting(args.seed)
  
  # 从database中选择num_samples个样本作为query dataset to train
  num_samples = 2000
  args.epoch = 10
  
  net = AlexNet(args.bit).to(args.device)
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
  train_loader, test_loader, dataset_loader = get_data(args)
  Best_mAP = 0
  
  database_labels = get_database_labels(args.device, dataset_loader)
  V = torch.zeros((args.num_database, args.bit)).to(args.device)
  
  for iter in range(args.iter):
    logger.info(f"[{iter+1}/{args.iter}] bit: {args.bit} dataset: {args.dataset} training ... ")
    net.train()
    # 从 num_dataset 个数据中随机选择 num_samples 个样本的索引，用作生成query dataset
    selected_indexs = np.random.permutation(range(args.num_database))[0:num_samples]
    # 从databse生成query data 用作训练集
    if "cifar10" in args.dataset:
      train_loader.dataset.data = np.array(dataset_loader.dataset.data)[selected_indexs]
      train_loader.dataset.targets = np.array(dataset_loader.dataset.targets)[selected_indexs]
    else:
      train_loader.dataset.imgs = np.array(dataset_loader.dataset.imgs)[selected_indexs].tolist()
    # query label
    query_labels = database_labels[selected_indexs]
    Sim = get_S_matrix(query_labels=query_labels, database_labels=database_labels)
    U = torch.zeros((num_samples, args.bit)).to(args.device)
    
    train_loss = 0 
    for epoch in range(1,args.epoch+1):
      for img, label, ind in train_loader:
        img = img.to(args.device)
        label = label.to(args.device)
        optimizer.zero_grad()
        S = get_S_matrix(query_labels=label, database_labels=database_labels)
        u = net(img)
        u = u.tanh()
        U[ind, :] = u.data
        loss = ADSH_loss(u=u, V=V, S=S, bit=args.bit, selected_indexs=selected_indexs, ind = ind, num_train=args.num_train)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
      
    train_loss = train_loss / len(train_loader) / epoch
    logger.info(f'[{epoch}] loss: {train_loss:.4f}') 
    
    # learn binary codes
    barU = torch.zeros((args.num_database, args.bit)).to(args.device)
    barU[selected_indexs, :] = U
    # calculate Q
    Q = -2 * args.bit * Sim @ U - 2 * args.gamma * barU
    for k in range(args.bit):
        sel_ind = np.setdiff1d([ii for ii in range(args.bit)], k)
        V_ = V[:, sel_ind]
        U_ = U[:, sel_ind]
        Uk = U[:, k]
        Qk = Q[:, k]
        # formula 10
        V[:, k] = -(2 * V_ @ (U_.t() @ Uk) + Qk).sign()
      
    if epoch % args.checkpoint == 0:
      mAP = validate(args, test_loader, dataset_loader, net)
      if Best_mAP < mAP:
        Best_mAP = mAP
        torch.save(net.state_dict(), os.path.join('save', 'ADSH', f'{args.bit}.pth'))
      logger.info(f'[{epoch}] retrieval mAP: {mAP:.4f},Best mAP: {Best_mAP:.4f}')
      
      
        
  