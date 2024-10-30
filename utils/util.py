import torch
import os
import random
import numpy as np
from torch.autograd import Variable
import time
from datetime import datetime
import argparse
from tqdm import tqdm

def seed_setting(seed=2021):
  """
  固定随机种子，使得每次训练结果相同，方便对比模型效果
  """
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # torch.backends.cudnn.benchmark = False # False make training process too slow!
  torch.backends.cudnn.deterministic = True

def compute_result(dataloader, net, device):
  bs, clses = [], []
  net.eval()
  with torch.no_grad():
    for img, cls, _ in tqdm(dataloader):
      clses.append(cls)
      bs.append(net(img.to(device)).data.cpu())
  return torch.sign(torch.cat(bs)), torch.cat(clses)

def compute_mAP(train_binary, test_binary, train_label, test_label):
  """
  compute mAP by searching testset from trainset
  """
  for x in train_binary, test_binary, train_label, test_label: x.long()
  
  AP = []
  Ns = torch.arange(1, train_binary.size(0)+1)
  for i in range(test_binary.size(0)):
    query_label, query_binary = test_label[i], test_binary[i]
    _, query_result = torch.sum((query_binary != train_binary).long(), dim=1).sort()
    correct = (query_label == train_label[query_result]).float()
    P = torch.cumsum(correct, dim=0) / Ns
    AP.append(torch.sum(P * correct) / torch.sum(correct))
  mAP = torch.mean(torch.Tensor(AP))
  return mAP

def calculate_map(qu_B, re_B, qu_L, re_L):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

def calculate_hamming(B1,B2):
  q = B2.shape[1]
  distH = 0.5 * (q - np.dot(B1, B2.transpose()))
  return distH

def getLogName():
  # 获取当前时间戳
  current_timestamp = time.time()

  # 将时间戳转换为 datetime 对象
  current_datetime = datetime.fromtimestamp(current_timestamp)

  # 格式化为字符串
  formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
  
  return formatted_time

def validate(args, test_loader, dataset_loader, net):
  test_binary, test_label = compute_result(test_loader, net, args.device)
  ds_binary, ds_label = compute_result(dataset_loader, net, args.device)
  mAP = calculate_map(test_binary.numpy(), ds_binary.numpy(), test_label.numpy(), ds_label.numpy())
  return mAP

def load_config():
  """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
  parser = argparse.ArgumentParser(description='DSH')
  
  ## Net basic params
  parser.add_argument('--lr',type=float,default=1e-4)
  parser.add_argument('--epoch',type=int, default=250)
  parser.add_argument('--alpha',type=float,default=0.05)
  parser.add_argument('-w','--weight_decay',type=float,default=10 ** -5)
  parser.add_argument('-m','--momentum',type=float,default=0.9)
  parser.add_argument('-b','--bit', type=int, default=48,help='Binary hash code length.(default: 48)')
  parser.add_argument('--batch_size',type=int, default=256)
  parser.add_argument('--checkpoint', type=int, default=50, help='checkpointing after batches')
  parser.add_argument('--mu',type=float,default=1e-2)
  parser.add_argument('--nu',type=float,default=1)
  parser.add_argument('--eta',type=float,default=1e-2)
  parser.add_argument('--gamma',type=float,default=200)
  
  # Data params
  parser.add_argument('-d','--dataset',type=str,default='cifar10',help='coco/nuswide/flick/imagenet/cifar10')
  parser.add_argument('-g','--gpu',type=int, default=0,help='Using gpu.(default: False)')
  parser.add_argument('--resize_size',type=int, default=256,help='transform the size of image')
  parser.add_argument('--crop_size',type=int, default=224,help='transform the size of image')
  parser.add_argument('--n_class',type=int,default=10)
  parser.add_argument('--num_train',type=int,default=5000)
  parser.add_argument('--num_query',type=int,default=1000)
  parser.add_argument('--num_database',type=int,default=54000)
  parser.add_argument('--num_workers',type=int,default=4)
  parser.add_argument('--iter',type=int,default="150",help="iteration times")
  
  # Flag params
  parser.add_argument('--seed',type=int, default=2021)
  # CNNH
  parser.add_argument("--T",type=int, default=10)
    
  args = parser.parse_args()
  
  
  if args.gpu is None:
    args.device = torch.device('cpu')
  else:
    args.device = torch.device("cuda:%d" % args.gpu)
    torch.cuda.set_device(args.gpu)
  return args