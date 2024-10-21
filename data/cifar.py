import os
import pickle
import sys

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as dsets

def load_data(args):
  """
  load cifar10 dataset
  
  Args
    args: argumetns
  
  Returns
    train_loader, test_loader, database_loader, train_index.shape[0], test_index.shape[0], database_index.shape[0]
  """
  cifar_dataset_root = '/data2/fyang/dataset/cifar10/'
  transform = transforms.Compose([
      transforms.Resize(args.crop_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  train_dataset = CIFAR10(root=cifar_dataset_root,train=True,transform=transform,download=True)
  test_dataset = CIFAR10(root=cifar_dataset_root,train=False,transform=transform)
  database_dataset = CIFAR10(root=cifar_dataset_root,train=False,transform=transform)
  
  X = np.concatenate((train_dataset.data, test_dataset.data))
  L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))
  
  # 每个类别的训练集和测试集数量
  train_size = 500
  test_size = 100
  # train_size = 5000
  # test_size = 1000
  # 随机划分三个数据集
  first = True
  for label in range(10):
    index = np.where(L == label)[0]
    N = index.shape[0]
    perm = np.random.permutation(N)
    index = index[perm]
    if first:
        test_index = index[:test_size]
        train_index = index[test_size: train_size + test_size]
        database_index = index[train_size + test_size:]
    else:
        test_index = np.concatenate((test_index, index[:test_size]))
        train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
        database_index = np.concatenate((database_index, index[train_size + test_size:]))
    first = False
  
  if train_size == 5000:
    database_index = train_index
  else:
    pass
  
  train_dataset.data = X[train_index]
  train_dataset.targets = L[train_index]
  test_dataset.data = X[test_index]
  test_dataset.targets = L[test_index]
  database_dataset.data = X[database_index]
  database_dataset.targets = L[database_index]
  print("train_dataset", train_dataset.data.shape[0])
  print("test_dataset", test_dataset.data.shape[0])
  print("database_dataset", database_dataset.data.shape[0])
  
  train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
  test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
  database_loader = DataLoader(dataset=database_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
  
  return train_loader, test_loader, database_loader, train_index.shape[0], test_index.shape[0], database_index.shape[0]

class CIFAR10(dsets.CIFAR10):
  """
  Cifar10 dataset
  """
  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)
    img = self.transform(img)
    # 将标签转为独热编码
    target = np.eye(10, dtype=np.int8)[np.array(target)]
    return img, target, index