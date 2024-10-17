import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as dsets
from torchvision import transforms

def load_data(num_query=100, num_train=500, batch_size=256, num_workers=4):
  """
  load CIFAR10 data
  
  Args
    num_query
  """
  return cifar_dataset(batch_size=batch_size, train_size=num_train, test_size=num_query, num_workers=num_workers)

def cifar_dataset(batch_size,  crop_size=224, train_size=500, test_size=100, num_workers=4):
  trans_train = transforms.Compose([
    transforms.Resize(crop_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  trans_test = transforms.Compose([
    transforms.Resize(crop_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  
  cifar_dataset_root = "dataset/cifar/"
  # get Dataset
  train_dataset = MyCIFAR10(root=cifar_dataset_root, train=True, transform=trans_train,download=True)
  test_dataset = MyCIFAR10(root=cifar_dataset_root, train=False, transform=trans_test)
  database_dataset = MyCIFAR10(root=cifar_dataset_root, train=False, transform=trans_test)
  # 合并训练集和测试集
  X = np.concatenate((train_dataset.data, test_dataset.data))
  L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))
  
  # 切割标签为测试集、训练集、database
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
  
  # 分割数据
  train_dataset.data = X[train_index]
  train_dataset.targets = L[train_index]
  test_dataset.data = X[test_index]
  test_dataset.targets = L[test_index]
  database_dataset.data = X[database_index]
  database_dataset.targets = L[database_index]

  print("train_dataset", train_dataset.data.shape[0])
  print("test_dataset", test_dataset.data.shape[0])
  print("database_dataset", database_dataset.data.shape[0])

  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)

  test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

  database_loader = DataLoader(dataset=database_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)

  return train_loader, test_loader, database_loader, train_index.shape[0], test_index.shape[0], database_index.shape[0]
  


class MyCIFAR10(dsets.CIFAR10):
  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)
    img = self.transform(img)
    # 标签转One-hot编码
    target = np.eye(10, dtype=np.int8)[np.array(target)]
    return img, target, index