import os
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import scipy.io as sio

# 允许 PIL 加载被截断或不完整的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_nuswide_data(root, tc, num_train, num_query, batch_size, num_workers):
  """
  Loading nus-wide dataset
  
  Args:
    root: dataset root
    tc(int): Top class
    num_query(int): Number of query images
    num_train(int): Number of training images
    batch_size(int): Batch size
    num_workers(int): Number of workers
  
  Returns:
    query_dataloader, train_dataloader, retrieval_dataloader: Data loader
  """
  query_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  
  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  
  root = os.path.join(root,"NUS-WIDE")
  if tc == 21:
    imgs_path = os.path.join(root, "nus-wide-tc21-IAll", "IAll.npy")
    labels_path = os.path.join(root,"nus-wide-tc21-lall.mat")
    split_path = os.path.join(root, "nuswide-tc21.100pc.500pc")
    # 获取所有图像和标签的numpy格式
    all_imgs = np.load(imgs_path)
    all_labels = sio.loadmat(labels_path)['LAll']
    
    # 划分query集合
    split_query_path = os.path.join(split_path, "idx_test.npy")
    idx_query = np.load(split_query_path)
    imgs_query = all_imgs[idx_query]
    labels_query = all_labels[idx_query]
    
    query_dataset = NusWideDatasetTC21(
      root,
      imgs_query,
      labels_query,
      transform=query_transform,
    )
    
    # 划分train集合
    split_train_path = os.path.join(split_path, "idx_labeled.npy")
    idx_train = np.load(split_train_path)
    imgs_train = all_imgs[idx_train]
    labels_train = all_labels[idx_train]
    
    train_dataset = NusWideDatasetTC21(
      root,
      imgs_train,
      labels_train,
      transform=train_transform,
      train=True,
      num_train=num_train,
    )
    
    # 划分retrieval集合
    split_retrieval_path = os.path.join(split_path, "idx_ret.npy")
    idx_retrieval = np.load(split_retrieval_path)
    imgs_retrieval = all_imgs[idx_retrieval]
    labels_retrieval = all_labels[idx_retrieval]
    
    retrieval_dataset = NusWideDatasetTC21(
      root,
      imgs_retrieval,
      labels_retrieval,
      transform=query_transform,
    )
    
    print("train_dataset", train_dataset.data.shape[0])
    print("query_dataset", query_dataset.data.shape[0])
    print("retrieval_dataset", retrieval_dataset.data.shape[0])
    
    query_dataloader = DataLoader(
      query_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
    )
    
    train_dataloader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
    )
    
    retrieval_dataloader = DataLoader(
      retrieval_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
    )
    
    return train_dataloader, query_dataloader, retrieval_dataloader

class NusWideDatasetTC21(Dataset):
  """
  NUS-WIDE dataset 21 classes
  
  Args:
    root(str): Path of image files
    imgs(numpy): Path of txt file containing image file name
    labels(numpy): Path of txt file containing image label
    transform(callable, optional): Transform images.
    train(bool, optional): Return training dataset.
    num_train(int, optional): Number of training data.
  """
  def __init__(self, root, imgs, labels, transform=None, train=False, num_train=None):
    self.root = root
    self.transform = transform
    self.data = imgs
    self.targets = labels
    
    if train is True:
      perm_index = np.random.permutation(len(self.data))[:num_train]
      self.data = self.data[perm_index]
      self.targets = self.targets[perm_index]
    
  def __getitem__(self, index):
    img = self.data[index]
    if self.transform is not None:
      img = Image.fromarray(img)
      img = self.transform(img)
    return img, self.targets[index], index

  def __len__(self):
    return len(self.data)

  def get_onehot_targets(self):
    return torch.from_numpy(self.targets).float()