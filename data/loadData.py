from .cifar import load_cifar_data
from .nus_wide import load_nuswide_data
def get_data(args):
  config_dataset(args)
  if args.dataset == "cifar10":
    return load_cifar_data(args.crop_size, args.num_train, args.num_query, args.num_workers, args.batch_size, args.n_class)
  elif args.dataset == "nuswide_21" or args.dataset == "nuswide_21_m":
    return load_nuswide_data(21, args.num_train, args.num_query, args.batch_size, args.num_workers)
  
  
  
def config_dataset(args):
  if "cifar10" in args.dataset:
    args.n_class = 10
    # 1. 5000 1000 54000
    # 2. 50000 10000 50000
    args.num_train = 5000
    args.num_query = 1000
    args.num_database = 54000
  elif "nuswide_21" in args.dataset or "nuswide_21_m" in args.dataset:
    args.num_train = 10500
    args.num_query = 2100
    args.num_database = 193734
    args.n_class = 21
  elif "nuswide_81" in args.dataset or "nuswide_81_m" in args.dataset:
    args.n_class = 81
  elif "coco" in args.dataset:
    args.n_class = 80
  elif "imagenet" in args.dataset:
    args.n_class = 100
  return args