from .cifar import load_data
def get_data(args):
  config_dataset(args)
  if args.dataset == "cifar10":
    return load_data(args.crop_size, args.num_train, args.num_query, args.num_workers, args.batch_size, args.n_class)

def config_dataset(args):
  if "cifar10" in args.dataset:
    args.n_class = 10
    # 1. 5000 1000 54000
    # 2. 50000 10000 50000
    args.num_train = 5000
    args.num_query = 1000
    args.num_database = 54000
  elif "nuswide_21" in args.dataset or "nuswide_21_m" in args.dataset:
    args.n_class = 21
  elif "nuswide_81" in args.dataset or "nuswide_81_m" in args.dataset:
    args.n_class = 81
  elif "coco" in args.dataset:
    args.n_class = 80
  elif "imagenet" in args.dataset:
    args.n_class = 100
  return args