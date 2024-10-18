from .cifar import load_data
def get_data(args):
  if args.dataset == "cifar10":
    return load_data(args)