import cifar
def get_data(args):
  if args.dataset == "cifar10":
    return cifar.load_data(args)