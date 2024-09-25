import argparse
import torch
from utils.util import seed_setting
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
  parser.add_argument('--lr',type=float,default=1e-5)
  parser.add_argument('--epoch',type=int, default=250)
  parser.add_argument('--alpha',type=float,default=0.05)
  parser.add_argument('--bit', type=int, default=48,help='Binary hash code length.(default: 48)')
  parser.add_argument('--batch_size',type=int, default=64)
  
  # Data params
  parser.add_argument('--dataset',type=str,default='imagenet',help='coco/nuswide/flick/imagenet')
  parser.add_argument('-g','--gpu',type=int, default=0,help='Using gpu.(default: False)')
  
  # Flag params
  parser.add_argument('--seed',type=int, default=2021)
  args = parser.parse_args()
  
  if args.gpu is None:
    args.device = torch.device('cpu')
  else:
    args.device = torch.device("cuda:%d" % args.gpu)
    torch.cuda.set_device(args.gpu)
  return args

if __name__ == '__main__':
  args = load_config()
  seed_setting(args.seed)
  