import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def kl_loss(mu,log_sigma):
    kl_loss = 0.5 * torch.mean(mu.pow(2) + (2*log_sigma).exp() - 2*log_sigma - 1)
    return(kl_loss)