"""Time-axis mean + population variance, mean first; no padding is present."""
import torch

def pool_mean_variance(hidden):
    if hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[1]<1 or hidden.shape[2]!=1024:
        raise ValueError(f'Expected one unpadded (1,T,1024) hidden sequence; got {tuple(hidden.shape)}')
    mean=hidden.mean(dim=1)
    variance=hidden.var(dim=1,unbiased=False)
    return torch.cat((mean,variance),dim=-1)[0]
