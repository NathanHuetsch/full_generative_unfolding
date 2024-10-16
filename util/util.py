import torch


def get_quantile_bins(data, n_bins=50, lower=0.001, upper=0.001):
    return torch.linspace(
        torch.nanquantile(data, lower), torch.nanquantile(data, 1 - upper), n_bins + 1
    )