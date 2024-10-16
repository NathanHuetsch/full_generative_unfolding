import torch
import torch.nn as nn
from torchdiffeq import odeint
import time


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], self.dims_x))
        self.network = nn.Sequential(*layers)

    def train(self, data_x, data_c, weights=None):
        if weights is None:
            weights = torch.ones((data_x.shape[0]))
        dataset = torch.utils.data.TensorDataset(data_x, data_c, weights)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params["batch_size"],
                                             shuffle=True)
        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * n_epochs)
        print(f"Training CFM for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        for epoch in range(n_epochs):
            losses = []
            for i, batch in enumerate(loader):
                x_hard, x_reco, weight = batch
                optimizer.zero_grad()
                loss = self.batch_loss(x_hard, x_reco, weight)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            if epoch % int(n_epochs / 5) == 0:
                print(
                    f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        print(
            f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data_c):
        predictions = []
        with torch.no_grad():
            for batch in torch.split(data_c, self.params["batch_size_sample"]):
                unfold_cfm = self.sample(batch).detach()
                predictions.append(unfold_cfm)
        predictions = torch.cat(predictions)
        return predictions


class CFM(Model):
    def __init__(self, dims_x, dims_c, params):
        super().__init__()
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.params = params
        self.dims_in = self.dims_x + self.dims_c + 1
        self.init_network()

    def sample(self, c):
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            v = self.network(torch.cat([t, x_t, c], dim=-1))
            return v

        x_0 = torch.randn((batch_size, self.dims_x)).to(device, dtype=dtype)
        x_t = odeint(func=net_wrapper, y0=x_0, t=torch.tensor([0., 1.]).to(device, dtype=dtype))
        return x_t[-1]

    def batch_loss(self, x, c, weight):
        x_0 = torch.randn((x.size(0), self.dims_x)).to(x.device)
        t = torch.rand((x.size(0), 1)).to(x.device)
        x_t = (1 - t) * x_0 + t * x
        x_t_dot = x - x_0
        v_pred = self.network(torch.cat([t, x_t, c], dim=-1))
        cfm_loss = ((v_pred - x_t_dot) ** 2 * weight.unsqueeze(-1)).mean()
        return cfm_loss


class DiDi(Model):
    def __init__(self, dims_x, dims_c, params):
        super().__init__()
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.params = params
        self.dims_in = self.dims_x + 1
        self.init_network()

    def sample(self, c):
        dtype = c.dtype
        device = c.device

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            v = self.network(torch.cat([t, x_t], dim=-1))
            return v

        x_t = odeint(func=net_wrapper, y0=c, t=torch.tensor([0., 1.]).to(device, dtype=dtype))
        return x_t[-1]

    def batch_loss(self, x, c, weight):
        t = torch.rand((x.size(0), 1)).to(x.device)
        x_t = (1 - t) * c + t * x
        x_t_dot = x - c
        v_pred = self.network(torch.cat([t, x_t], dim=-1))
        cfm_loss = ((v_pred - x_t_dot) ** 2 * weight.unsqueeze(-1)).mean()
        return cfm_loss
