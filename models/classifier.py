import torch
import torch.nn as nn
import time


class Classifier(nn.Module):
    def __init__(self, dims_in, params):
        super().__init__()
        self.dims_in = dims_in
        self.params = params
        self.init_network()

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], 1))
        self.network = nn.Sequential(*layers)

    def batch_loss(self, x, y, w):
        pred = self.network(x).squeeze()
        loss = torch.nn.BCEWithLogitsLoss(weight=w)(pred, y)
        return loss

    def train(self, data_true, data_false, weights_true=None, weights_false=None, balanced=True):
        if weights_true is None:
            weights_true = torch.ones((data_true.shape[0])).to(data_true.device, dtype=data_true.dtype)
        if weights_false is None:
            weights_false = torch.ones((data_false.shape[0])).to(data_true.device, dtype=data_true.dtype)

        dataset_true = torch.utils.data.TensorDataset(data_true, weights_true)
        loader_true = torch.utils.data.DataLoader(dataset_true, batch_size=self.params["batch_size"],
                                             shuffle=True)
        dataset_false = torch.utils.data.TensorDataset(data_false, weights_false)
        loader_false = torch.utils.data.DataLoader(dataset_false, batch_size=self.params["batch_size"],
                                                  shuffle=True)
        if not balanced:
            class_weight = len(data_true)/len(data_false)
            print(f"    Training with unbalanced training set with weight {class_weight}")
        else:
            class_weight = 1

        n_epochs = self.params["n_epochs"] * int(class_weight)
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        print(f"Training classifier for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        for epoch in range(n_epochs):
            losses = []
            for i, (batch_true, batch_false) in enumerate(zip(loader_true, loader_false)):
                x_true, weight_true = batch_true
                x_false, weight_false = batch_false
                label_true = torch.ones((x_true.shape[0])).to(x_true.device)
                label_false = torch.zeros((x_false.shape[0])).to(x_false.device)
                optimizer.zero_grad()
                loss = self.batch_loss(x_true, label_true, weight_true) * class_weight
                loss += self.batch_loss(x_false, label_false, weight_false)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % int(n_epochs / 5) == 0:
                print(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        print(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data, return_weights=True):
        predictions = []
        with torch.no_grad():
            for batch in torch.split(data, self.params["batch_size_sample"]):
                pred = self.network(batch).squeeze().detach()
                predictions.append(pred)
        predictions = torch.cat(predictions)
        return predictions.exp().clip(0, 30) if return_weights else torch.sigmoid(predictions)
