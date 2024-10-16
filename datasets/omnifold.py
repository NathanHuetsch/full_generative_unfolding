import torch
import numpy as np


class Omnifold:

    def __init__(self, params):
        self.params = params

        self.path = params["path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.apply_preprocessing()

    def init_dataset(self):
        try:
            dataset = np.load(self.path)
        except:
            dataset = np.load(self.params["path2"])
        n_data = self.params["n_data"]
        self.mc_gen = torch.tensor(dataset["pythia_gen"][:n_data, :5]).float().to(self.device)
        self.mc_rec = torch.tensor(dataset["pythia_rec"][:n_data, :5]).float().to(self.device)
        print(self.mc_gen.shape)
        data = np.load("/remote/gpu07/huetsch/theofold/data/analysis_transfer.npz")
        self.data_rec = torch.tensor(data["reco"][:n_data, [0, 2, 5, 1, 3]]).float().to(self.device)
        self.data_gen = torch.tensor(data["truth"][:n_data, [0, 2, 5, 1, 3]]).float().to(self.device)

    def apply_preprocessing(self, reverse=False):

        if not reverse:
            # add noise to the jet multiplicity to smear out the integer structure
            noise = torch.rand_like(self.mc_rec[:, 1]) - 0.5
            self.mc_rec[:, 1] = self.mc_rec[:, 1] + noise
            noise = torch.rand_like(self.mc_gen[:, 1]) - 0.5
            self.mc_gen[:, 1] = self.mc_gen[:, 1] + noise
            noise = torch.rand_like(self.data_rec[:, 1]) - 0.5
            self.data_rec[:, 1] = self.data_rec[:, 1] + noise
            noise = torch.rand_like(self.data_gen[:, 1]) - 0.5
            self.data_gen[:, 1] = self.data_gen[:, 1] + noise

            # standardize events
            self.rec_mean = self.mc_rec.mean(0)
            self.rec_std = self.mc_rec.std(0)
            self.gen_mean = self.mc_gen.mean(0)
            self.gen_std = self.mc_gen.std(0)

            self.mc_gen = (self.mc_gen - self.gen_mean)/self.gen_std
            self.mc_rec = (self.mc_rec - self.rec_mean)/self.rec_std

            self.data_gen = (self.data_gen - self.gen_mean) / self.gen_std
            self.data_rec = (self.data_rec - self.rec_mean) / self.rec_std

            self.mc_gen = self.mc_gen[:, :5]
            self.mc_rec = self.mc_rec[:, :5]
            self.data_gen = self.data_gen[:, :5]
            self.data_rec = self.data_rec[:, :5]

        else:
            if not hasattr(self, "rec_mean"):
                raise ValueError("Trying to run reverse preprocessing before forward preprocessing")

            # undo standardization
            self.mc_gen = self.mc_gen * self.gen_std + self.gen_mean
            self.mc_rec = self.mc_rec * self.rec_std + self.rec_mean

            self.data_gen = self.data_gen * self.gen_std + self.gen_mean
            self.data_rec = self.data_rec * self.rec_std + self.rec_mean

            # round jet multiplicity back to integers
            self.mc_rec[:, 1] = torch.round(self.mc_rec[:, 1])
            self.data_rec[:, 1] = torch.round(self.data_rec[:, 1])
            self.mc_gen[:, 1] = torch.round(self.mc_gen[:, 1])
            self.data_gen[:, 1] = torch.round(self.data_gen[:, 1])
