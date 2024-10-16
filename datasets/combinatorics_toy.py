import torch


class CombinatoricsToy:

    def __init__(self, params):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()

    def apply_detector(self, x_gen):
        detector_effect = torch.randn_like(x_gen) * self.params["detector_sigma"] + self.params["detector_mu"]
        return x_gen + detector_effect

    def init_dataset(self):
        self.top_gen = (torch.randn(self.params["n_mc"]) * self.params["top_sigma"] + self.params["top_mu"]).unsqueeze(-1)
        self.W_gen = (torch.randn(self.params["n_mc"]) * self.params["W_sigma"] + self.params["W_mu"]).unsqueeze(-1)
        b_gen = self.top_gen - self.W_gen
        q1_gen = torch.rand(self.params["n_mc"]).unsqueeze(-1)*self.W_gen
        q2_gen = self.W_gen-q1_gen

        self.mc_gen = torch.cat([b_gen, q1_gen, q2_gen], dim=1).to(self.device)

        self.mc_rec_matched = self.apply_detector(self.mc_gen)
        self.mc_rec = self.mc_rec_matched.sort(dim=1, descending=True)[0]
        self.mc_rec.to(self.device)

        self.data_gen = self.mc_gen.clone()
        self.data_rec = self.mc_rec.clone()
