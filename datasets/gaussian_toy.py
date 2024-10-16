import torch


class GaussianToy:

    def __init__(self, params):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()

    def apply_detector(self, x_gen):
        detector_effect = torch.randn_like(x_gen) * self.params["detector_sigma"] + self.params["detector_mu"]
        return x_gen + detector_effect

    def init_dataset(self):
        self.mc_gen = (torch.randn(self.params["n_mc"], self.params["n_dim"]) * self.params["mc_sigma"] + self.params["mc_mu"]).to(self.device)
        self.mc_rec = self.apply_detector(self.mc_gen).to(self.device)

        self.data_gen = (torch.randn(self.params["n_data"], self.params["n_dim"]) * self.params["data_sigma"] + self.params["data_mu"]).to(self.device)
        #data_mus = torch.rand(self.params["n_dim"])*10. - 5.
        #data_sigmas = torch.rand(self.params["n_dim"]) * 2. + 3
        #print(data_mus, data_sigmas)
        #self.data_gen = (
        #            torch.randn(self.params["n_data"], self.params["n_dim"]) * data_sigmas + data_mus).to(self.device)
        self.data_signal_rec = self.apply_detector(self.data_gen).to(self.device)

        if self.params["n_background"] != 0:
            self.data_background_rec = (torch.randn(self.params["n_background"], self.params["n_dim"]) * self.params["background_sigma"] + self.params["background_mu"]).to(self.device)
            self.mc_background_rec = (torch.randn(self.params["n_background"], self.params["n_dim"]) * self.params["background_sigma"] + self.params["background_mu"]).to(self.device)
            self.data_rec = torch.cat([self.data_signal_rec, self.data_background_rec])
        else:
            self.data_rec = self.data_signal_rec

        if self.params["mc_rec_cut"]:
            cut_position = self.params["mc_rec_cut_position"]
            self.mc_rec_mask = ~((self.mc_rec > cut_position[0]).squeeze() * (self.mc_rec < cut_position[1]).squeeze())
            #self.mc_rec = self.mc_rec[self.mc_rec_mask]
            #self.mc_gen = self.mc_gen[self.mc_rec_mask]

        if self.params["mc_gen_cut"]:
            cut_position = self.params["mc_gen_cut_position"]
            self.mc_gen_mask = ~((self.mc_gen > cut_position[0]).squeeze() * (self.mc_gen < cut_position[1]).squeeze())
            self.mc_rec = self.mc_rec[self.mc_gen_mask]
            self.mc_gen = self.mc_gen[self.mc_gen_mask]

        if self.params["data_rec_cut"]:
            cut_position = self.params["data_rec_cut_position"]
            self.data_rec_mask = ~((self.data_rec > cut_position[0]).squeeze() * (self.data_rec < cut_position[1]).squeeze())
            #self.data_rec = self.data_rec[self.data_rec_mask]
            #self.data_gen = self.data_gen[self.data_rec_mask]

        if self.params["data_gen_cut"]:
            cut_position = self.params["data_gen_cut_position"]
            self.data_gen_mask = ~((self.data_gen > cut_position[0]).squeeze() * (self.data_gen < cut_position[1]).squeeze())
            self.data_rec = self.data_rec[self.data_gen_mask]
            self.data_gen = self.data_gen[self.data_gen_mask]
