import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic

from models.cfm import CFM, DiDi
from models.classifier import Classifier
from datasets.gaussian_toy import GaussianToy
from datasets.omnifold import Omnifold
from datasets.combinatorics_toy import CombinatoricsToy
from util.util import get_quantile_bins


class IterativeCFM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        print(f"Using device {self.device}")

        # get the dataset
        self.dataset = eval(self.params["dataset_params"]["type"])(self.params["dataset_params"])
        # get gen and rec dimension from the dataset
        self.dims_gen = self.dataset.mc_gen.shape[-1]
        self.dims_rec = self.dataset.mc_rec.shape[-1]

        # build the cfm and the iteration classifier
        self.cfm = eval(self.params["cfm_params"]["model"])(dims_x=self.dims_gen, dims_c=self.dims_rec, params=self.params["cfm_params"]).to(self.device)
        self.iteration_classifier = Classifier(dims_in=self.dims_gen, params=self.params["classifier_params"]).to(self.device)

        # initialize weights as ones
        self.iteration = 1
        # these will be the weights from the iteration classifier to be applied to mc_gen
        self.iteration_weights = torch.ones((self.dataset.mc_gen.shape[0])).to(self.device)
        # these will be the weights from the background reweighting classifier to be applied to data_rec
        self.background_weights = torch.ones((self.dataset.data_rec.shape[0])).to(self.device)

    def run(self):
        # if True, make a plot of the MC and data distributions
        if self.params.get("plot_data", True):
            print("\n")
            print("--------------------------------------------------------")
            print("         Making dataset plot")
            print("--------------------------------------------------------")
            file_plots = os.path.join(self.run_dir, f"dataset.pdf")
            with PdfPages(file_plots) as pp:
                for dim in range(self.dataset.mc_gen.shape[1]):
                    # this plot shows the obtained unfolding of the data rec
                    # CFM should be on top of Data gen
                    bins = get_quantile_bins(torch.cat([self.dataset.mc_rec[:, dim], self.dataset.data_rec[:, dim]]), n_bins=20)
                    plt.hist(self.dataset.mc_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC gen")
                    plt.hist(self.dataset.mc_rec[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC rec")
                    plt.hist(self.dataset.data_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Data gen")
                    plt.hist(self.dataset.data_rec[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Data rec")
                    if isinstance(self.dataset, CombinatoricsToy):
                        plt.hist(self.dataset.mc_rec_matched[:, dim].cpu(), density=True, histtype="step", bins=bins,
                                 label="MC rec matched")
                    plt.legend()
                    plt.title(f"Dataset dim{dim}")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()

                if isinstance(self.dataset, CombinatoricsToy):
                    for ind in [[0, 1], [0, 2], [1, 2]]:
                        rec = self.dataset.mc_rec[:, ind[0]].cpu() + self.dataset.mc_rec[:, ind[1]].cpu()
                        rec_matched = self.dataset.mc_rec_matched[:, ind[0]].cpu() + self.dataset.mc_rec[:, ind[1]].cpu()
                        gen = self.dataset.mc_gen[:, ind[0]].cpu() + self.dataset.mc_gen[:, ind[1]].cpu()
                        bins = get_quantile_bins(rec, n_bins=20)
                        plt.hist(rec, density=True, histtype="step", bins=bins, label=f"m_{ind[0], ind[1]} rec")
                        plt.hist(rec_matched, density=True, histtype="step", bins=bins, label=f"m_{ind[0], ind[1]} rec matched")
                        plt.hist(gen, density=True, histtype="step", bins=bins, label=f"m_{ind[0], ind[1]} gen")
                        plt.hist(self.dataset.W_gen.squeeze(), density=True, histtype="step", bins=bins, label=f"m_W gen")
                        plt.legend()
                        plt.title(f"Dataset Mass {ind[0], ind[1]}")
                        plt.savefig(pp, format="pdf", bbox_inches="tight")
                        plt.close()

                    rec = (self.dataset.mc_rec[:, 0] + self.dataset.mc_rec[:, 1] + self.dataset.mc_rec[:, 2]).cpu()
                    gen = (self.dataset.mc_gen[:, 0] + self.dataset.mc_gen[:, 1] + self.dataset.mc_gen[:, 2]).cpu()
                    bins = get_quantile_bins(rec, n_bins=20)
                    plt.hist(rec, density=True, histtype="step", bins=bins, label=f"m_012 rec")
                    plt.hist(gen, density=True, histtype="step", bins=bins, label=f"m_012 gen")
                    plt.hist(self.dataset.top_gen.squeeze(), density=True, histtype="step", bins=bins, label=f"m_t gen")
                    plt.legend()
                    plt.title(f"Dataset Mass 012")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()



        # if set to true, do the background suppression stuff
        # this will train the classifier and overwrite the self.background_weights with the predictions for the data_rec weights
        if self.params.get("background_classifier", False):
            print("\n")
            print("--------------------------------------------------------")
            print("         Running background removal")
            print("--------------------------------------------------------")
            self.train_background_classifier()
        self.train()

    # train the background suppression classifier
    def train_background_classifier(self):
        # build the datasets with according weights
        data_true = torch.cat([self.dataset.data_rec, self.dataset.mc_background_rec], dim=0)
        weights_true = torch.cat([torch.ones((self.dataset.data_rec.shape[0])), -1 * torch.ones((self.dataset.mc_background_rec.shape[0]))], dim=0).to(self.device)

        data_false = self.dataset.data_rec
        weights_false = torch.ones((self.dataset.data_rec.shape[0])).to(self.device)

        # build and train the classifier
        print("Starting classifier training")
        self.background_classifier = Classifier(dims_in=self.dims_rec, params=self.params["classifier_params"]).to(self.device)
        self.background_classifier.train(data_true, data_false, weights_true, weights_false)

        # Predict and normalize the weights
        print("Finished classifier training. Starting classifier predictions")
        self.background_weights = self.background_classifier.evaluate(self.dataset.data_rec)
        self.background_weights = self.background_weights * len(self.dataset.data_signal_rec) / self.background_weights.sum()

        # Make a plot to check if the background suppression worked as intended
        print("Making Plots")
        file = os.path.join(self.run_dir, f"background_classifier.pdf")
        with PdfPages(file) as pp:
            for dim in range(self.dataset.mc_gen.shape[1]):
                bins = get_quantile_bins(self.dataset.data_rec[:, dim], n_bins=20)
                plt.hist(self.dataset.data_rec[:, dim].cpu(), histtype="step", bins=bins, label="Data rec")
                plt.hist(self.dataset.data_rec[:, dim].cpu(), histtype="step", bins=bins, label="Data rec rew.",
                         weights=self.background_weights.cpu())
                plt.hist(self.dataset.data_signal_rec[:, dim].cpu(), histtype="step", bins=bins, label="Signal rec")
                plt.hist(self.dataset.data_background_rec[:, dim].cpu(), histtype="step", bins=bins, label="Background rec")
                plt.legend()
                plt.title(f"Background suppression reweighting")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()
        print("Finished background removal")

    # run the main training loop
    def train(self):
        for i in range(self.params["iterations"]):
            print("\n")
            print("--------------------------------------------------------")
            print("         Starting iteration ", self.iteration)
            print("--------------------------------------------------------")
            print("Starting CFM training")
            self.cfm.train(self.dataset.mc_gen, self.dataset.mc_rec, self.iteration_weights)
            print("Finished CFM training. Starting CFM predictions")
            mc_unfolded = self.cfm.evaluate(self.dataset.mc_rec)
            data_unfolded = self.cfm.evaluate(self.dataset.data_rec)
            print("Finished CFM predictions")
            print("--------------------------------------------------------")

            print("Starting classifier training")
            self.iteration_classifier.train(data_unfolded, self.dataset.mc_gen, self.background_weights, None)
            print("Finished classifier training. Starting classifier predictions")
            mc_gen_iteration_weights = self.iteration_classifier.evaluate(self.dataset.mc_gen)
            data_unfolded_iteration_weights = self.iteration_classifier.evaluate(data_unfolded)
            print("Finished classifier predictions")
            print("--------------------------------------------------------")

            print("Making Plots")
            self.make_plots(mc_unfolded, data_unfolded, mc_gen_iteration_weights, data_unfolded_iteration_weights)

            if self.params.get("save_models", True):
                print("Saving trained models")
                os.makedirs(os.path.join(self.run_dir, "model"), exist_ok=True)
                cfm_path = os.path.join(self.run_dir, "model", f"cfm_iteration{self.iteration}.pth")
                classifier_path = os.path.join(self.run_dir, "model", f"classifier_iteration{self.iteration}.pth")
                torch.save(self.cfm.state_dict(), cfm_path)
                torch.save(self.iteration_classifier.state_dict(), classifier_path)

            if self.params.get("save_samples", True):
                print("Saving unfolded samples")
                os.makedirs(os.path.join(self.run_dir, "samples"), exist_ok=True)
                samples_path = os.path.join(self.run_dir, "samples", f"unfolding_iteration{self.iteration}.pt")
                torch.save(data_unfolded, samples_path)

            self.iteration_weights = mc_gen_iteration_weights
            print("Finished iteration", self.iteration)
            self.iteration += 1

    def make_plots(self, mc_unfolded, data_unfolded, weights_mc_gen, weights_data_unfolded):

        file_cfmplots = os.path.join(self.run_dir, f"cfm_iteration{self.iteration}.pdf")
        with PdfPages(file_cfmplots) as pp:
            for dim in range(self.dataset.mc_gen.shape[1]):
                # this plot just shows the CFM training on the MC rec and gen
                # allows to check if the CFM training has properly converged
                # CFM rew. should be on top of MC gen rew.
                bins = get_quantile_bins(self.dataset.mc_rec[:, dim], n_bins=20)
                plt.hist(self.dataset.mc_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC gen rew.",
                            weights=self.iteration_weights.cpu())
                plt.hist(self.dataset.mc_rec[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC rec rew.",
                         weights=self.iteration_weights.cpu())
                plt.hist(mc_unfolded[:, dim].cpu(), density=True, histtype="step", bins=bins, label="CFM rew.",
                            weights=self.iteration_weights.cpu())
                plt.legend()
                plt.title(f"MC Dim {dim} Iteration {self.iteration}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

            if isinstance(self.dataset, CombinatoricsToy):
                if isinstance(self.dataset, CombinatoricsToy):
                    for ind in [[0, 1], [0, 2], [1, 2]]:
                        rec = self.dataset.mc_rec[:, ind[0]].cpu() + self.dataset.mc_rec[:, ind[1]].cpu()
                        gen = self.dataset.mc_gen[:, ind[0]].cpu() + self.dataset.mc_gen[:, ind[1]].cpu()
                        unfold = mc_unfolded[:, ind[0]].cpu() + mc_unfolded[:, ind[1]].cpu()
                        bins = get_quantile_bins(rec, n_bins=20)
                        plt.hist(rec, density=True, histtype="step", bins=bins, label=f"m_{ind[0], ind[1]} rec")
                        plt.hist(gen, density=True, histtype="step", bins=bins, label=f"m_{ind[0], ind[1]} gen")
                        plt.hist(unfold, density=True, histtype="step", bins=bins, label=f"m_{ind[0], ind[1]} unfold")
                        plt.legend()
                        plt.title(f"MC Corr {ind[0], ind[1]} Iteration {self.iteration}")
                        plt.savefig(pp, format="pdf", bbox_inches="tight")
                        plt.close()

                    rec = (self.dataset.mc_rec[:, 0] + self.dataset.mc_rec[:, 1] + self.dataset.mc_rec[:, 2]).cpu()
                    gen = (self.dataset.mc_gen[:, 0] + self.dataset.mc_gen[:, 1] + self.dataset.mc_gen[:, 2]).cpu()
                    unfold = (mc_unfolded[:, 0] + mc_unfolded[:, 1] + mc_unfolded[:, 2]).cpu()
                    bins = get_quantile_bins(rec, n_bins=20)
                    plt.hist(rec, density=True, histtype="step", bins=bins, label=f"m_012 rec")
                    plt.hist(gen, density=True, histtype="step", bins=bins, label=f"m_012 gen")
                    plt.hist(unfold, density=True, histtype="step", bins=bins, label=f"m_012 unfold")
                    plt.legend()
                    plt.title(f"MC Corr 012 Iteration {self.iteration}")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()


            for dim in range(self.dataset.mc_gen.shape[1]):
                # this plot shows the obtained unfolding of the data rec
                # CFM should be on top of Data gen
                bins = get_quantile_bins(torch.cat([self.dataset.mc_rec[:, dim], self.dataset.data_rec[:, dim]]), n_bins=20)
                plt.hist(self.dataset.mc_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC gen")
                plt.hist(self.dataset.mc_rec[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC rec")
                plt.hist(self.dataset.data_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Data gen")
                plt.hist(self.dataset.data_rec[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Data rec",
                            weights=self.background_weights.cpu())
                plt.hist(data_unfolded[:, dim].cpu(), density=True, histtype="step", bins=bins, label="CFM",
                            weights=self.background_weights.cpu())
                if isinstance(self.dataset, GaussianToy):
                    data_unfolded_transfer = self.dataset.apply_detector(data_unfolded[:, dim])
                    plt.hist(data_unfolded_transfer.cpu(), density=True, histtype="step", bins=bins, label="CFM Transfer",
                             weights=self.background_weights.cpu())
                plt.legend()
                plt.title(f"Data Dim {dim} Iteration {self.iteration}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

        file_classifierplots = os.path.join(self.run_dir, f"classifier_iteration{self.iteration}.pdf")
        with PdfPages(file_classifierplots) as pp:
            # this plot just shows the classifier weight distributions
            # allows to check if it looks reasonable
            plt.hist(weights_mc_gen.cpu(), density=True, histtype="step", bins=30,  label="MC gen weights")
            plt.hist(weights_data_unfolded.cpu(), density=True, histtype="step", bins=30,label="Unfolded weights")
            plt.title(f"Weight distributions {self.iteration}")
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.yscale("log")
            plt.close()

            for dim in range(self.dataset.mc_gen.shape[1]):
                # this plot shows of the classifier has converged properly
                # MC gen rew. should be on top of Unfolded
                bins = get_quantile_bins(self.dataset.mc_gen[:, dim], n_bins=20)
                plt.hist(self.dataset.mc_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC gen")
                plt.hist(self.dataset.mc_gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="MC gen rew.",
                         weights=weights_mc_gen.cpu())
                plt.hist(data_unfolded[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Unfolded",
                         weights=self.background_weights.cpu())
                plt.legend()
                plt.title(f"Rew. MC Gen Dim {dim} Iteration {self.iteration}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()


