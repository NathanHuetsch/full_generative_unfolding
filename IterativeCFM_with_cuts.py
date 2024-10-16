import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic

from models.cfm import CFM
from models.classifier import Classifier
from datasets.gaussian_toy import GaussianToy
from util.util import get_quantile_bins


class IterativeCFM_with_Cuts(nn.Module):
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
        self.cfm = CFM(dims_x=self.dims_gen, dims_c=self.dims_rec, params=self.params["cfm_params"])
        self.iteration_classifier = Classifier(dims_in=self.dims_gen, params=self.params["classifier_params"])

        # initialize weights as ones
        self.iteration = 1
        self.iteration_weights = torch.ones((self.dataset.mc_gen.shape[0]))
        self.background_weights = torch.ones((self.dataset.data_rec.shape[0]))

        self.mc_gen_efficiency_weights = torch.zeros((self.dataset.mc_gen.shape[0]))
        self.mc_unfolded_efficiency_weights = torch.ones((self.dataset.mc_rec.shape[0]))
        self.data_unfolded_efficiency_weights = torch.ones((self.dataset.data_rec.shape[0]))

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
                    bins = get_quantile_bins(torch.cat([self.dataset.mc_rec[:, dim], self.dataset.data_rec[:, dim], self.dataset.mc_gen[:, dim], self.dataset.data_gen[:, dim]]),
                                             n_bins=30)
                    plt.hist(self.dataset.mc_gen[:, dim].cpu(), density=False, histtype="step", bins=bins,
                             label="MC gen")
                    plt.hist(self.dataset.mc_rec[:, dim].cpu(), density=False, histtype="step", bins=bins,
                             label="MC rec")
                    plt.hist(self.dataset.data_gen[:, dim].cpu(), density=False, histtype="step", bins=bins,
                             label="Data gen")
                    plt.hist(self.dataset.data_rec[:, dim].cpu(), density=False, histtype="step", bins=bins,
                             label="Data rec")
                    plt.hist(self.dataset.mc_gen[self.dataset.mc_rec_mask.squeeze(), dim].cpu(), density=False, histtype="step", bins=bins,
                             label="MC gen Cut")
                    plt.hist(self.dataset.mc_rec[self.dataset.mc_rec_mask.squeeze(), dim].cpu(), density=False, histtype="step", bins=bins,
                             label="MC rec Cut")

                    plt.legend()
                    plt.title(f"Dataset dim{dim}")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()

        if self.params.get("background_classifier", False):
            print("\n")
            print("--------------------------------------------------------")
            print("         Running background removal")
            print("--------------------------------------------------------")
            self.train_background_classifier()
        if self.params.get("efficiency_classifier", False):
            print("\n")
            print("--------------------------------------------------------")
            print("         Training efficiency classifier")
            print("--------------------------------------------------------")
            self.train_efficiency_classifier()
        self.train()

    # train the background suppression classifier
    def train_background_classifier(self):
        data_true = torch.cat([self.dataset.data_rec, self.dataset.mc_background_rec], dim=0)
        weights_true = torch.cat([torch.ones((self.dataset.data_rec.shape[0])), -1 * torch.ones((self.dataset.mc_background_rec.shape[0]))], dim=0)

        data_false = self.dataset.data_rec
        weights_false = torch.ones((self.dataset.data_rec.shape[0]))

        self.background_classifier = Classifier(dims_in=self.dims_rec, params=self.params["classifier_params"])
        print("Starting classifier training")
        self.background_classifier.train(data_true, data_false, weights_true, weights_false)
        print("Finished classifier training. Starting classifier predictions")
        self.background_weights = self.background_classifier.evaluate(self.dataset.data_rec)
        self.background_weights = self.background_weights * len(self.dataset.data_signal_rec) / self.background_weights.sum()
        print("Making Plots")
        file = os.path.join(self.run_dir, f"background_classifier.pdf")
        with PdfPages(file) as pp:
            for dim in range(self.dataset.mc_gen.shape[1]):
                bins = get_quantile_bins(self.dataset.data_rec[:, dim], n_bins=30)
                plt.hist(self.dataset.data_rec[:, dim], histtype="step", bins=bins, label="Data rec")
                plt.hist(self.dataset.data_rec[:, dim], histtype="step", bins=bins, label="Data rec rew.",
                         weights=self.background_weights)
                plt.hist(self.dataset.data_signal_rec[:, dim], histtype="step", bins=bins, label="Signal rec")
                plt.hist(self.dataset.data_background_rec[:, dim], histtype="step", bins=bins, label="Background rec")
                plt.legend()
                plt.title(f"Background suppression reweighting")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()
        print("Finished background removal")

    # train the gen level efficiency classifier
    def train_efficiency_classifier(self):
        data_true = self.dataset.mc_gen[self.dataset.mc_rec_mask.squeeze()]
        weights_true = torch.ones((data_true.shape[0]))

        data_false = self.dataset.mc_gen[~self.dataset.mc_rec_mask.squeeze()]
        weights_false = torch.ones((data_false.shape[0]))

        self.efficiency_classifier = Classifier(dims_in=self.dims_gen, params=self.params["classifier_params"])
        print("Starting classifier training")
        self.efficiency_classifier.train(data_true, data_false, weights_true, weights_false, False)
        print("Finished classifier training. Starting classifier predictions")
        self.mc_gen_efficiency_weights = self.efficiency_classifier.evaluate(self.dataset.mc_gen, return_weights=False)
        self.data_gen_efficiency_weights = self.efficiency_classifier.evaluate(self.dataset.data_gen, return_weights=False)
        print("Making Plots")
        file = os.path.join(self.run_dir, f"efficiency_classifier.pdf")
        bins = torch.linspace(-15, 25, 50)
        with PdfPages(file) as pp:
            mc_events = self.dataset.mc_gen.numpy()
            mc_preds = self.mc_gen_efficiency_weights.numpy()
            mc_labels = self.dataset.mc_rec_mask.squeeze().numpy()
            data_events = self.dataset.data_gen.numpy()
            data_preds = self.data_gen_efficiency_weights.numpy()
            data_labels = self.dataset.data_rec_mask.squeeze().numpy()
            for dim in range(self.dataset.mc_gen.shape[1]):
                binned_efficiency_predicted, _, _ = binned_statistic(mc_events[:, dim], mc_preds, bins=bins)
                binned_efficiency_true, _, _ = binned_statistic(mc_events[:, dim], mc_labels, bins=bins)
                plt.step(bins[1:], binned_efficiency_predicted, label="Predicted MC")
                plt.step(bins[1:], binned_efficiency_true, label="True MC")

                binned_efficiency_predicted, _, _ = binned_statistic(data_events[:, dim], data_preds, bins=bins)
                binned_efficiency_true, _, _ = binned_statistic(data_events[:, dim], data_labels, bins=bins)
                plt.step(bins[1:], binned_efficiency_predicted, label="Predicted Data")
                plt.step(bins[1:], binned_efficiency_true, label="True Data")
                plt.legend()
                plt.title(f"Efficiency Dim {dim}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

        print("Finished efficiency training")

    # train for one iteration
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

            if self.params.get("efficiency_classifier", False):
                print("Starting efficiency predictions")
                self.mc_unfolded_efficiency_weights = self.efficiency_classifier.evaluate(mc_unfolded, return_weights=False)
                self.data_unfolded_efficiency_weights = self.efficiency_classifier.evaluate(data_unfolded, return_weights=False)
                print("--------------------------------------------------------")

            mc_cut_unfolded = mc_unfolded[self.dataset.mc_rec_mask.squeeze()]
            data_cut_unfolded = data_unfolded[self.dataset.data_rec_mask.squeeze()]

            mc_cut_unfolded_weights = (self.iteration_weights/self.mc_unfolded_efficiency_weights)[self.dataset.mc_rec_mask.squeeze()]
            data_cut_unfolded_weights = (self.background_weights/self.data_unfolded_efficiency_weights)[self.dataset.data_rec_mask.squeeze()]

            mc_unfolded_full = torch.cat([mc_cut_unfolded, self.dataset.mc_gen])
            data_unfolded_full = torch.cat([data_cut_unfolded, self.dataset.mc_gen])

            mc_unfolded_full_weights = torch.cat([self.iteration_weights[self.dataset.mc_rec_mask.squeeze()], 1. - self.mc_gen_efficiency_weights])
            data_unfolded_full_weights = torch.cat([self.background_weights[self.dataset.data_rec_mask.squeeze()], 1. - self.mc_gen_efficiency_weights])

            file_cfmplots = os.path.join(self.run_dir, f"cfm_iteration{self.iteration}.pdf")
            with PdfPages(file_cfmplots) as pp:
                for dim in range(self.dataset.mc_gen.shape[1]):
                    bins = get_quantile_bins(torch.cat([self.dataset.mc_rec[:, dim], self.dataset.mc_gen[:, dim]]), n_bins=30)
                    plt.hist(self.dataset.mc_gen[:, dim], density=False, histtype="step", bins=bins, label="MC gen rew.",
                             weights=self.iteration_weights)
                    plt.hist(self.dataset.mc_rec[:, dim], density=False, histtype="step", bins=bins, label="MC rec rew.",
                             weights=self.iteration_weights)
                    plt.hist(mc_unfolded[:, dim], density=False, histtype="step", bins=bins, label="CFM rew.",
                             weights=self.iteration_weights)

                    plt.hist(self.dataset.mc_gen[self.dataset.mc_rec_mask.squeeze(), dim], density=False, histtype="step", bins=bins, label="MC gen rew. Cut",
                             weights=self.iteration_weights[self.dataset.mc_rec_mask.squeeze()])
                    plt.hist(self.dataset.mc_rec[self.dataset.mc_rec_mask.squeeze(), dim], density=False, histtype="step", bins=bins, label="MC rec rew. Cut",
                             weights=self.iteration_weights[self.dataset.mc_rec_mask.squeeze()])
                    plt.hist(mc_cut_unfolded[:, dim], density=False, histtype="step", bins=bins, label="CFM rew. Cut",
                             weights=self.iteration_weights[self.dataset.mc_rec_mask.squeeze()])

                    plt.hist(mc_cut_unfolded[:, dim], density=False, histtype="step", bins=bins, label="CFM rew. Cut upweight",
                             weights=mc_cut_unfolded_weights)

                    plt.hist(mc_unfolded_full[:, dim], density=False, histtype="step", bins=bins, label="CFM rew. Full",
                             weights=mc_unfolded_full_weights)


                    plt.legend()
                    plt.title(f"MC Dim {dim} Iteration {self.iteration}")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()

                    bins = get_quantile_bins(torch.cat([self.dataset.data_rec[:, dim], self.dataset.data_gen[:, dim]]), n_bins=30)
                    plt.hist(self.dataset.data_gen[:, dim], density=False, histtype="step", bins=bins,
                             label="Data gen")
                    plt.hist(self.dataset.data_rec[:, dim], density=False, histtype="step", bins=bins,
                             label="Data rec",
                             weights=self.background_weights)
                    plt.hist(data_unfolded[:, dim], density=False, histtype="step", bins=bins, label="CFM",
                             weights=self.background_weights)

                    plt.hist(self.dataset.data_gen[self.dataset.data_rec_mask.squeeze(), dim], density=False,
                             histtype="step", bins=bins, label="Data gen Cut")
                    plt.hist(self.dataset.data_rec[self.dataset.data_rec_mask.squeeze(), dim], density=False,
                             histtype="step", bins=bins, label="Data rec Cut",
                             weights=self.background_weights[self.dataset.data_rec_mask.squeeze()])
                    plt.hist(data_cut_unfolded[:, dim], density=False, histtype="step", bins=bins, label="CFM Cut",
                             weights=self.background_weights[self.dataset.data_rec_mask.squeeze()])

                    plt.hist(data_cut_unfolded[:, dim], density=False, histtype="step", bins=bins,
                             label="CFM Cut upweight",
                             weights=data_cut_unfolded_weights)

                    plt.hist(data_unfolded_full[:, dim], density=False, histtype="step", bins=bins, label="CFM Cut Full",
                             weights=data_unfolded_full_weights)

                    plt.legend()
                    plt.title(f"MC Dim {dim} Iteration {self.iteration}")
                    plt.savefig(pp, format="pdf", bbox_inches="tight")
                    plt.close()

            print("Starting classifier training")
            self.iteration_classifier.train(data_cut_unfolded, self.dataset.mc_gen, None, None)#data_cut_unfolded_weights, None)
            print("Finished classifier training. Starting classifier predictions")
            mc_gen_iteration_weights = self.iteration_classifier.evaluate(self.dataset.mc_gen)
            data_unfolded_full_iteration_weights = self.iteration_classifier.evaluate(data_unfolded_full)
            print("Finished classifier predictions")
            print("--------------------------------------------------------")

            print("Making Plots")
            #self.make_plots(mc_unfolded, data_unfolded, mc_gen_iteration_weights, data_unfolded_weights, mc_gen_iteration_weights, data_unfolded_iteration_weights)
            #self.make_plots(mc_unfolded_cut, data_unfolded_cut, mc_gen_iteration_weights[self.dataset.mc_rec_mask.squeeze()], 1,
            #                mc_gen_iteration_weights[self.dataset.mc_rec_mask.squeeze()], 2)

            if self.params.get("save_models", False):
                print("Saving trained models")
                os.makedirs(os.path.join(self.run_dir, "model"), exist_ok=True)
                cfm_path = os.path.join(self.run_dir, "model", f"cfm_iteration{self.iteration}.pth")
                classifier_path = os.path.join(self.run_dir, "model", f"classifier_iteration{self.iteration}.pth")
                torch.save(self.cfm.state_dict(), cfm_path)
                torch.save(self.iteration_classifier.state_dict(), classifier_path)

            if self.params.get("save_samples", False):
                print("Saving unfolded samples")
                os.makedirs(os.path.join(self.run_dir, "samples"), exist_ok=True)
                samples_path = os.path.join(self.run_dir, "samples", f"unfolding_iteration{self.iteration}.pt")
                torch.save(data_unfolded, samples_path)

            self.iteration_weights = mc_gen_iteration_weights
            print("Finished iteration", self.iteration)
            self.iteration += 1

    def make_plots(self, mc_unfolded, data_unfolded, mc_unfolded_weights, data_unfolded_weights, mc_gen_iteration_weights, data_unfolded_iteration_weights):

        file_cfmplots = os.path.join(self.run_dir, f"cfm_iteration{self.iteration}.pdf")
        with PdfPages(file_cfmplots) as pp:
            for dim in range(self.dataset.mc_gen.shape[1]):
                bins = get_quantile_bins(self.dataset.mc_rec[:, dim], n_bins=30)
                plt.hist(self.dataset.mc_gen[:, dim], density=True, histtype="step", bins=bins, label="MC gen rew.",
                            weights=self.iteration_weights)
                plt.hist(self.dataset.mc_rec[:, dim], density=True, histtype="step", bins=bins, label="MC rec rew.",
                         weights=self.iteration_weights)
                plt.hist(mc_unfolded[:, dim], density=True, histtype="step", bins=bins, label="CFM rew.")#,
                            #weights=self.iteration_weights)
                plt.legend()
                plt.title(f"MC Dim {dim} Iteration {self.iteration}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

            for dim in range(self.dataset.mc_gen.shape[1]):
                bins = get_quantile_bins(torch.cat([selx, self.dataset.data_rec[:, dim]]),
                                         n_bins=30)
                plt.hist(self.dataset.mc_gen[:, dim], density=True, histtype="step", bins=bins, label="MC gen")
                plt.hist(self.dataset.mc_rec[:, dim], density=True, histtype="step", bins=bins, label="MC rec")
                plt.hist(self.dataset.data_gen[:, dim], density=True, histtype="step", bins=bins, label="Data gen")
                plt.hist(self.dataset.data_rec[:, dim], density=True, histtype="step", bins=bins, label="Data rec",
                            weights=self.background_weights)
                plt.hist(data_unfolded[:, dim], density=True, histtype="step", bins=bins, label="CFM",
                            weights=self.background_weights)
                plt.legend()
                plt.title(f"Data Dim {dim} Iteration {self.iteration}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()

        file_classifierplots = os.path.join(self.run_dir, f"classifier_iteration{self.iteration}.pdf")
        with PdfPages(file_classifierplots) as pp:
            plt.hist(mc_gen_iteration_weights, density=True, histtype="step", bins=30,  label="MC gen weights")
            plt.hist(data_unfolded_iteration_weights, density=True, histtype="step", bins=30,label="Unfolded weights")
            plt.title(f"Weight distributions {self.iteration}")
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.yscale("log")
            plt.close()

            for dim in range(self.dataset.mc_gen.shape[1]):
                bins = get_quantile_bins(self.dataset.mc_gen[:, dim], n_bins=30)
                plt.hist(self.dataset.mc_gen[:, dim], density=True, histtype="step", bins=bins, label="MC gen")
                plt.hist(self.dataset.mc_gen[:, dim], density=True, histtype="step", bins=bins, label="MC gen rew.",
                         weights=mc_gen_iteration_weights)
                plt.hist(data_unfolded[:, dim], density=True, histtype="step", bins=bins, label="Unfolded",
                         weights=mc_unfolded_weights)
                plt.legend()
                plt.title(f"Rew. MC Gen Dim {dim} Iteration {self.iteration}")
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()


