###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn
import magnify.latent_ode.lib.utils as utils
from magnify.latent_ode.lib.encoder_decoder import *
from magnify.latent_ode.lib.likelihood_eval import *

from torch.distributions.normal import Normal


def create_classifier(z0_dim, n_labels):
    return nn.Sequential(
            nn.Linear(z0_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels),)


def create_regressor(z0_dim, n_labels):
    return nn.Sequential(
            nn.Linear(z0_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels),)


class Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim, device,
                 obsrv_std=0.01, predict_params=False,
                 classif_per_tp=False,
                 use_poisson_proc=False,
                 linear_classifier=False,
                 n_labels=1):
        super(Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        self.device = device

        self.predict_params = predict_params
        self.classif_per_tp = classif_per_tp
        self.use_poisson_proc = use_poisson_proc
        self.linear_classifier = linear_classifier

        z0_dim = latent_dim
        if use_poisson_proc:
            z0_dim += latent_dim

        if predict_params:
            if linear_classifier:
                self.classifier = nn.Sequential(
                    nn.Linear(z0_dim, n_labels))
            else:
                self.classifier = create_classifier(z0_dim, n_labels)
            utils.init_network_weights(self.classifier)


    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = masked_gaussian_log_density(pred_y, truth,
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)

        # Compute the total density
        # Take mean over n_traj_samples
        log_density = torch.mean(log_density_data, 0)

        # shape: [n_traj]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth, mask = mask)
        # shape: [1]
        return torch.mean(log_density_data)


    def compute_all_losses(self, batch_dict,
        n_tp_to_sample = None, n_traj_samples = 1, kl_coef = 1.):

        # Condition on subsampled points
        # Make predictions for all the points
        pred_x, info = self.get_reconstruction(batch_dict["tp_to_predict"],
            batch_dict["observed_data"], batch_dict["observed_tp"],
            mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        # Compute likelihood of all the points
        likelihood = self.get_gaussian_likelihood(batch_dict["data_to_predict"], pred_x,
            mask = batch_dict["mask_predicted_data"])

        mse = self.get_mse(batch_dict["data_to_predict"], pred_x,
            mask = batch_dict["mask_predicted_data"])

        ################################
        # Compute CE loss for binary classification on Physionet
        # Use only last attribute -- mortatility in the hospital
        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)

        if (batch_dict["labels"] is not None) and self.predict_params:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

            if torch.isnan(ce_loss):
                print("label pred")
                print(info["label_predictions"])
                print("labels")
                print( batch_dict["labels"])
                raise Exception("CE loss is Nan!")

        if (batch_dict["labels"] is not None) and self.use_regressor:
            assert (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1)
            mse_param_loss = compute_param_mse_loss(
                info["label_predictions"],
                batch_dict["labels"])

            if torch.isnan(mse_param_loss):
                print("label pred")
                print(info["label_predictions"])
                print("labels")
                print( batch_dict["labels"])
                raise Exception("mse_param_loss is Nan!")

        pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_x,
                info, mask = batch_dict["mask_predicted_data"])
            # Take mean over n_traj
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        loss = - torch.mean(likelihood)

        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood

        if self.predict_params:
            loss = loss +  ce_loss * 100

        # Take mean over the number of samples in a batch
        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl"] = 0.
        results["kl_first_p"] =  0.
        results["std_first_p"] = 0.

        if batch_dict["labels"] is not None and self.predict_params:
            results["label_predictions"] = info["label_predictions"].detach()
        return results


class VAE_Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim,
                 z0_prior, device,
                 obsrv_std = 0.01,
                 predict_params = False,
                 use_classifier=False,
                 use_regressor=False,
                 classif_per_tp = False,
                 use_poisson_proc = False,
                 linear_classifier = False,
                 n_labels = 1):

        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.n_labels = n_labels

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.z0_prior = z0_prior

        self.classif_per_tp = classif_per_tp
        self.use_poisson_proc = use_poisson_proc
        self.use_classifier = use_classifier
        self.use_regressor = use_regressor
        self.linear_classifier = linear_classifier
        self.predict_params = self.use_regressor or self.linear_classifier

        z0_dim = latent_dim
        if use_poisson_proc:
            z0_dim += latent_dim

        if self.use_classifier:
            if linear_classifier:
                self.classifier = nn.Sequential(
                    nn.Linear(z0_dim, n_labels))
            else:
                self.classifier = create_classifier(z0_dim, n_labels)
            utils.init_network_weights(self.classifier)

        if self.use_regressor:
            print(f"Initializing regression network without output dimension {n_labels}...")
            self.regressor = create_regressor(z0_dim, n_labels)
            utils.init_network_weights(self.regressor)


    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(pred_y, truth_repeated,
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
        # shape: [1]
        return torch.mean(log_density_data)


    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
        # Condition on subsampled points
        # Make predictions for all the points
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"],
            batch_dict["observed_data"], batch_dict["observed_tp"],
            mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        #print("get_reconstruction done -- computing likelihood")
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = torch.exp(fp_std)
        fp_distr = Normal(fp_mu, fp_std)

        assert(torch.sum(fp_std < 0) == 0.)

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_traj_samples]
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y,
                info, mask = batch_dict["mask_predicted_data"])
            # Take mean over n_traj
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ################################
        # Compute CE/MSE loss for joint classification/regression
        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)
        if (batch_dict["labels"] is not None) and self.predict_params:
            if self.use_classifier:
                if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                    ce_loss = compute_binary_CE_loss(
                        info["label_predictions"],
                        batch_dict["labels"])
                else:
                    ce_loss = compute_multiclass_CE_loss(
                        info["label_predictions"],
                        batch_dict["labels"],
                        mask = batch_dict["mask_predicted_data"])

            if self.use_regressor:
                #assert (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1)
                mse_param_loss = compute_param_mse_loss(
                    info["label_predictions"],
                    batch_dict["labels"])

                if torch.isnan(mse_param_loss):
                    print("label pred")
                    print(info["label_predictions"])
                    print("labels")
                    print( batch_dict["labels"])
                    raise Exception("mse_param_loss is Nan!")

        # IWAE loss
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood

        if self.use_classifier:
            loss = loss + ce_loss*100

        if self.use_regressor:
            loss = loss + mse_param_loss*100


        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()

        if batch_dict["labels"] is not None and self.predict_params:
            results["label_predictions"] = info["label_predictions"].detach()

        return results



