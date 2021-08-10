###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
# Modified by Ji Won Park (@jiwoncpark) for joint reconstruction for
# multiple filters and parameter regression
###########################

from functools import partial
import torch
import lib.utils as utils
from periodic_utils import Periodic_1d, Periodic1dDataset
import drw_utils
from torch.distributions import uniform
from torch.utils.data import DataLoader


def parse_datasets(args, device):

    def basic_collate_fn(batch, time_steps, args = args, device=device, data_type="train"):
        tseries, labels = map(list, zip(*batch))
        tseries = torch.stack(tseries, dim=0)
        labels = torch.stack(labels, dim=0)
        tseries = tseries.to(device)  # [n_samples, n_times, input_dim]
        labels = labels.to(device)  # [n_samples, n_labels]
        # batch = torch.stack(batch)  # [B, n_times, 2, 1]
        data_dict = {
            "data": tseries,
            "time_steps": time_steps}
        # physionet did this before calling split_and_subsample_batch
        data_dict["labels"] = labels
        data_dict = utils.split_and_subsample_batch(data_dict, args,
                                                    data_type=data_type)
        return data_dict

    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap
    max_t_extrap = args.max_t / args.timepoints * n_total_tp

    ##################################################################

    if dataset_name == 'drw':
        train_seed = 123
        val_seed = 456
        train_dataset, test_dataset = drw_utils.get_drw_datasets(train_seed,
                                                                 val_seed)
        # record_id, tt, y_vals, labels, mask = train_dataset[0]
        input_dim = train_dataset[0]['y'].shape[-1]  # [n_filters]
        n_labels = len(train_dataset.param_names)

        batch_size = min(min(len(train_dataset), args.batch_size), args.n)
        print("batch size", batch_size)
        # record_id, tt, vals, mask, labels = train_data[0]

        # n_samples = len(total_dataset)
        data_min, data_max = drw_utils.get_data_min_max(train_dataset,
                                                        device)
        print("Data min: ", data_min)
        print("Data max: ", data_max)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=partial(drw_utils.variable_time_collate_fn,
                                                         args=args,
                                                         device=device,
                                                         data_type="train",
                                                         data_min=data_min,
                                                         data_max=data_max))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=len(test_dataset),
                                     shuffle=False,
                                     collate_fn=partial(drw_utils.variable_time_collate_fn,
                                                        args=args,
                                                        device=device,
                                                        data_type="test",
                                                        data_min=data_min,
                                                        data_max=data_max))

        attr_names = train_dataset.param_names
        data_objects = {"dataset_obj": train_dataset,
                        "train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "attr": attr_names,  # optional
                        "classif_per_tp": False,  # optional
                        "n_labels": n_labels}  # optional
        return data_objects

    ########### 1d datasets ###########

    # Sampling args.timepoints time points in the interval [0, args.max_t]
    # Sample points for both training sequence and explapolation (test)
    distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
    time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]

    dataset_obj = None
    ##################################################################
    # Sample a periodic function
    if dataset_name == "periodic":
        dataset_obj = Periodic_1d(
            init_freq = None, init_amplitude = 1.,
            final_amplitude = 1., final_freq = None,
            z0 = 1.)

    ##################################################################

    if dataset_obj is None:
        raise Exception("Unknown dataset: {}".format(dataset_name))

    print("n_samples", args.n)
    dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples=args.n,
                                      noise_weight=args.noise_weight)

    # Process small datasets
    time_steps_extrap = time_steps_extrap.to(device)

    train_y, test_y = utils.split_train_test(dataset, train_frac=0.8)
    train_data = Periodic1dDataset(train_y)
    test_data = Periodic1dDataset(test_y)

    # first example (0), first in tuple for tseries (0), 2nd dim of each tseries
    input_dim = train_y[0].size(-1)  # which-dimensional time series?

    batch_size = min(args.batch_size, args.n)
    print("batch size", batch_size)
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=lambda b: basic_collate_fn(b, time_steps_extrap, data_type="train"))
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.n,
                                 shuffle=False,
                                 collate_fn=lambda b: basic_collate_fn(b, time_steps_extrap, data_type = "test"))
    print("number of train batches", len(train_dataloader))
    print("number of test batches", len(test_dataloader))
    data_objects = {"train_dataloader": utils.inf_generator(train_dataloader),
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "n_labels": 1,
                    "classif_per_tp": False, }
    return data_objects


