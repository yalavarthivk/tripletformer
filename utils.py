# pylint: disable=E1101
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import model_selection
import pdb
import torch.nn.functional as F



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    # pdb.set_trace()
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask



def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()


def evaluate_model(
    net,
    dim,
    train_loader,
    sample_tp=0.5,
    shuffle=False,
    k_iwae=1,
    device='cuda',
):
    # torch.manual_seed(seed=0)
    # np.random.seed(seed=0)
    train_n = 0
    avg_loglik, mse, mae = 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            subsampled_mask = subsample_timepoints(
                train_batch[:, :, dim:2 * dim].clone(),
                sample_tp,
                shuffle=shuffle,
            )
            recon_mask = train_batch[:, :, dim:2 * dim] - subsampled_mask
            context_y = torch.cat((
                train_batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1],
                context_y,
                train_batch[:, :, -1],
                torch.cat((
                    train_batch[:, :, :dim] * recon_mask, recon_mask
                ), -1),
                num_samples=k_iwae,
            )
            num_context_points = recon_mask.sum().item()
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    print(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n
        )
    )


def get_dataset(batch_size, dataset, test_batch_size=5, filter_anomalies=True):
    if dataset == 'physionet':
        x = np.load("/home/yalavarthi/interpol/hetvae/data/physionet.npz")
    elif dataset == 'mimiciii':
        x = np.load("/home/yalavarthi/interpol/hetvae/data/mimiciii.npz")
    elif dataset == 'FaceDetection':
        x = np.load("/home/yalavarthi/interpol/hetvae/data/FaceDetection.npz")
    elif dataset == 'PenDigits':
        x = np.load("/home/yalavarthi/interpol/hetvae/data/PenDigits.npz")
    elif dataset == 'physionet2019':
        x = np.load("/home/yalavarthi/interpol/hetvae/data/physionet2019.npz")
    elif dataset == 'PhonemeSpectra':
        x = np.load("/home/yalavarthi/interpol/hetvae/data/PhonemeSpectra.npz")
    else:
        print("No dataset found")
    train_data, val_data, test_data = x['train'], x['val'], x['test']
    input_dim = (train_data[-1] - 1)//2

    print(train_data.shape, val_data.shape, test_data.shape)

    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    test_data = torch.from_numpy(test_data).float()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    
    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim
    }
    return data_objects



def subsample_timepoints(mask, percentage_tp_to_sample=None, shuffle=False):
    # Subsample percentage of points from each time series
    if not shuffle:
        seed = 0
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
    for i in range(mask.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = max(1, int(n_tp_current * percentage_tp_to_sample))
        n_to_sample = min((n_tp_current - 1), n_to_sample)
        subsampled_idx = sorted(
            np.random.choice(non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.
    return mask

def subsample_bursts(mask, percentage_tp_to_sample=None, shuffle=False):
    # Subsample percentage of points from each time series
    if not shuffle:
        seed = 0
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
    asd = mask.cpu()
    for i in range(asd.shape[0]):
        # pdb.set_trace()
        total_times = asd[i].sum(-1).to(torch.bool).sum()
        #total_times = current_mask.sum().cpu()
        n_tp_to_sample = max(1, total_times*(1-percentage_tp_to_sample))
        n_tp_to_sample = min((total_times - 1), n_tp_to_sample)
        start_times = total_times - n_tp_to_sample
        start_tp = np.random.randint(start_times+1)
        missing_tp = np.arange(start_tp, start_tp+n_tp_to_sample)
        if mask is not None:
            mask[i, missing_tp] = 0
    return mask


def test_result(
    net,
    dim,
    train_loader,
    sample_type='random',
    sample_tp=0.5,
    shuffle=False,
    k_iwae=1,
    device='cuda'):
    # torch.manual_seed(seed=0)
    # np.random.seed(seed=0)
    train_n = 0
    avg_loglik, mse, mae, crps = 0, 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            if sample_type == 'random':
                subsampled_mask = subsample_timepoints(
                    train_batch[:, :, dim:2 * dim].clone(),
                    sample_tp,
                    shuffle=shuffle,
                )
            elif sample_type == 'bursts':
                subsampled_mask = subsample_bursts(
                    train_batch[:, :, dim:2 * dim].clone(),
                    sample_tp,
                    shuffle=shuffle,
                )
            recon_mask = train_batch[:, :, dim:2 * dim] - subsampled_mask
            context_y = torch.cat((
                train_batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1],
                context_y,
                train_batch[:, :, -1],
                torch.cat((train_batch[:, :, :dim] * recon_mask, recon_mask), -1),
                num_samples=k_iwae
            )
            num_context_points = recon_mask.sum().item()
            crps += loss_info.crps * num_context_points
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.loglik * num_context_points
            # avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    print(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}, mean_crps: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n,
            crps/train_n
        )
    )
    return -avg_loglik/train_n