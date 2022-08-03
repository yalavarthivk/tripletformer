# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.optim as optim

from random import SystemRandom
import models
import utils
import pdb
import sys
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl-annealing', action='store_true')
parser.add_argument('--kl-zero', action='store_true')
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--const-var', action='store_true')
parser.add_argument('--var-per-dim', action='store_true')
parser.add_argument('--std', type=float, default=0.1)
parser.add_argument('--sample-tp', type=float, default=0.5)
parser.add_argument('--bound-variance', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--recon-loss', action='store_true')
parser.add_argument('--normalize-input', type=str, default='znorm')
parser.add_argument('--sample-type', type=str, default='random')
parser.add_argument('--net', type=str, default='GP')

args = parser.parse_args()

asd = sys.argv
print(' '.join(asd))

if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 10000000)
    print(args, experiment_id)
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'toy':
        data_obj = utils.get_synthetic_data(args)
    else:
        data_obj = utils.get_dataset(args.batch_size, args.dataset)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    union_tp = utils.union_time(train_loader)

    net = models.load_network(args, dim, union_tp, device)
    # pdb.set_trace()
    params = list(net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=0.00001, verbose=True)
    print('parameters:', utils.count_parameters(net))

    best_val = 1000000
    early_stop = 0
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, avg_kl, mse, mae = 0, 0, 0, 0
        if args.kl_annealing:
            wait_until_kl_inc = 10000
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.999999 ** (itr - wait_until_kl_inc))
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)
            if args.dataset == 'toy':
                subsampled_mask = torch.zeros_like(
                    train_batch[:, :, dim:2 * dim]).to(device)
                seqlen = train_batch.size(1)
                for i in range(batch_len):
                    length = np.random.randint(low=3, high=10)
                    obs_points = np.sort(
                        np.random.choice(np.arange(seqlen), size=length, replace=False)
                    )
                    subsampled_mask[i, obs_points, :] = 1
            else:
                if args.sample_type == 'random':
                    subsampled_mask = utils.subsample_timepoints(
                        train_batch[:, :, dim:2 * dim].clone(),
                        args.sample_tp,
                        shuffle=args.shuffle,
                    )
                elif args.sample_type == 'bursts':
                    subsampled_mask = utils.subsample_bursts(
                        train_batch[:, :, dim:2 * dim].clone(),
                        args.sample_tp,
                        shuffle=args.shuffle,
                    )
            if args.recon_loss or args.sample_tp == 1.0:
                recon_mask = train_batch[:, :, dim:2 * dim]
            else:
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
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward()
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_kl += 0
            mse += loss_info.mean_mse * batch_len
            mae += loss_info.mean_mae * batch_len
            train_n += batch_len
        print(
            'Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg kl: {:.4f}, '
            'mse: {:.6f}, mae: {:.6f}'.format(
                itr,
                train_loss / train_n,
                -avg_loglik / train_n,
                avg_kl / train_n,
                mse / train_n,
                mae / train_n
            )
        )
        if itr % 1 == 0:
            val_loss = utils.test_result(net,dim,val_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=1,device=device)

            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({    'args': args,
                                'epoch': itr,
                                'state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss / train_n,
                            }, 'saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
                # torch.save(net.state_dict(), 'saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
                # test_loss = utils.test_result_hetvae(net,dim,test_loader,0.5,shuffle=False,k_iwae=50)
                early_stop = 0
            else:
                early_stop += 1
            if early_stop == 30:
                print("Early stopping because of no improvement in val. metric for 30 epochs")
                chp = torch.load('saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
                net.load_state_dict(chp['state_dict'])
                # net.load_state_dict(torch.load('saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5'))
                test_loss = utils.test_result(net,dim,test_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=50,device=device)

                print('best_val_loss: ', best_val.cpu().detach().numpy(), ', test_loss: ', test_loss.cpu().detach().numpy())
                
                break


    if itr == args.niters:
        chp = torch.load('saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
        net.load_state_dict(chp['state_dict'])
        test_loss = utils.test_result(net,dim,test_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=50,device=device)

        print('best_val_loss: ', best_val.cpu().detach().numpy(), ', test_loss: ', test_loss.cpu().detach().numpy())
