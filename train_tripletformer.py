# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from random import SystemRandom
import models
import utils
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num-ref-points', type=int, default=32)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=4)
parser.add_argument('--dec-num-heads', type=int, default=4)
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--net', type=str, default='triple')
parser.add_argument('--sample-tp', type=float, default=0.5)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--mse-weight', type=float, default=0.0)
parser.add_argument('--imab-dim', type=int, default=128)
parser.add_argument('--cab-dim', type=int, default=128)
parser.add_argument('--decoder-dim', type=int, default=128)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--sample-type', type=str, default='random')

args = parser.parse_args()
print(' '.join(sys.argv))


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 10000000)
    print(args, experiment_id)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
        
    data_obj = utils.get_dataset(args.batch_size, args.dataset)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    union_tp = None
    early_stop = 0
    net = TRIPLETFORMER(input_dim=dim,
            enc_num_heads=args.enc_num_heads,
            dec_num_heads=args.dec_num_heads,
            num_ref_points=args.num_ref_points,
            mse_weight=args.mse_weight,
            norm=args.norm,
            imab_dim = args.imab_dim,
            cab_dim = args.cab_dim,
            decoder_dim = args.decoder_dim,
            n_layers=args.nlayers,
            device=device
            ).to(device)

    params = list(net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=0.00001, verbose=True)
    print('parameters:', utils.count_parameters(net))
    best_val_loss = 10000
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, mse, mae = 0, 0, 0, 0

        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)
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
                ), -1)
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 1)
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            mse += loss_info.mse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
        print(
            'Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, '
            'mse: {:.6f}, mae: {:.6f}'.format(
                itr,
                train_loss / train_n,
                -avg_loglik / train_n,
                mse / train_n,
                mae / train_n
            )
        )
        if itr % 1 == 0:
            val_loss = utils.test_result(net,dim,val_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
                torch.save({    'args': args,
                                'epoch': itr,
                                'state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss / train_n,
                            }, 'saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
                early_stop = 0
            else:
                early_stop += 1
            if early_stop == 30:
                print("Early stopping because of no improvement in val. metric for 30 epochs")
                chp = torch.load('saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
                net.load_state_dict(chp['state_dict'])
                test_loss = utils.test_result(net,dim,test_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=1)
                print('best_val_loss: ', best_val_loss.cpu().detach().numpy(), ' test_loss: ', test_loss.cpu().detach().numpy())
                
                break
            scheduler.step(val_loss)
    if itr == args.niters:
        print("Trained completed")
        chp = torch.load('saved_models/'+args.dataset + '_' + str(experiment_id) + '.h5')
        net.load_state_dict(chp['state_dict'])
        test_loss = utils.test_result(net,dim,test_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=1)
        print('best_val_loss: ', best_val_loss.cpu().detach().numpy(), ' test_loss: ', test_loss.cpu().detach().numpy())