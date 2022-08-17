from tripletformer import TRIPLETFORMER
import pdb
def load_network(args, dim, device="cuda"):
    if args.net == 'triple':
        net = TRIPLETFORMER(
            input_dim=dim,
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

    elif args.net == 'GP':
        net = GP(
            input_dim=dim, device=device).to(device)
    else:
        raise ValueError("Network not available")
    return net
