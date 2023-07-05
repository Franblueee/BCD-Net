import argparse
import torch

def custom_list(string):
    return [int(x) for x in string.split(',')]

def set_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help="Mode to run the code (train/test/train_test)")
    parser.add_argument('--use_wandb', action='store_true', help="Use wandb or not")
    parser.add_argument('--wandb_project', default='DVBCD', type=str, help="Wandb project name")
    parser.add_argument('--load_at_init', action='store_true', help="Load files at init")

    parser.add_argument(
                        '--camelyon_data_path', default='', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument(
                        '--wssb_data_path', default='', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument('--weights_dir', default='', type=str, metavar='PATH', help="Path to save the model weights")

    parser.add_argument('--num_workers', default=16, type=int, help="Number of workers to load data")

    # model settings
    parser.add_argument('--cnet_name', type=str, default='unet_64_6', help='CNet model name')
    parser.add_argument('--mnet_name', type=str, default='mobilenetv3s_50', help='MNet model name')

    # trainning settings
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size of each data sample")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--pretrain_epochs', type=int, default=1, help="Number of pretraining epochs")
    parser.add_argument('--n_samples_train', type=int, default=60000, help="Number of samples to use for training")
    parser.add_argument('--n_samples_val', type=int, default=8000, help="Number of samples to use for validation")
    parser.add_argument('--train_centers', type=custom_list, default='0,2,4', help="Centers to use for training")
    parser.add_argument('--val_centers', type=custom_list, default='1,3', help="Centers to use for validation")

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="Decaying rate for the learning rate")
    parser.add_argument('--early_stop_patience', type=int, default=10, help="Early Stopping patience")
    parser.add_argument('--clip_grad', type=float, default=1e5, help="Value to clip the gradients")

    # hyper-parameters
    parser.add_argument('--sigma_rui_sq', default=0.05, type=float, help="Prior hematoxylin/eosin variance of M")
    parser.add_argument('--theta_val', default=0.5, type=float, help="theta parameter to balance the loss function")
    parser.add_argument('--pretrain_theta_val', default=0.99, type=float, help="theta parameter to balance the loss function during pretraining")


    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print('using device:', args.device)

    return args