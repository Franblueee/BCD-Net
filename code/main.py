import os
import torch

import numpy as np
import random

import wandb

from options import set_opts

from datasets import CamelyonDataset, WSSBDatasetTest
from models import BCDnet

from engine import Trainer, evaluate, evaluate_GT

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def build_model(args):
    return BCDnet(args.cnet_name, args.mnet_name)

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(args):

    if args.use_wandb:
        logger = wandb.run
    else:
        logger = None

    train_dataset = CamelyonDataset(args.camelyon_data_path, args.train_centers, patch_size=args.patch_size, n_samples=args.n_samples_train, load_at_init=args.load_at_init)
    val_dataset = CamelyonDataset(args.camelyon_data_path, args.val_centers, patch_size=args.patch_size, n_samples=args.n_samples_val, load_at_init=args.load_at_init)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, sampler=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=None)

    model = build_model(args)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_decay < 1.0:
        lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay, patience=args.early_stop_patience//3, verbose=True)
    
    trainer = Trainer(model, optimizer, args.device, early_stop_patience=args.early_stop_patience, lr_sch=lr_sch, sigma_rui_sq=args.sigma_rui_sq, theta_val=args.theta_val, clip_grad=args.clip_grad, logger=logger)

    if args.pretrain_epochs > 0:
        pretrain_dataset = CamelyonDataset(args.camelyon_data_path, args.train_centers, patch_size=args.patch_size, n_samples=args.n_samples_train//args.pretrain_epochs, load_at_init=args.load_at_init)
        pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, sampler=None)
        print('Pretraining...')
        trainer.pretrain(args.pretrain_epochs, pretrain_dataloader, args.pretrain_theta_val)
    
    print('Training...')
    trainer.train(args.epochs, train_dataloader, val_dataloader)

    best_model = trainer.get_best_model()

    if not os.path.exists(os.path.dirname(args.weights_path)):
        os.makedirs(os.path.dirname(args.weights_path))

    if torch.cuda.device_count() > 1:
        state_dict = best_model.module.state_dict()
    else:
        state_dict = best_model.state_dict()
    torch.save(state_dict, args.weights_path)

def test(args):
    wssb_dataloader_dic = {}
    for organ in ['Lung', 'Breast', 'Colon']:
        wssb_dataset = WSSBDatasetTest(args.wssb_data_path, organ_list=[organ], load_at_init=False)
        wssb_dataloader_dic[organ] = torch.utils.data.DataLoader(wssb_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    cam_dataset = CamelyonDataset(args.camelyon_data_path, args.val_centers, patch_size=args.patch_size, n_samples=args.n_samples_val, load_at_init=False)
    cam_dataloader = torch.utils.data.DataLoader(cam_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


    model = build_model(args)
    weights_dict = torch.load(args.weights_path)
    model.load_state_dict(weights_dict)

    metrics_organ = {}
    for organ in ['Lung', 'Breast', 'Colon']:
        m = evaluate_GT(model, wssb_dataloader_dic[organ], sigma_rui_sq=args.sigma_rui_sq, theta_val=args.theta_val, device='cpu')
        metrics_organ[organ] = m
    metrics = {}
    metrics_list = metrics_organ['Lung'].keys()

    for metric in metrics_list:
        metrics[f'test/wssb/{metric}'] = np.mean([metrics_organ[organ][metric] for organ in metrics_organ])
        for organ in metrics_organ:
            metrics[f'test/wssb/{organ}/{metric}'] = m[metric]
    
    m = evaluate(model, cam_dataloader, sigma_rui_sq=args.sigma_rui_sq, theta_val=args.theta_val)
    for metric in m:
        metrics[f'test/camelyon/{metric}'] = m[metric]

    for metric in metrics:
        print('{:<25s}: {:s}'.format(metric, str(metrics[metric])))

    if wandb.run is not None:
        wandb.log(metrics)

def main():
    args = set_opts()

    print('Arguments:')
    for arg in vars(args):
        print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))
    
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args)
        args.weights_path = wandb.run.dir + '/weights/best.pt'
    else:
        run_name = f"{args.cnet_name}_{args.mnet_name}_{args.pretraining_epochs}pe_{args.patch_size}ps_{args.theta_val}theta_{args.sigma_rui_sq}sigmarui_{args.n_samples_train}nsamples"
        args.weights_path = args.weights_dir + f'/{run_name}/best.pt'

    seed_everything()

    if args.mode == 'train':   
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'train_test':
        train(args)
        test(args)
    
    print('Done!')

if __name__ == "__main__":
    main()