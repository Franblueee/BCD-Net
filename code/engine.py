import torch
import numpy as np
import copy

from tqdm import tqdm
from utils import rgb2od, od2rgb, direct_deconvolution, peak_signal_noise_ratio, structural_similarity

class LossBCD(torch.nn.Module):
    def __init__(self, sigma_rui_sq=0.05, theta_val=0.5) -> None:
        super().__init__()
        self.sigma_rui_sq = sigma_rui_sq
        self.theta_val = theta_val
        self.M_ruifrok = torch.tensor([
                    [0.6442, 0.0928],
                    [0.7166, 0.9541],
                    [0.2668, 0.2831]
                    ]).type(torch.float32)
    
    def forward(self, Y, M_mean, M_var, C_mean):
        """ Forward method for LossBCD
        Args:
            Y (tensor): (batch_size, 3, H, W)
            M_mean (tensor): (batch_size, 3, 2)
            M_var (tensor): (batch_size, 1, 2)
            C_mean (tensor): (batch_size, 2, H, W)
        """

        batch_size = Y.shape[0]
        M_ruifrok = self.M_ruifrok.repeat(batch_size, 1, 1).to(Y.device) # (batch_size, 3, 2)
        M_var = M_var.repeat(1, 3, 1) # (batch_size, 3, 2)        

        loss_kl = (0.5 / self.sigma_rui_sq) * torch.nn.functional.mse_loss(M_mean, M_ruifrok, reduction='none') + 1.5 * ( M_var / self.sigma_rui_sq - torch.log(M_var / self.sigma_rui_sq) - 1) # (batch_size, 3, 2)
        loss_kl = torch.sum(loss_kl) / batch_size # (1)

        M_sample = M_mean + torch.sqrt(M_var) * torch.randn_like(M_mean) # (batch_size, 3, 2)

        Y_rec = torch.einsum('bcs,bshw->bchw', M_sample, C_mean) # (batch_size, 3, H, W)

        loss_rec = torch.sum(torch.nn.functional.mse_loss(Y, Y_rec, reduction='none')) / batch_size # (1) 

        loss = (1.0 - self.theta_val)*loss_rec + self.theta_val*loss_kl

        return loss, loss_rec, loss_kl

def evaluate_GT(model, dataloader, sigma_rui_sq=0.05, theta_val=0.5, device='cuda'):

    criterion = LossBCD(sigma_rui_sq, theta_val)

    model = model.to(device)
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description("Test")

    metrics_dict = { 
        'loss': 0.0, 'loss_rec': 0.0, 'loss_kl': 0.0, 
        'mse_rec' : 0.0, 'psnr_rec': 0.0, 'ssim_rec': 0.0,
        'mse_gt_h': 0.0, 'mse_gt_e': 0.0, 'mse_gt': 0.0,
        'psnr_gt_h': 0.0, 'psnr_gt_e': 0.0, 'psnr_gt': 0.0,
        'ssim_gt_h': 0.0, 'ssim_gt_e': 0.0, 'ssim_gt': 0.0
        }

    with torch.no_grad():
        for batch_idx, batch in pbar:
            
            Y, M_gt = batch # Y: (batch_size, 3, H, W), M_gt: (batch_size, 3, 2)
            batch_size, _, H, W = Y.shape
            Y = Y.to(device)
            Y_od = rgb2od(Y) # (batch_size, 3, H, W)
            M_gt = M_gt.to(device)
            C_gt = direct_deconvolution(Y_od, M_gt).to(device) # (batch_size, 2, H, W)

            H_gt_od = torch.einsum('bcs,bshw->bschw', M_gt, C_gt)[:,0,:,:] # (batch_size, H, W)
            H_gt = torch.clamp(od2rgb(H_gt_od), 0.0, 255.0) # (batch_size, 3, H, W)
            E_gt_od = torch.einsum('bcs,bshw->bschw', M_gt, C_gt)[:,1,:,:] # (batch_size, H, W)
            E_gt = torch.clamp(od2rgb(E_gt_od), 0.0, 255.0) # (batch_size, 3, H, W)

            M_mean, M_var, C_mean = model(Y_od) # M_mean: (batch_size, 3, 2), M_var: (batch_size, 3, 2), C_mean: (batch_size, 2, H, W)
            loss, loss_rec, loss_kl = criterion(Y_od, M_mean, M_var, C_mean)

            Y_rec_od = torch.einsum('bcs,bshw->bchw', M_mean, C_mean) # (batch_size, 3, H, W)
            Y_rec = torch.clamp(od2rgb(Y_rec_od), 0.0, 255.0) # (batch_size, 3, H, W)

            H_rec_od = torch.einsum('bcs,bshw->bschw', M_mean, C_mean)[:,0,:,:] # (batch_size, H, W)
            H_rec = torch.clamp(od2rgb(H_rec_od), 0.0, 255.0) # (batch_size, 3, H, W)
            E_rec_od = torch.einsum('bcs,bshw->bschw', M_mean, C_mean)[:,1,:,:] # (batch_size, H, W)
            E_rec = torch.clamp(od2rgb(E_rec_od), 0.0, 255.0) # (batch_size, 3, H, W)

            metrics_dict['loss'] += loss.item()
            metrics_dict['loss_rec'] += loss_rec.item()
            metrics_dict['loss_kl'] += loss_kl.item()

            metrics_dict['mse_rec'] += torch.sum(torch.pow(Y_rec_od - Y_od, 2)).item() / (3.0*H*W)
            metrics_dict['psnr_rec'] += torch.sum(peak_signal_noise_ratio(Y_rec, Y)).item()
            metrics_dict['ssim_rec'] += torch.sum(structural_similarity(Y_rec, Y)).item()

            metrics_dict['mse_gt_h'] += torch.sum(torch.pow(H_gt_od - H_rec_od, 2)).item() / (3.0*H*W)
            metrics_dict['mse_gt_e'] += torch.sum(torch.pow(E_gt_od - E_rec_od, 2)).item() / (3.0*H*W)

            metrics_dict['psnr_gt_h'] += torch.sum(peak_signal_noise_ratio(H_gt, H_rec)).item()
            metrics_dict['psnr_gt_e'] += torch.sum(peak_signal_noise_ratio(E_gt, E_rec)).item()

            metrics_dict['ssim_gt_h'] += torch.sum(structural_similarity(H_gt, H_rec)).item()
            metrics_dict['ssim_gt_e'] += torch.sum(structural_similarity(E_gt, E_rec)).item()
            
    metrics_dict['mse_gt'] = (metrics_dict['mse_gt_h'] + metrics_dict['mse_gt_e'])/2.0
    metrics_dict['psnr_gt'] = (metrics_dict['psnr_gt_h'] + metrics_dict['psnr_gt_e'])/2.0
    metrics_dict['ssim_gt'] = (metrics_dict['ssim_gt_h'] + metrics_dict['ssim_gt_e'])/2.0

    for key in metrics_dict.keys():
        metrics_dict[key] /= len(dataloader.dataset)

    return metrics_dict

def evaluate(model, dataloader, sigma_rui_sq=0.05, theta_val=0.5, device='cuda'):

    criterion = LossBCD(sigma_rui_sq, theta_val)

    model = model.to(device)
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description("Test")

    metrics_dict = { 
        'loss': 0.0, 'loss_rec': 0.0, 'loss_kl': 0.0, 
        'mse_rec' : 0.0, 'psnr_rec': 0.0, 'ssim_rec': 0.0,
        }

    with torch.no_grad():
        for batch_idx, batch in pbar:
            Y = batch # Y: (batch_size, 3, H, W), M_gt: (batch_size, 3, 2)
            batch_size, _, H, W = Y.shape
            Y = Y.to(device)
            Y_od = rgb2od(Y)

            M_mean, M_var, C_mean = model(Y_od) # M_mean: (batch_size, 3, 2), M_var: (batch_size, 3, 2), C_mean: (batch_size, 2, H, W)
            loss, loss_rec, loss_kl = criterion(Y_od, M_mean, M_var, C_mean)

            Y_rec_od = torch.einsum('bcs,bshw->bchw', M_mean, C_mean) # (batch_size, 3, H, W)
            Y_rec = torch.clamp(od2rgb(Y_rec_od), 0.0, 255.0) # (batch_size, 3, H, W)

            metrics_dict['loss'] += loss.item()
            metrics_dict['loss_rec'] += loss_rec.item()
            metrics_dict['loss_kl'] += loss_kl.item()

            metrics_dict['mse_rec'] += torch.sum(torch.pow(Y_rec_od - Y_od, 2)).item() / (3.0*H*W)
            metrics_dict['psnr_rec'] += torch.sum(peak_signal_noise_ratio(Y_rec, Y)).item()
            metrics_dict['ssim_rec'] += torch.sum(structural_similarity(Y_rec, Y)).item()

    for key in metrics_dict.keys():
        metrics_dict[key] /= len(dataloader.dataset)

    return metrics_dict

class Trainer:
    def __init__(self, model, optimizer, device='cuda', early_stop_patience=10, lr_sch=None, sigma_rui_sq=0.05, theta_val=0.5, clip_grad=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.early_stop_patience = early_stop_patience
        self.lr_sch = lr_sch
        self.logger = logger

        if self.early_stop_patience is None:
            self.early_stop_patience = np.inf

        self.sigma_rui_sq = sigma_rui_sq
        self.theta_val = theta_val
        self.clip_grad = clip_grad

        self.criterion = LossBCD(sigma_rui_sq=self.sigma_rui_sq, theta_val=self.theta_val)

        self.best_model = None
        self.best_metric = None
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
    
    def pretrain(self, max_epochs, train_dataloader, pretrain_theta_val=0.99):

        dif = self.theta_val - pretrain_theta_val
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.model.train()
        for epoch in range(1, max_epochs+1):
            theta = pretrain_theta_val + (epoch-1) * dif / max_epochs
            self.criterion.theta_val = theta

            # Train loop
            self.model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            pbar.set_description(f"Pretrain - Epoch {epoch}, theta: {theta:.4f}")
            sum_loss = 0.0
            sum_loss_kl = 0.0
            sum_loss_rec = 0.0
            sum_psnr_rec = 0.0
            for batch_idx, batch in pbar:
                Y_rgb = batch # (batch_size, 3, H, W)
                batch_size = Y_rgb.shape[0]
                Y_rgb = Y_rgb.to(self.device)
                Y_od = rgb2od(Y_rgb)

                self.optimizer.zero_grad()
                M_mean, M_var, C_mean = self.model(Y_od) # M_mean: (batch_size, 3, 2), M_var: (batch_size, 3, 2), C_mean: (batch_size, 2, H, W)
                loss, loss_rec, loss_kl = self.criterion(Y_od, M_mean, M_var, C_mean)
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

                sum_loss += loss.item()
                sum_loss_kl += loss_kl.item()
                sum_loss_rec += loss_rec.item()

                Y_rec_rgb = torch.clamp(od2rgb(torch.einsum('bcs,bshw->bchw', M_mean, C_mean).detach()), 0.0, 255.0) # (batch_size, 3, H, W)
                sum_psnr_rec += torch.mean(peak_signal_noise_ratio(Y_rec_rgb, Y_rgb)).item()

                train_metrics = {
                    'pretrain/psnr_rec' : sum_psnr_rec / (batch_idx + 1),
                    'pretrain/loss' : sum_loss / (batch_idx + 1),
                    'pretrain/loss_kl' : sum_loss_kl / (batch_idx + 1),
                    'pretrain/loss_rec' : sum_loss_rec / (batch_idx + 1)
                }
                pbar.set_postfix(train_metrics)

                if batch_idx == (len(train_dataloader) - 1):
                    if self.logger is not None:
                        self.logger.log(train_metrics)
            pbar.close()
        
        self.criterion.theta_val = self.theta_val
        self.best_model = self.copy_model()
    
    def train(self, max_epochs, train_dataloader, val_dataloader=None):

        if val_dataloader is None:
            val_dataloader = train_dataloader
        
        if self.best_model is None:
            self.best_model = self.copy_model()
        if self.best_metric is None:
            self.best_metric = -np.inf
        early_stop_count = 0
        self.model = self.model.to(self.device)
        for epoch in range(1, max_epochs+1):
            # Train loop
            self.model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            pbar.set_description(f"Train - Epoch {epoch} ")
            sum_psnr_rec = 0.0
            sum_loss = 0.0
            sum_loss_kl = 0.0
            sum_loss_rec = 0.0
            for batch_idx, batch in pbar:
                Y_rgb = batch # (batch_size, 3, H, W)
                batch_size = Y_rgb.shape[0]
                Y_rgb = Y_rgb.to(self.device)
                Y_od = rgb2od(Y_rgb)

                self.optimizer.zero_grad()

                M_mean, M_var, C_mean = self.model(Y_od) # M_mean: (batch_size, 3, 2), M_var: (batch_size, 1, 2), C_mean: (batch_size, 2, H, W)
                loss, loss_rec, loss_kl = self.criterion(Y_od, M_mean, M_var, C_mean)
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

                sum_loss += loss.item()
                sum_loss_kl += loss_kl.item()
                sum_loss_rec += loss_rec.item()

                Y_rec_rgb = torch.clamp(od2rgb(torch.einsum('bcs,bshw->bchw', M_mean, C_mean).detach()), 0.0, 255.0) # (batch_size, 3, H, W)
                sum_psnr_rec += torch.mean(peak_signal_noise_ratio(Y_rec_rgb, Y_rgb)).item()

                train_metrics = {
                    'train/psnr_rec' : sum_psnr_rec / (batch_idx + 1),
                    'train/loss' : sum_loss / (batch_idx + 1),
                    'train/loss_kl' : sum_loss_kl / (batch_idx + 1),
                    'train/loss_rec' : sum_loss_rec / (batch_idx + 1)
                }
                pbar.set_postfix(train_metrics)

                if batch_idx == (len(train_dataloader) - 1):
                    if self.logger is not None:
                        self.logger.log(train_metrics)
            pbar.close()

            # Validation loop
            self.model.eval()
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            pbar.set_description(f"Validation - Epoch {epoch}")
            sum_psnr_rec = 0.0
            sum_loss = 0.0
            sum_loss_kl = 0.0
            sum_loss_rec = 0.0
            with torch.no_grad():
                for batch_idx, batch in pbar:
                    Y_rgb = batch # (batch_size, 3, H, W)
                    Y_rgb = Y_rgb.to(self.device)
                    Y_od = rgb2od(Y_rgb)

                    M_mean, M_var, C_mean = self.model(Y_od)
                    loss, loss_rec, loss_kl = self.criterion(Y_od, M_mean, M_var, C_mean)

                    sum_loss += loss.item()
                    sum_loss_kl += loss_kl.item()
                    sum_loss_rec += loss_rec.item()

                    Y_rec_rgb = torch.clamp(od2rgb(torch.einsum('bcs,bshw->bchw', M_mean, C_mean).detach()), 0.0, 255.0) # (batch_size, 3, H, W)
                    sum_psnr_rec += torch.mean(peak_signal_noise_ratio(Y_rec_rgb, Y_rgb)).item()

                    val_metrics = {
                            'val/psnr_rec' : sum_psnr_rec / (batch_idx + 1),
                            'val/loss' : sum_loss / (batch_idx + 1),
                            'val/loss_kl' : sum_loss_kl / (batch_idx + 1),
                            'val/loss_rec' : sum_loss_rec / (batch_idx + 1)
                        }
                    pbar.set_postfix(val_metrics)
                    if batch_idx == (len(val_dataloader) - 1):
                        if self.logger is not None:
                            self.logger.log(val_metrics)
                
            if self.lr_sch is not None:
                self.lr_sch.step(val_metrics['val/psnr_rec'])

            if val_metrics['val/psnr_rec'] <= self.best_metric:
                early_stop_count += 1
                print(f'Early stopping count: {early_stop_count}')
            else:
                self.best_metric = val_metrics['val/psnr_rec']
                del self.best_model
                self.best_model = self.copy_model()
                early_stop_count = 0
                
            if early_stop_count >= self.early_stop_patience:
                print('Reached early stopping condition')
                break

            pbar.close()
    
    def copy_model(self):
        model_copy = copy.deepcopy(self.model.to('cpu')).to('cpu')
        self.model = self.model.to(self.device)
        return model_copy
    
    def get_best_model(self):
        return self.best_model