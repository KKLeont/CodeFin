import torch, math
import pandas as pd
import lightning as pl
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda')


class TCNLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation_rate):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, 
                              padding=(kernel_size-1) * dilation_rate, dilation=dilation_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    
class TCN(nn.Module):
    def __init__(self, input_channels, window, output_dim):
        super(TCN, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, output_dim)
                
    def forward(self, x):
        x = self.mlp(x)
        x = self.fc(x)
        return x
    

    
class ResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualNet, self).__init__()
        self.mu_model = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.logvar_model = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(input_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu = self.mu_model(x)
        logvar = self.logvar_model(x)
        x = self.reparameterize(mu, logvar)
        # print('noise reparameterize shape: ', x.shape)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        # print('noise shape: ', x.shape)
        return x, mu, logvar
    
class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_size, hiddens, num_layers, output_size):
        super().__init__()
        self.hidden_size = hiddens
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, 
                            hidden_size=hiddens, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.fc = torch.nn.Linear(hiddens, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))  
   
        out = self.fc(out[:, -1, :])
        # print('out.shape: ',out.shape)
        out = out.reshape(-1)
        # print('out.shape.reshape: ',out.shape)
        return out

class SignalNoiseDecomposer(nn.Module):
    def __init__(self, input_size, hidden_size, length, output_size):
        super().__init__()
        self.signal_encoder = TCN(input_channels=input_size, window=length, output_dim=output_size)
        self.noise_encoder = ResidualNet(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size)
        self.predictor = LSTMPredictor(input_size=output_size, hiddens=256, num_layers=2, output_size=1)

    def forward(self, input):
        signal = self.signal_encoder(input)
        noise, mu, logvar = self.noise_encoder(input)
        # print('noise: ', noise)
        prediction = self.predictor(signal)
        return signal, noise, prediction, mu, logvar
        
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim), 
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, signal, noise):
        combined = torch.cat([signal, noise], dim=-1)  
        return self.net(combined)

class LightningModel(pl.LightningModule):
    def __init__(self, input_size, length, learning_rate):
        super().__init__()
        self.automatic_optimization = False
        self.input_size = input_size
        self.length = length
        self.lr_g = learning_rate
        self.lr_d = learning_rate
        self.decomposer = SignalNoiseDecomposer(input_size=input_size, hidden_size=128, length=length, output_size=4)
        self.discriminator = Discriminator(input_dim=4, hidden_dim=128)
        self.recon_loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()
        self.pred_loss = nn.MSELoss()
        self.lambda_adv = 0.5
        self.lambda_sparse = 0.3
        self.lambda_pred = 1
        self.lambda_kl = 0.1

    def kl_divergence(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-20, max=20)  
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp())
        return kl.mean()

    def training_step(self, batch, batch_idx):
        batch_data = batch
        data = batch_data[0].to(self.device)
        code = batch_data[1]
        # timestamp_embedding = batch_data[2].to(self.device)
        label = batch_data[3].to(self.device)
        # print(label)
        label = label[:,-1]
      
        
        opt_g, opt_d = self.optimizers()
        
        # discrimiantor train
        self.toggle_optimizer(opt_d)

        s, n, _, _, _ = self.decomposer(data)
        real_labels = torch.ones(data.size(0), self.length, 1, device=self.device)
        
        # discrimiantor loss
        n_shuffle = n[torch.randperm(n.size(0))]
        real_output = self.discriminator(s.detach(), n_shuffle.detach())
        fake_output = self.discriminator(s.detach(), n.detach())
        # print('real_output: ', real_output.shape)
        # print('real_labels: ', real_labels.shape)
        d_loss_real = self.adv_loss(real_output, real_labels)
        d_loss_fake = self.adv_loss(fake_output, 1 - real_labels)
        d_loss = d_loss_real + d_loss_fake
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # decomposer train
        self.toggle_optimizer(opt_g)

        s, n, pred, mu, logvar = self.decomposer(data)

        # decomposer loss
        # print('s+n.shape: ', (s + n).shape)
        # print('data.shape: ', data.shape)
        recon_loss = self.recon_loss(s + n, data)
        sparse_loss = torch.mean(torch.abs(n))
        # print('label: ', label.shape )
        pred_loss = self.pred_loss(pred, label)
        fake_output = self.discriminator(s, n)
        g_loss_adv = self.adv_loss(fake_output, real_labels)
        kl_loss = self.kl_divergence(mu, logvar)

        # total loss
        # + self.lambda_adv*g_loss_adv
        total_loss = recon_loss + self.lambda_kl*kl_loss + self.lambda_sparse*sparse_loss + self.lambda_pred*pred_loss + self.lambda_adv*g_loss_adv
        # total_loss = recon_loss

        opt_g.zero_grad()
        self.manual_backward(total_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log_dict(
            {
                "train/d_loss": d_loss,
                "train/g_loss": total_loss,
                "train/recon_loss": recon_loss,
                "train/kl_loss": kl_loss,
                "train/sparse_loss": sparse_loss,
                "train/pred_loss": pred_loss,
                "train/adv_loss": g_loss_adv
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )
        return total_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.decomposer.parameters(), lr=self.lr_g)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        return [opt_g, opt_d]