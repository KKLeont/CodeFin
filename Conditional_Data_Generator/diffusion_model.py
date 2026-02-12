import torch, math
import pandas as pd
import lightning as pl
import matplotlib.pyplot as plt
import code_embedding
import trend_emb
from mean_std_embedding import get_mean_std_emb
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda')

class DDIMScheduler():

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"):
        self.num_train_timesteps = num_train_timesteps
        if beta_schedule == "scaled_linear":
            betas = torch.square(torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps, dtype=torch.float32))
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented")
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).tolist()
        self.set_timesteps(num_train_timesteps)


    def set_timesteps(self, num_inference_steps, denoising_strength=1.0):
        max_timestep = max(round(self.num_train_timesteps * denoising_strength) - 1, 0)
        num_inference_steps = min(num_inference_steps, max_timestep + 1)
        if num_inference_steps == 1:
            self.timesteps = [max_timestep]
        else:
            step_length = max_timestep / (num_inference_steps - 1)
            self.timesteps = [round(max_timestep - i*step_length) for i in range(num_inference_steps)]


    def denoise(self, model_output, sample, alpha_prod_t, alpha_prod_t_prev):
        weight_e = math.sqrt(1 - alpha_prod_t_prev) - math.sqrt(alpha_prod_t_prev * (1 - alpha_prod_t) / alpha_prod_t)
        weight_x = math.sqrt(alpha_prod_t_prev / alpha_prod_t)
        prev_sample = sample * weight_x + model_output * weight_e
        return prev_sample


    def step(self, model_output, timestep, sample, to_final=False):
        alpha_prod_t = self.alphas_cumprod[timestep]
        timestep_id = self.timesteps.index(timestep)
        if to_final or timestep_id + 1 >= len(self.timesteps):
            alpha_prod_t_prev = 1.0
        else:
            timestep_prev = self.timesteps[timestep_id + 1]
            alpha_prod_t_prev = self.alphas_cumprod[timestep_prev]

        return self.denoise(model_output, sample, alpha_prod_t, alpha_prod_t_prev)
    
    
    def add_noise(self, original_samples, noise, timestep):
        sqrt_alpha_prod = math.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alpha_prod = math.sqrt(1 - self.alphas_cumprod[timestep])
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

class Attention(torch.nn.Module):
    def __init__(self, num_features, heads, embed_dim, num_layers, sequence_length, num_train_timesteps):
        super().__init__()
        self.num_features = num_features
        self.heads = heads
        self.fc1 = torch.nn.Linear(num_features, out_features=embed_dim)
        self.GELU = torch.nn.GELU()
        self.mlps = torch.nn.ModuleList([torch.nn.Linear(sequence_length, sequence_length) for _ in range(num_layers)])
        self.attentions = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.heads, batch_first=True) for _ in range(num_layers)])
        self.fc2 = torch.nn.Linear(embed_dim, num_features)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.timestep_embedding = torch.nn.Parameter(torch.zeros(num_train_timesteps, sequence_length, 1))
        self.code_emb = code_embedding.code_emb
        self.mean_std_emb = get_mean_std_emb
        self.fc3 = torch.nn.Linear(32, 512)
        self.mlp2 = torch.nn.Linear(512, 512)
        self.trend_emb = trend_emb.TrendEmb
        self.emb_fc = torch.nn.Linear(292,512)

    def forward(self, x, timesteps, code, trend):
        x = self.fc1(x)
        x = self.GELU(x)
        # print('!!!trend: ',len(trend))
        stock_embedding = self.code_emb(code).get_stock_embedding().to(device)
        # print('stock_embedding.shape: ', stock_embedding.shape)
        # stock_embedding = stock_embedding.squeeze(0)
        # stock_embedding = self.fc3(stock_embedding)

        trend_embedding = self.trend_emb(trend).get_trend_embedding().to(device)
        # print('trend_embedding.shape: ', trend_embedding.shape)
        # trend_embedding = self.fc_trend(trend_embedding)
        

        means_stds_emb = self.mean_std_emb(code).to(device)

        mean_o = means_stds_emb[:,0,:].unsqueeze(1)
        std_o = means_stds_emb[:,1,:].unsqueeze(1)
        mean_h = means_stds_emb[:,2,:].unsqueeze(1)
        std_h = means_stds_emb[:,3,:].unsqueeze(1)
        mean_l = means_stds_emb[:,4,:].unsqueeze(1)
        std_l = means_stds_emb[:,5,:].unsqueeze(1)
        mean_c = means_stds_emb[:,6,:].unsqueeze(1)
        std_c = means_stds_emb[:,7,:].unsqueeze(1)

        control_emb = torch.cat((mean_o, std_o, mean_h, std_h, mean_l, std_l, mean_c, std_c, stock_embedding, trend_embedding),dim=2)
        control_emb = self.emb_fc(control_emb)

        stock_embedding = stock_embedding = self.fc3(stock_embedding)

        for norm, mlp, attention in zip(self.norms, self.mlps, self.attentions):
            res = x
            x = norm(x)
            x = x + self.timestep_embedding[timesteps] + stock_embedding + control_emb
            x1 = x.permute(0, 2, 1) # x.shape before permute:  torch.Size([1, 30, 512])
            x1 = mlp(x1)
            x1 = self.GELU(x1)
            x1 = x1.permute(0, 2, 1)
            #TODO: x = x+ x
            x2 = self.mlp2(x)
            x2 = self.GELU(x2)
            x = x1 + x2
            x, _ = attention(x, x, x)
            x = x + res
        x = self.fc2(x)
        return x
    


class LightningModel(pl.LightningModule):
    def __init__(self, input_size, length, learning_rate):
        super().__init__()
        self.input_size = input_size
        self.length = length
        self.learning_rate = learning_rate
        self.noise_scheduler = DDIMScheduler()
        self.denoising_model = Attention(num_features=self.input_size, 
                                         heads=32, 
                                         embed_dim=512, 
                                         num_layers=32, 
                                         sequence_length=self.length, num_train_timesteps=1000)


    def training_step(self, batch, batch_idx):
        batch_data = batch
        data = batch_data[0].to(self.device)
        code = batch_data[1]
        # print('code: ', code)
        # timestamp_embedding = batch_data[2].to(self.device)
        label = batch_data[3].to(self.device)
        # print(label)
        label = label[:,-1]
        trend = list(batch_data[4])

       
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (data.shape[0],), device=self.device)
        noise = torch.randn_like(data)
        noisy_data = torch.stack([self.noise_scheduler.add_noise(data[i], noise[i], timesteps[i]) for i in range(data.shape[0])])
        predicted_noise = self.denoising_model(noisy_data, timesteps, code, trend)

        diffusion_loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction="mean")
        self.log("train_diffusion_loss", diffusion_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return diffusion_loss


    def configure_optimizers(self):
        optimizer_diffusion = torch.optim.AdamW(self.denoising_model.parameters(), lr=self.learning_rate)

        return optimizer_diffusion
    
    
    @torch.no_grad()
    def sampling_loop(self, num_inference_steps, sequence_length, num_features, guidance_scale, code, trend):
        gelu = torch.nn.GELU()
        sample = torch.randn((1, sequence_length, num_features), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            timestep = torch.LongTensor((timestep,))
            condi_noise = self.denoising_model(sample, timestep, code, trend) #CFG
            uncondi_noise = self.denoising_model(sample, timestep, "avg", 'Volatile')
            # predicted_noise = (uncondi_noise ) * guidance_scale + condi_noise
            predicted_noise = (condi_noise - uncondi_noise) * guidance_scale + uncondi_noise
            sample = self.noise_scheduler.step(predicted_noise, timestep, sample)
        sample = gelu(sample)
        return sample
    
    @torch.no_grad()
    def seperate_signal(self, x, code):
        signal = self.sns_model(x, code)
        return signal
    
    @torch.no_grad()
    def sampling_loop_with_signal(self, num_inference_steps, sequence_length, num_features, guidance_scale, code):
        gelu = torch.nn.GELU()
        sample = torch.randn((1, sequence_length, num_features), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            timestep = torch.LongTensor((timestep,))
            condi_noise = self.denoising_model(sample, timestep, code) #CFG
            uncondi_noise = self.denoising_model(sample, timestep, "avg")
            predicted_noise = (condi_noise - uncondi_noise) * guidance_scale + uncondi_noise
            sample = self.noise_scheduler.step(predicted_noise, timestep, sample)
        sample = gelu(sample)
        signal = self.sns_model(sample, code)
        return signal

    
    @torch.no_grad()
    def augmenting_sampling_loop(self, num_inference_steps, sequence_length, num_features, org_data):
        noise = torch.randn((1, sequence_length, num_features), device=self.device)
        self.noise_scheduler.set_timesteps(int(num_inference_steps*0.05), denoising_strength=0.05)
        # sample = self.noise_scheduler.add_noise(org_data, noise, self.noise_scheduler.timesteps[0])
        sample = org_data
        gelu = torch.nn.GELU()
        for timestep in self.noise_scheduler.timesteps:
            timestep = torch.LongTensor((timestep,))
            predicted_noise = self.denoising_model(sample, timestep)
            sample = self.noise_scheduler.step(predicted_noise, timestep, sample)
        sample = gelu(sample)
        return sample
    


