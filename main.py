import torch
import numpy as np
# from diffusion.diffusion_1d import Unet1D, GaussianDiffusion1D
from diffusion.Unet1D import Unet1D
from diffusion.diffusion_pytorch_id import GaussianDiffusion1D
from dataset import *
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
import torch.nn.functional as F
from evaluate import *
import datetime
import os
import pandas as pd
from pathlib import Path

def get_mean_dev(y):
    b=y.shape[0]
    y=y.reshape(b,-1)
    mean=np.mean(y,axis=1)
    variance = np.var(y,axis=1)
    # print(mean,variance)
    return mean , variance

# Specify output directory here
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# time
cur_time=datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
time_out_dir=os.path.join(output_dir,cur_time)
if not os.path.exists(time_out_dir):
    os.makedirs(time_out_dir)
#use gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

Batch_Size=128
norm_type = '1-1'
index='SQ'

if index=='SQ':
    datasets,SQ_data,cond=build_dataset(
        dataset_type='SQ',
        b=Batch_Size,
        normlizetype=norm_type,
        rpm=19,
        state='outer3')
    indices = np.random.choice(
        len(SQ_data),
        size=Batch_Size,
        replace=False)
    data_np = np.array(SQ_data)[indices]
    sr = 25600
elif index=='CW':
    datasets, data_np, cond = build_dataset(
        dataset_type='CW',
        normlizetype=norm_type,
        ch=5)
    sr = 12000
else:
    datasets, data_np, cond = build_dataset(
        dataset_type='CW',
        normlizetype=norm_type,
        ch=5)
    sr = 12000

ori_path=os.path.join(output_dir,cur_time,cond+'_ori.png')
plot_np(data_np,path= None,show_mode=False)

train_dataloader = DataLoader(datasets["train"], batch_size=Batch_Size, shuffle=True)
# val_dataloader = DataLoader(datasets["val"], batch_size=Batch_Size, shuffle=False)

def linear_beta_schedule(timesteps):
    beta_start = 0.000001
    beta_end = 0.01
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 1000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


model = Unet1D(
    dim = 32,
    num_layers=4,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

model.to(device)


optimizer = AdamW(params=model.parameters(),lr=1e-4,betas=(0.9, 0.999), eps=1e-08,weight_decay=0.1)
# optimizer = Adam(model.parameters(), lr=1e-4)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 1024,
    timesteps = 1000,
    objective = 'pred_v',
    beta_schedule='linear',
    auto_normalize=False
)

diffusion.to(device)

epochs = 200

for epoch in range(epochs):
    for step, (inputs, labels) in enumerate(train_dataloader):
      # print('batch',type(inputs))
      optimizer.zero_grad()

      batch_size = inputs.shape[0]
      batch = inputs.to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        learning_rate = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("Epoch:{},Loss:{};Mem:{}MB,Lr:{}".format(epoch,loss.item(),memory_used,learning_rate))

      loss.backward()
      optimizer.step()


# save_model
save_index=True
if save_index is True:
    model_path=time_out_dir
    paths = [
        f'./pretrained/{index}/best_{cur_time}_{cond}.pt',
        f'./results/{index}/{cur_time}_{cond}.csv',
    ]
    for path in paths:
        dirname = os.path.dirname(path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), paths[0])

sampled_seq = diffusion.sample(batch_size = Batch_Size)
# for i in range(16):

out_path=cond+'_out.png'
out_np=sampled_seq.cuda().data.cpu().numpy()
# plot_np(out_np,path= None, show_mode=False)
plot_two_np(out_np,data_np,path=os.path.join(time_out_dir,cur_time+cond+'_time.png'),show_mode='time',sample_rate=sr)
plot_two_np(out_np,data_np,path=os.path.join(time_out_dir,cur_time+cond+'_fft.png'),show_mode='fft',sample_rate=sr)

df_m,df_v=get_mean_dev(sampled_seq.cuda().data.cpu().numpy())
# print(sampled_seq.shape) # (4, 32, 128)t