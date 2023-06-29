import torch
import numpy as np
from diffusion.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D
from dataset.CWRU import CWRU
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
import torch.nn.functional as F
from evaluate import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mean_dev(y):
    b=y.shape[0]
    y=y.reshape(b,-1)
    mean=np.mean(y,axis=1)
    variance = np.var(y,axis=1)
    # print(mean,variance)
    return mean , variance


Batch_Size=128

data_dir = "/home/lucian/Documents/datas/CW"
normlizetype = '0-1'
data_set=CWRU(data_dir, normlizetype)
datasets={}
# datasets['train'], datasets['val'] = data_set.data_preprare()
k=0
datasets['train']= data_set.data_ch(ch=k)
cw_data=[]
for i in datasets['train']:
    cw_data.append(i[0])
cw_np=np.array(cw_data)
print(cw_np.shape)
gradient = np.gradient(cw_np,axis=2)
print(gradient.shape)
plot_np(gradient)
plot_np(cw_np,show_mel=True)

cw_m,cw_v=get_mean_dev(cw_np)


train_dataloader = DataLoader(datasets["train"], batch_size=Batch_Size, shuffle=True)
# val_dataloader = DataLoader(datasets["val"], batch_size=Batch_Size, shuffle=False)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 800

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
    objective = 'pred_v'
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

      loss = p_losses(model, batch, t, loss_type="l1")

      if step % 100 == 0:
        learning_rate = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("Epoch:{},Loss:{};Mem:{}MB,Lr:{}".format(epoch,loss.item(),memory_used,learning_rate))

      loss.backward()
      optimizer.step()

sampled_seq = diffusion.sample(batch_size = Batch_Size)
# for i in range(16):
fig=plt.figure(figsize=(12,3))
plot_np(sampled_seq.cuda().data.cpu().numpy(),show_mel=True)
df_m,df_v=get_mean_dev(sampled_seq.cuda().data.cpu().numpy())
# print(sampled_seq.shape) # (4, 32, 128)t