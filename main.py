import torch
import numpy as np
import math
# from diffusion.diffusion_1d import Unet1D, GaussianDiffusion1D
from model.diffusion.Unet1D import Unet1D_crossatt,Unet1D
from model.diffusion.diffusion import GaussianDiffusion1D
from dataset import *
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from evaluate import *
import datetime
import os
from pathlib import Path
from utils.logger import create_logger
from torch.optim import lr_scheduler

def get_mean_dev(y):
    b=y.shape[0]
    y=y.reshape(1,b)
    mean=np.mean(y,axis=1)
    variance = np.var(y,axis=1)
    # print(mean,variance)
    return mean, variance

def default_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Specify output directory here
output_dir = "./output"
default_dir(output_dir)
Batch_Size = 32
norm_type = '1-1'
index='SQ_M'

length=2048
data_num='all'
patch = 8 if Batch_Size >= 64 else 4


cond_np=None
# time
new_dir=default_dir(os.path.join(output_dir,index))
cur_time=datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
time_out_dir=default_dir(os.path.join(new_dir,cur_time))
logger = create_logger(output_dir=time_out_dir,  name=f"{index}.txt")
logger.info("index:{},norm_type:{};data_num:{}"
                .format(index,norm_type,data_num))


# diffusion para
dif_object = 'pred_v'
beta_schedule= 'linear'
beta_start = 0.0001
beta_end = 0.02
timesteps = 1000
epochs = 200
loss_type='huber'

logger.info("dif_object:{},beta_schedule:{},beta:{}-{};epochs:{};diffusion time step:{};loss type:{}"
                .format(dif_object,beta_schedule,beta_start,beta_end,epochs,timesteps,loss_type))

#use gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

if index=='SQ':
    datasets,SQ_data,cond=build_dataset(
        dataset_type='SQ',
        b=Batch_Size,
        normlizetype=norm_type,
        #rpm=19,
        state='normal',
        data_num=data_num,
        length=length,
        )
    indices = np.random.choice(
        len(SQ_data),
        size=Batch_Size,
        replace=False)
    data_np = np.array(SQ_data)[indices]
    sr = 25600

elif index=='SQ_M':
    datasets,SQ_data,cond,SQ_C=build_dataset(
        dataset_type='SQ_M',
        b=Batch_Size,
        normlizetype=norm_type,
        #rpm=19,
        state='outer3', # normal,inner,outer
        data_num=data_num,
        length=length,
        )
    indices = np.random.choice(
        len(SQ_data),
        size=Batch_Size,
        replace=False)
    data_np = np.array(SQ_data)[indices]
    cond_np= np.array(SQ_C)[indices]
    sr = 25600

elif index=='SQV':
    datasets,SQ_data,cond=build_dataset(
        dataset_type='SQV',
        b=Batch_Size,
        normlizetype=norm_type,
        state='NC',
        data_num=data_num,
        length=length,
    )
    indices = np.random.choice(
        len(SQ_data),
        size=Batch_Size,
        replace=False)
    data_np = np.array(SQ_data)[indices]
    sr = 25600

elif index=='SQV_M':
    datasets,SQV_data,cond,SQV_C=build_dataset(
        dataset_type='SQV_M',
        b=Batch_Size,
        normlizetype=norm_type,
        state='OF_3',
        data_num=data_num,
        length=length,
        )
    indices = np.random.choice(
        len(SQV_data),
        size=Batch_Size,
        replace=False)
    data_np = np.array(SQV_data)[indices]
    cond_np= np.array(SQV_C)[indices]
    sr = 25600

elif index=='CW':
    datasets, data_np, cond = build_dataset(
        dataset_type='CW',
        normlizetype=norm_type,
        ch=5,
        data_num=data_num,
        length=length,
    )
    sr = 12000
else:
    raise ('unexpected data index, please choose data index form SQ,SQV,SQ_M,SQV_M,CW')

# plot origin data
logger.info("condition:{}".format(cond))
ori_path=os.path.join(time_out_dir,cur_time+cond+'_ori.png')
if '_M' in index:
    plot_np(y=data_np,z=cond_np,patch=patch,path= ori_path,show_mode=False)
else:
    plot_np(y=data_np,patch=patch,path= ori_path,show_mode=False)

# if index in ['SQ' , 'SQ_M']:
#     target_label=19
#     train_dataset,val_dataset=split_dataset(datasets['train'],target_label)
#     train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True,drop_last=True)
# else:
train_dataset,val_dataset=get_loaders(
    datasets['train'],
    val_ratio=0.3,
    batch_size=Batch_Size,
    with_test=False,
    with_label=False,
)
train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8,drop_last=True)
logger.info("train_num:{};val_num:{}".format(len(train_dataset),len(val_dataset)))


# define beta schedule
def linear_beta_schedule(timesteps,beta_start = 0.0001,beta_end = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

betas = linear_beta_schedule(
    timesteps=timesteps,
    beta_start=beta_start,
    beta_end=beta_end)

#define alphas
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

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1",context=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, context=context)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

model = Unet1D_crossatt(
    dim=32,
    num_layers=4,
    dim_mults=(1, 2, 4, 8),
    #context_dim=length,
    channels=1,
    use_crossatt=False
)
# model=Unet1D(
#     dim=32,
#     num_layers=4,
#     dim_mults=(1, 2, 4, 8),
#     channels=1,
# )


model.to(device)


optimizer = AdamW(params=model.parameters(),lr=1e-4,betas=(0.9, 0.999), eps=1e-08,weight_decay=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# optimizer = Adam(model.parameters(), lr=1e-4)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = length,
    timesteps = timesteps, #1000
    objective = dif_object, #'pred_v'
    beta_schedule=beta_schedule,#'linear'
    auto_normalize=False
)

diffusion.to(device)


for epoch in range(epochs):
    for step, (inputs, labels,context) in enumerate(train_dataloader):
      optimizer.zero_grad()
      batch_size = inputs.shape[0]
      batch = inputs.to(device).float()
      context = context.to(device).float()

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()
      loss = p_losses(model, batch, t, loss_type=loss_type,context=context)

      if step % 100 == 0:
        learning_rate = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info("Epoch:{},Loss:{};Mem:{}MB,Lr:{}".format(epoch,loss.item(),memory_used,learning_rate))

      loss.backward()
      optimizer.step()
    scheduler.step()

# save_model check
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

rsme_seqs= []
all_val_inputs = []
all_val_conds = []
all_sampled_seqs = []
all_psnr=[]
all_cos=[]

for batch in val_dataloader:

    val_input = batch[0].to(device).float()
    val_conds = batch[2].to(device).float()
    B=val_input.shape[0]

    if '_M' in index:
        sampled_seq = diffusion.sample(batch_size=B)#, cond=val_conds)
        evl_out = eval_all(sampled_seq.detach().cpu().numpy(), val_input.detach().cpu().numpy())
        all_val_inputs.append(val_input)
        all_val_conds.append(val_conds)
        all_sampled_seqs.append(sampled_seq)
        rsme_seqs.append(evl_out[0])
        all_psnr.append(evl_out[1])
        all_cos.append(evl_out[2])

all_val_inputs = torch.cat(all_val_inputs, dim=0)
all_val_conds = torch.cat(all_val_conds, dim=0)
all_sampled_seqs = torch.cat(all_sampled_seqs, dim=0)

all_rsme = np.concatenate(rsme_seqs, axis=None).reshape(-1)
rsme_mean, rsme_var = get_mean_dev(all_rsme)
print('RSME', rsme_mean, rsme_var, all_rsme)

all_psnrs = np.concatenate(all_psnr, axis=None).reshape(-1)
psnr_mean, psnr_var = get_mean_dev(all_psnrs)
print('PSNR', psnr_mean, psnr_var, all_psnrs)

all_frecos = np.concatenate(all_cos, axis=None).reshape(-1)
cos_mean, cos_var = get_mean_dev(all_frecos)
print('fre_cos', cos_mean, cos_var, all_frecos)
print('All_eval', rsme_mean, rsme_var, psnr_mean, psnr_var, cos_mean, cos_var)

# for i in range(16):
out_path=cond+'_out.png'
out_np=all_sampled_seqs[:Batch_Size].cuda().data.cpu().numpy()

val_data_path=os.path.join(time_out_dir,cur_time+cond+'.npy')
val_in=all_val_inputs[:Batch_Size].cuda().data.cpu().numpy()
val_conds_out=all_val_conds[:Batch_Size].cuda().data.cpu().numpy()
out_npy=np.concatenate((out_np,val_in,val_conds_out),axis=1)
np.save(val_data_path,arr=out_npy)

# add condition in plot when use cross attention
if '_M' in index:
    val_data_path=os.path.join(time_out_dir,cur_time+cond+'.npy')
    val_output=val_input.cuda().data.cpu().numpy()
    val_conds_out=val_conds.cuda().data.cpu().numpy()
    out_npy=np.concatenate((out_np,val_output,val_conds_out),axis=1)
    np.save(val_data_path,arr=out_npy)
    plot_two_np(x=out_np,
                y=val_output,
                z1=None,
                z2=val_conds_out,
                patch=patch,
                path=os.path.join(time_out_dir,cur_time+cond+'_time.png'),
                show_mode='time',
                sample_rate=sr)
    plot_two_np(x=out_np,
                y=val_input.cuda().data.cpu().numpy(),
                patch=patch,
                path=os.path.join(time_out_dir,cur_time+cond+'_fft.png'),
                show_mode='fft',
                sample_rate=sr)
else:
    plot_two_np(x=out_np,
                y=data_np,
                path=os.path.join(time_out_dir,cur_time+cond+'_time.png'),
                patch=patch,
                show_mode='time',
                sample_rate=sr)
    plot_two_np(x=out_np,
                y=data_np,
                patch=patch,
                path=os.path.join(time_out_dir,cur_time+cond+'_fft.png'),
                show_mode='fft',
                sample_rate=sr)
#
# df_m,df_v=get_mean_dev(sampled_seq.cuda().data.cpu().numpy())
# print(sampled_seq.shape) # (4, 32, 128)t