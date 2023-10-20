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
from Resnet1d import resnet18
from evaluate.visualize_utils import TSNEVisualizer

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

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
Batch_Size = 128
norm_type = 'mean-std'
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

print("dif_object:{},beta_schedule:{},beta:{}-{};epochs:{};diffusion time step:{};loss type:{}"
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
        rpm=39,
        state='outer3', # normal,inner,outer
        data_num=data_num,
        length=length,
        use_label=True,
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
print("condition:{}".format(cond))
train_dataset,val_dataset=get_loaders(
    datasets['train'],
    val_ratio=0.3,
    batch_size=Batch_Size,
    with_test=False,
    with_label=False,
)
train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True,num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False,num_workers=8,drop_last=False)
print("train_num:{};val_num:{}".format(len(train_dataset),len(val_dataset)))

model=resnet18()
model.load_state_dict(torch.load('//results/resnet18.pt'))
model.to(device)
# optimizer = AdamW(params=model.parameters(),lr=1e-4,betas=(0.9, 0.999), eps=1e-08,weight_decay=0.1)
#
# criterion = torch.nn.CrossEntropyLoss()
#
# for epoch in range(200):
#     model.train()
#     optimizer.zero_grad()
#     loss_meter = AverageMeter()
#     acc1_meter = AverageMeter()
#
#     for idx, (samples, targets,_) in enumerate(train_dataloader):
#         samples = samples.cuda(non_blocking=True)
#         targets = targets.cuda(non_blocking=True)
#         outputs = model(samples)
#         loss = criterion(outputs, targets)
#         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#         loss.backward()
#         optimizer.zero_grad()
#         loss_meter.update(loss.item(), targets.size(0))
#         acc1_meter.update(acc1.item(), targets.size(0))
#     print(f'epoch{epoch:.2f} loss {loss_meter.avg:.4f} Acc@1 {acc1_meter.avg:.3f}\t')
# torch.save(model.state_dict(), './results/resnet18.pt')
tsne_out = TSNEVisualizer(model, val_dataloader, device)
tsne_out.visualize()