import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
# from utils.SequenceDatasets import dataset
from utils.sequence_transform import *
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,Subset
import torch
from collections import Counter

import matplotlib.pyplot as plt
from collections import Counter

box1={0:'NC',1:'NC',2:'IP+CF',3:'IP+CF',4:'MP',5:'MP',6:'IP',7:'IP',}
box2={0:'NC',1:'NC',2:'MP',3:'MP',4:'IP',5:'IP',6:'IP',7:'IP'}

def plot_label_counts(label_counter):
    # 提取标签和计数
    labels = label_counter.keys()
    counts = label_counter.values()

    # 创建柱状图
    fig, ax = plt.subplots()
    ax.bar(labels, counts)

    # 设置标题和标签
    ax.set_title('Label Counts')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')

    # 自动调整标签旋转
    plt.xticks(rotation=45)

    # 显示图形
    plt.show()

def hs_filter_df_by_labels(df, labels_dict):
    '''
    df: Input pandas dataframe with data path and labels
    labels_dict: Dictionary with label column names as keys and label values/ranges as values.
                 For example, {"Label1": "value1", "Label2": (lower_bound2, upper_bound2)}.
    Returns a new dataframe filtered by the given labels.
    '''
    if not labels_dict:
        raise ValueError("No labels provided for filtering.")
    df_filtered = df.copy()
    for label, value in labels_dict.items():
        if isinstance(value, tuple):  # Treat as a range
            lower_bound, upper_bound = value
            df_filtered = df_filtered[df_filtered[label].between(lower_bound, upper_bound)]
        else:  # Treat as a single value
            df_filtered = df_filtered[df_filtered[label] == value]
    return df_filtered

def create_labels_dict(**kwargs):
    """
    Creates a dictionary with label column names as keys and provided label values as values.
    **kwargs: Key-value pairs where keys are label names and values are desired label values.
    """
    return kwargs

def get_files(dir,is_train=True,mutli=None,**kwargs):
    '''

    :param dir: csv direction, pd.dataframe
    :return:
    '''
    csv_list=dir['path'].tolist()
    index=dir['box'].tolist()
    if mutli is None:
        data_dict ,label_dict= data_load(csv_list,index,**kwargs)
        return [data_dict ,label_dict]
    else:
        data_dict, label_dict = data_load(csv_list, index, **kwargs)
        assist_dict,_=data_load(csv_list, index,mutli=mutli, **kwargs)
        return [data_dict, label_dict, assist_dict]

def data_load(dir,labels,length=1024,data_num=20,mutli=None,ch=None):
    assert len(dir)==len(labels)
    data_all=[]
    label_all=[]
    # data_dict={}
    for i in range(len(dir)):
        data_df = pd.read_csv(dir[i])

        if mutli is None:
            fl = data_df.iloc[:, 2:].values
            if ch in range(8):
                indice = [ch]
            else:
                indice = range(8)
            fl_data = fl[:, indice]
        else:
            if isinstance(ch , int):
                if ch in [0,1,4,5]:
                    fl = data_df.iloc[:, 0].values.reshape(-1,1)
                elif ch in [2,3,6,7]:
                    fl = data_df.iloc[:, 1].values.reshape(-1,1)
            else:
                fl = data_df.iloc[:, 0].values.reshape(-1, 1)
            indice=[0]
            fl_data = fl

        if labels[i]==1:
            box=box1
        elif labels[i]==2:
            box=box2
        else:
            raise ('box choose is wrong')

            # print(fl.shape)
        for j in range(len(indice)):
            # print(j,indice[j],box[indice[j]])
            if data_num=='all':
                print(dir[i].split('/')[-3:],'all_data_num for one csv is', int(fl.shape[0]//length))
                data = [fl_data[start:end,j].reshape(-1,length) for start, end in zip(range(0, fl.shape[0], length),
                                                                        range(length, fl.shape[0] + 1,length))]
            elif data_num< fl.shape[0]//length:
                print(dir[i].split('/')[-3:], 'data_num for one csv is', data_num)
                data = [fl_data[i:i+length,j].reshape(-1,length) for i in range(0, data_num*length, length)]
            else:
                raise ('data length choose wrong')

            for k in data:
                # print('j', j.shape)
                data_all.append(k)
                label_temp=box[indice[j]]

                label_all.append(label_temp)

    return data_all, label_all

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            # Reshape(),
            Normalize(normlize_type),
            # RandomAddGaussian(),
            # RandomScale(),
            # RandomStretch(),
            # RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            # Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]

def get_loaders(train_dataset, seed, batch, val_ratio=0.2):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch,
                                  shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

    return train_dataloader, val_dataloader

def convert_labels_to_numbers(label_set):
    """
    Convert string labels in a set to numeric labels.
    label_set: Set of string labels.
    Returns a dictionary with numeric labels.
    """
    unique_labels = sorted(list(label_set))
    label_mapping = {label: index for index, label in enumerate(unique_labels)}

    numeric_label_dict = {label: label_mapping[label] for label in unique_labels}

    return numeric_label_dict

class HS(Dataset):

    def __init__(self, dir, normlizetype, is_train=True,**kwargs):
        self.data_dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        list_data= get_files(self.data_dir, is_train=self.is_train,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        self.cls_num=set(list_data[1])
        self.cls_no=convert_labels_to_numbers( self.cls_num)

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = None

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):

            data = self.seq_data[idx]
            label = self.labels[idx]

            if self.transform:
                data = self.transform(data)
            return data, label

    def get_classes_num(self):
        return len(self.cls_num)# num, name

    def get_class_number(self):
        return  self.cls_no

class HS_Mutli(Dataset):

    def __init__(self, dir, normlizetype, is_train=True,**kwargs):
        self.data_dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        list_data= get_files(self.data_dir, is_train=self.is_train,mutli=True,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1],'assist':list_data[2]})
        self.cls_num=set(list_data[1])
        self.cls_no=convert_labels_to_numbers( self.cls_num)

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = data_transforms('train', self.normlizetype)
            self.seq_assist = self.data_pd['assist'].tolist()
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = None
            self.seq_assist = self.data_pd['assist'].tolist()

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):

        data = self.seq_data[idx]
        if self.transform:
            data = self.transform(data)
        data= data
        label = self.labels[idx]
        assist=self.seq_assist[idx]

        return data, label,assist

    def get_classes_num(self):
        return len(self.cls_num)# num, name

    def get_class_number(self):
        return  self.cls_no

if __name__ == '__main__':
    ori_root = '/home/lucian/Documents/datas/Graduate_data/Highspeed_train/dataframe.csv'
    ori_csv_pd = pd.read_csv(ori_root)
    # print(df.info)
    labels_dict = create_labels_dict(speed=(100,350),box=1)
    # label=['path','label_all','box','rpm','load','state']

    # labels_dict={}
    df_out = hs_filter_df_by_labels(ori_csv_pd, labels_dict)
    # print(df_out.info)
    # out=get_files(df_out,data_num=20)

    normlizetype = 'mean-std'
    datasets = {}
    # datasets_train = HS(df_out, normlizetype, is_train=True,data_num=20,ch='V')

    datasets_train = HS_Mutli(df_out, normlizetype, is_train=True, data_num=10, ch=3)
    # train_dataloader, val_dataloader = get_loaders(datasets_train, seed=5, batch=128)
    #
    # for id, (data, label) in enumerate(train_dataloader):
    #     print(id, data.shape, label)
    # # datasets_test = HS(df_out,label_index, normlizetype, is_train=False)
    # cls_num_=datasets_train.get_classes_num()
    # cls=datasets_train.get_class_number()


