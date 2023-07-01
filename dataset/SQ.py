import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from utils.sequence_transform import *
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,Subset
import torch
from collections import Counter


import matplotlib.pyplot as plt
from collections import Counter

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


def filter_df_by_labels(df, labels_dict):
    '''
    df: Input pandas dataframe with data path and labels
    labels_dict: Dictionary with label column names as keys and desired label values as values.
                 For example, {"Label1": "value1", "Label2": "value2"}.
    Returns a new dataframe filtered by the given labels.
    '''
    if not labels_dict:
        raise ValueError("No labels provided for filtering.")
    df_filtered = df.copy()
    for label, value in labels_dict.items():
        df_filtered = df_filtered[df_filtered[label] == value]
    return df_filtered

def create_labels_dict(**kwargs):
    """
    Creates a dictionary with label column names as keys and provided label values as values.
    **kwargs: Key-value pairs where keys are label names and values are desired label values.
    """
    return kwargs


def get_files(dir,label='state',is_train=True,plot_counter=None,**kwargs):
    '''

    :param dir: csv direction, pd.dataframe
    :return:
    '''
    csv_list=dir['path'].tolist()
    csv_label=dir[label].tolist()
    if plot_counter is True:
        element_counts = Counter(csv_label)
        plot_label_counts(element_counts)
        print('label_counter',element_counts)


    data_dict ,label_dict= data_load(csv_list,labels=csv_label,**kwargs)

    return [data_dict ,label_dict]

def data_load(dir,labels,length=1024,data_num=20):
    assert len(dir)==len(labels)
    data_all=[]
    label_all=[]
    # data_dict={}
    for i in range(len(dir)):
        data_df = pd.read_table(dir[i], sep='\t', skiprows=range(1, 17))
        label_df = labels[i]

        # 获取第2列的数值并转换为数组
        fl = data_df.iloc[:, 1:].values
        # 划分数据
        if data_num=='all':
            print('all_data_num for one csv is', int(fl.shape[0]//length))
            data = [fl[start:end].reshape(1, length) for start, end in zip(range(0, fl.shape[0], length),
                                                                                range(length, fl.shape[0] + 1,
                                                                                      length))]
        elif data_num< fl.shape[0]//length:
             data = np.split(fl[:length * data_num, :], data_num, axis=0)
        else:
            raise ('data length choose wrong')
        # print(len(data))
        for j in data:
            # print('j', j.shape,label_df)
            data_all.append(j)
            label_all.append(label_df)
    # data_dict['data']=data_all
    # data_dict['label']=label_all
    # print(len(data_dict['data']),len(data_dict['label']))
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

class SQ(Dataset):

    def __init__(self, dir, labels_dict, label_index, normlizetype, is_train=True, dict=None,**kwargs):
        self.dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.label_index=label_index

        self.data_dir = filter_df_by_labels(self.dir, labels_dict)

        list_data= get_files(self.data_dir, label=self.label_index, is_train=self.is_train,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        self.cls_num=set(list_data[1])

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.transform = None

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        if self.is_train:
            data = self.seq_data[idx]
            label = self.labels[idx]

            if self.transform:
                data = self.transform(data)
            return data, label
        else:
            data = self.seq_data[idx]
            label = self.labels[idx]
            if self.transform:
                data = self.transform(data)
            return data,label

    def get_classes_num(self):
        return len(self.cls_num),self.cls_num # num, name

signal_size = 1024
if __name__ == '__main__':
    ori_root = '/home/lucian/Documents/datas/Graduate_data/SQdata/dataframe.csv'
    ori_csv_pd = pd.read_csv(ori_root)
    # print(df.info)
    labels_dict = create_labels_dict(rpm=9,state='normal')

    label_index='state'
    # print(df_out.info)
    # out=get_files(df_out)

    normlizetype = 'mean-std'
    datasets = {}
    datasets_train = SQ(ori_csv_pd,labels_dict, label_index,normlizetype, is_train=True,data_num='all')
    train_dataloader, val_dataloader = get_loaders(datasets_train, seed=5, batch=128)
    for id, (data, label) in enumerate(train_dataloader):
        print(id, data.shape, label)
    # datasets_test = SQ(df_out,label_index, normlizetype, is_train=False)
    # cls,_=datasets_train.get_classes_num()


