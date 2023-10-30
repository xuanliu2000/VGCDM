from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from utils.sequence_transform import *


def plot_label_counts(label_counter):
    labels = label_counter.keys()
    counts = label_counter.values()

    fig, ax = plt.subplots()
    ax.bar(labels, counts)

    ax.set_title('Label Counts')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')

    plt.xticks(rotation=45)

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
    *args: Specify the label column names as positional arguments.
    **kwargs: Key-value pairs where keys are label names and values are desired label values.
    """
    sq_index={'rpm','state','path'}
    labels_dict = {}

    # Add labels specified as key-value pairs
    for label_name, label_value in kwargs.items():
        if label_name in sq_index:
            labels_dict[label_name] = label_value

    return labels_dict

def get_files(dir,label='state',is_train=True,mutli=None,plot_counter=None,**kwargs):
    '''

    :param dir: csv direction, pd.dataframe
    :return:
    '''
    if 'length' in kwargs:
        length=kwargs['length']
    else:
        length=1024
    if 'data_num' in kwargs:
        data_num=kwargs['data_num']
    else:
        data_num='all'

    if mutli is None:
        csv_list=dir['path'].tolist()
        csv_label=dir[label].tolist()
        if plot_counter is True:
            element_counts = Counter(csv_label)
            plot_label_counts(element_counts)
            print('label_counter',element_counts)
        data_dict ,label_dict= data_load(csv_list,labels=csv_label,**kwargs)

        return [data_dict ,label_dict]
    else:
        csv_list = dir['path'].tolist()
        assist_list=[i.replace('ch2','ch3') for i in csv_list]
        # print(csv_list)
        # print(assist_list)
        csv_label = dir[label].tolist()
        data_dict, label_dict = data_load(csv_list, labels=csv_label, length=length,data_num=data_num)
        assist_dict, _ = data_load(assist_list, labels=csv_label, length=length,data_num=data_num)
        return [data_dict, label_dict,assist_dict]

def data_load(dir,labels,length=1024,data_num=20):
    assert len(dir)==len(labels)
    data_all=[]
    label_all=[]
    # data_dict={}
    for i in range(len(dir)):
        data_df = pd.read_table(dir[i], sep='\t', skiprows=range(1, 17))
        label_df = labels[i]

        fl = data_df.iloc[:, 1:].values
        # 划分数据
        if data_num=='all':
            print('all_data_num for one csv is', int(fl.shape[0]//length))
            data = [fl[start:end].reshape(1, length) for start, end in zip(range(0, fl.shape[0], length),
                                                                    range(length, fl.shape[0] + 1,length))]
        elif data_num< fl.shape[0]//length:
             data = np.split(fl[:length * data_num, :].reshape(-1, length), data_num, axis=0)
        else:
            raise ('data length choose wrong')
        # print(len(data))
        for j in data:
            data_all.append(j)
            label_all.append(label_df)
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

    def __init__(self,
                 dir,
                 labels_dict,
                 label_index,
                 normlizetype,
                 is_train=True,
                 dict=None,
                 **kwargs):

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

class SQ_Mutli(Dataset):

    def __init__(self,
                 dir,
                 labels_dict,
                 label_index,
                 normlizetype,
                 is_train=True,
                 dict=None,
                 use_label=None,
                 **kwargs):

        self.dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.label_index=label_index

        self.data_dir = filter_df_by_labels(self.dir, labels_dict)
        list_data= get_files(self.data_dir, label=self.label_index, is_train=self.is_train,mutli=True,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1],'assist':list_data[2]})
        self.cls_num=set(list_data[1])
        if use_label:
            self.labels = LabelEncoder().fit_transform(list(self.data_pd['label']))
        else:
            self.labels = self.data_pd['label'].tolist()
        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.seq_assist = self.data_pd['assist'].tolist()
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.seq_assist = self.data_pd['assist'].tolist()
            self.transform = None

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
        return len(self.cls_num),self.cls_num # num, name

def split_dataset(datasets_train, target_label=None,isprint=False):
    if target_label is None:
        return datasets_train,datasets_train

    positive_indices = [index for index, label in enumerate(datasets_train.labels) if label == target_label]
    positive_subset = Subset(datasets_train, positive_indices)

    # 根据指定标签筛选出不包含标签的子集
    negative_indices = [index for index, label in enumerate(datasets_train.labels) if label != target_label]
    negative_subset = Subset(datasets_train, negative_indices)

    # 输出子集的长度
    if isprint:
        neg = DataLoader(negative_subset, batch_size=128)
        for i, batch in enumerate(neg):
            print(batch[1])

        pos = DataLoader(positive_subset, batch_size=128)
        for i, batch in enumerate(pos):
            print(batch[1])

    return negative_subset, positive_subset

if __name__ == '__main__':
    ori_root = '/home/lucian/Documents/datas/Graduate_data/SQdata/dataframe.csv'
    ori_csv_pd = pd.read_csv(ori_root)
    # print(df.info)
    labels_dict = create_labels_dict(state='inner3',datanum=100)
    out=filter_df_by_labels(ori_csv_pd, labels_dict)
    label_index='rpm'

    normlizetype = 'mean-std'
    datasets = {}
    datasets_train = SQ_Mutli(ori_csv_pd, labels_dict, label_index, normlizetype, is_train=True, data_num=20, use_label=True)

    target_label = 29
    tra,cal=split_dataset(datasets_train,target_label,isprint=True)
