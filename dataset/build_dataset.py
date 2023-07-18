from .CWRU import CWRU
from .SQ import SQ,SQ_Multi
from .SQV import SQV_Multi,SQV
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader,Subset,random_split
import torch
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from torch.utils.data.sampler import Sampler,RandomSampler,SequentialSampler

def check_dict(key,dict):
    return  True if key in dict else False

def create_labels_dict(**kwargs):
    """
    Creates a dictionary with label column names as keys and provided label values as values.
    *args: Specify the label column names as positional arguments.
    **kwargs: Key-value pairs where keys are label names and values are desired label values.
    """
    sq_index={'rpm','state','path','label'}
    labels_dict = {}

    # Add labels specified as key-value pairs
    for label_name, label_value in kwargs.items():
        if label_name in sq_index:
            labels_dict[label_name] = label_value

    return labels_dict

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

def get_loaders(train_dataset,
                val_ratio=0.2,
                batch_size=128,
                seed=0,
                with_test=True,
                **kwargs
                ):
    dataset_len = len(train_dataset)
    labels = train_dataset.labels
    ## label same, split random
    if len(set(labels))==1:
        total_size = len(train_dataset)
        train_size = int((1-val_ratio) * total_size)
        valid_size = total_size - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, valid_size])
        return train_dataset,val_dataset

    # 使用 StratifiedShuffleSplit 进行标签分层随机划分
    sss1 = StratifiedShuffleSplit(n_splits=1,
                                  test_size=val_ratio,
                                  random_state=seed)  # for splitting into training and the rest
    sss2 = StratifiedShuffleSplit(n_splits=1,
                                  test_size=0.5,
                                  random_state=seed)  # for splitting the rest into validation and testing

    train_indices, rest_indices = next(sss1.split(range(dataset_len), labels))
    if with_test:

        # Extract the labels of the rest part
        rest_labels = [labels[i] for i in rest_indices]

        # Split the rest part into validation and testing
        val_indices, test_indices = next(sss2.split(rest_indices, rest_labels))

        # Adjust the validation and testing indices to the original indices
        val_indices = [rest_indices[i] for i in val_indices]
        test_indices = [rest_indices[i] for i in test_indices]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        test_subset= Subset(train_dataset,test_indices)
        return train_subset,val_subset,test_subset

    else:
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, rest_indices)
        return train_subset,val_subset

def build_dataset(dataset_type,b=128, normlizetype = '1-1',**kwargs):
    # For CW dataset
    # k=[0-9]
    if dataset_type == 'CW':
        data_dir = "/home/lucian/Documents/datas/CW"
        normlizetype = '1-1'
        data_set = CWRU(data_dir, normlizetype, is_train=True, **kwargs)
        datasets = {'train': data_set}
        cw_data = [i[0] for i in datasets['train']]
        cw_np = np.array(cw_data)
        condition='_cw_ch'+str(data_set.ch)
        return datasets,cw_np,condition

    # For SQ dataset
    # **kwags is rpm, state
    elif dataset_type == 'SQ':
        ori_root = '/home/lucian/Documents/datas/Graduate_data/SQdata/dataframe.csv'
        ori_csv_pd = pd.read_csv(ori_root)
        labels_dict = create_labels_dict(**kwargs)
        label_index = 'state'

        datasets_train = SQ(ori_csv_pd, labels_dict, label_index, normlizetype, is_train=True, **kwargs)
        datasets = {'train': datasets_train}
        sq_data = []
        for data, _, in datasets['train']:
            sq_data.append(data)
        indices = np.random.choice(len(sq_data), size=b, replace=False)
        sq_np = np.array(sq_data)[indices]
        condition =  '_sq_rpm' + str(labels_dict['rpm']) + '_' + labels_dict['state']
        return datasets,sq_np,condition

    elif dataset_type == 'SQ_M':
        ori_root = '/home/lucian/Documents/datas/Graduate_data/SQdata/dataframe.csv'
        ori_csv_pd = pd.read_csv(ori_root)
        labels_dict = create_labels_dict(**kwargs)
        if check_dict('rpm',labels_dict):
            rpm=labels_dict.get('rpm')
        else:
            rpm='all_rpm'
        if check_dict('state',labels_dict):
            state=labels_dict.get('state')
        else:
            state = 'all_state'
        label_index = 'rpm'

        datasets_train = SQ_Multi(ori_csv_pd, labels_dict, label_index, normlizetype, is_train=True,**kwargs)
        datasets = {'train': datasets_train}
        sq_data = []
        sq_cond = []
        for data, _, cond in datasets['train']:
            sq_data.append(data)
            sq_cond.append(cond)
        indices = np.random.choice(len(sq_data), size=b, replace=False)
        sq_np = np.array(sq_data)[indices]
        sq_c=np.array(sq_cond)[indices]
        condition =  '_sq_rpm' + str(rpm) + '_' + str(state)
        return datasets,sq_np,condition,sq_c

    elif dataset_type == 'SQV':
        ori_root = '/home/lucian/Documents/datas/Graduate_data/SQV/dataframe.csv'
        ori_csv_pd = pd.read_csv(ori_root)
        labels_dict = create_labels_dict(**kwargs)
        label_index = 'state'

        datasets_train = SQV(ori_csv_pd, labels_dict, label_index, normlizetype, is_train=True, **kwargs)
        datasets = {'train': datasets_train}
        sq_data = []
        for data, _, in datasets['train']:
            sq_data.append(data)
        indices = np.random.choice(len(sq_data), size=b, replace=False)
        sq_np = np.array(sq_data)[indices]
        condition =  '_sq_' + labels_dict['state']
        return datasets,sq_np,condition

    elif dataset_type == 'SQV_M':
        ori_root = '/home/lucian/Documents/datas/Graduate_data/SQV/dataframe.csv'
        ori_csv_pd = pd.read_csv(ori_root)
        labels_dict = create_labels_dict(**kwargs)
        label_index = 'state'
        print('kwargs',kwargs)
        datasets_train = SQV_Multi(ori_csv_pd, labels_dict, label_index, normlizetype, is_train=True,**kwargs)
        datasets = {'train': datasets_train}
        sqv_data = []
        sqv_cond = []
        for data, _, cond in datasets['train']:
            sqv_data.append(data)
            sqv_cond.append(cond)
        indices = np.random.choice(len(sqv_data), size=b, replace=False)
        sqv_np = np.array(sqv_data)[indices]
        sqv_c = np.array(sqv_cond)[indices]
        condition = '_sqv_multi_' + labels_dict['state']
        return datasets,sqv_np,condition,sqv_c

    else:
        print("Invalid dataset_type. Choose 'CW','SQ' or 'SQV")

