from .CWRU import CWRU
from .SQ import SQ,SQ_Multi,create_labels_dict
import numpy as np
import pandas as pd
import os

def check_dict(key,dict):
    return  True if key in dict else False


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
        SQ_data = [i[0] for i in datasets['train']]
        indices = np.random.choice(len(SQ_data), size=b, replace=False)
        SQ_np = np.array(SQ_data)[indices]
        condition =  '_sq_rpm' + str(labels_dict['rpm']) + '_' + labels_dict['state']
        return datasets,SQ_np,condition
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


        label_index = 'state'

        datasets_train = SQ_Multi(ori_csv_pd, labels_dict, label_index, normlizetype, is_train=True,**kwargs)
        datasets = {'train': datasets_train}
        SQ_data = [i[0] for i in datasets['train']]
        SQ_cond = [i[2] for i in datasets['train']]
        indices = np.random.choice(len(SQ_data), size=b, replace=False)
        SQ_np = np.array(SQ_data)[indices]
        SQ_c=np.array(SQ_cond)[indices]
        condition =  '_sq_rpm' + str(rpm) + '_' + str(state)
        return datasets,SQ_np,condition,SQ_c

    else:
        print("Invalid dataset_type. Choose 'CW' or 'SQ'")

