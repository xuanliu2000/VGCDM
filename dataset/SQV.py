from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset, random_split

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
    sq_index={'state','path'}
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
        csv_list = dir['path'].tolist() # get vibration signals
        assist_list=[i.replace('ch2','ch3') for i in csv_list] ## get current signals
        csv_label = dir[label].tolist()
        data_dict, label_dict = data_load(csv_list, labels=csv_label, length=length,data_num=data_num)
        assist_dict, _ = data_load(assist_list, labels=csv_label, length=length,data_num=data_num)
        return [data_dict, label_dict,assist_dict]

def data_load(dir,labels,length=1024,data_num=20):
    assert len(dir)==len(labels)
    data_all = []
    label_all = []

    for i, filename in enumerate(dir):
        data_df = pd.read_table(filename, sep='\t', skiprows=range(1, 17))
        label_df = labels[i]

        fl = data_df.iloc[:, 1:].values
        alldata = [fl[start:end].reshape(1, length) for start, end in zip(range(0, fl.shape[0], length),
                                                                          range(length, fl.shape[0] + 1, length))]
        total_num = fl.shape[0] // length

        if data_num == 'all':
            print('all_data_num for one csv is', total_num)
            data = alldata
        elif data_num <= total_num:
            data = random.sample(alldata, data_num)
        else:
            raise ValueError('data_num is over total_num')

        data_all.extend(data)
        label_all.extend([label_df] * len(data))

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

class SQV(Dataset):

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
        self.cls_no = convert_labels_to_numbers(self.cls_num)

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

class SQV_Multi(Dataset):

    def __init__(self,
                 dir,
                 labels_dict,
                 label_index,
                 normlizetype,
                 is_train=True,
                 dict=None,**kwargs):

        self.dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.label_index=label_index

        self.data_dir = filter_df_by_labels(self.dir, labels_dict)
        list_data= get_files(self.data_dir, label=self.label_index, is_train=self.is_train,mutli=True,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1],'assist':list_data[2]})
        self.cls_num=set(list_data[1])

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.seq_assist = self.data_pd['assist'].tolist()
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
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


if __name__ == '__main__':
    ori_root = '/home/lucian/Documents/datas/Graduate_data/SQV/dataframe.csv'
    ori_csv_pd = pd.read_csv(ori_root)
    # print(df.info)
    labels_dict = create_labels_dict(state='OF_3')
    out=filter_df_by_labels(ori_csv_pd, labels_dict)
    label_index='state'

    normlizetype = 'mean-std'
    datasets_train = SQV_Multi(ori_csv_pd,labels_dict, label_index,normlizetype, is_train=True,data_num=50,length=2048)
    datasets = {'train': datasets_train}
    train_dataset,val_dataset=get_loaders(datasets['train'],batch=128)

    # target_label = 29
    # tra,cal=split_dataset(datasets_train,target_label,isprint=True)
