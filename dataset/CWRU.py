import os

import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from utils.sequence_transform import *

signal_size = 1024
from collections import Counter


datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
dataname5 = ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
             "315.mat"]  # 1797rpm
dataname6 = ["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
             "316.mat"]  # 1772rpm
dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname9 = ["109.mat", "122.mat", "135.mat", "174.mat", "189.mat", "201.mat", "213.mat", "250.mat",
             "262.mat"]  # 1797rpm
dataname10 = ["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "251.mat",
              "263.mat"]  # 1772rpm
dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "252.mat",
              "264.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "253.mat",
              "265.mat"]  # 1730rpm
# label
label_all = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False,ch=None,**kwargs):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    if ch is None:
        data_root1 = os.path.join('/tmp', root, datasetname[3])
        data_root2 = os.path.join('/tmp', root, datasetname[0])

        path1 = os.path.join('/tmp', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
        data, lab = data_load(path1, axisname=normalname[0],label=0,**kwargs)  # nThe label for normal data is 0

        for i in tqdm(range(len(dataname1))):
            path2 = os.path.join('/tmp', data_root2, dataname1[i])

            data1, lab1 = data_load(path2, dataname1[i], label=label_all[i])
            data += data1
            lab += lab1
        return [data, lab]
    else:
        data_root1 = os.path.join('/tmp', root, datasetname[3])
        data_root2 = os.path.join('/tmp', root, datasetname[0])
        if ch==0:
            path1 = os.path.join('/tmp', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
            data, lab = data_load(path1, axisname=normalname[0], label=0)  # nThe label for normal data is 0
            return [data, lab]
        elif ch>0 and ch<=9:
            print('ch label is',ch)
            path2 = os.path.join('/tmp', data_root2, dataname1[ch-1])

            data, lab = data_load(path2, dataname1[ch-1], label=label_all[ch-1])

            return [data, lab]
        else :
            raise('ch is not exist')


def data_load(filename, axisname, label,length=1024,data_num=20):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    if data_num == 'all':
        print('all_data_num for one csv is', int(fl.shape[0] // length))
        data = [fl[start:end].reshape(1, length) for start, end in zip(range(0, fl.shape[0], length),
                                                                       range(length, fl.shape[0] + 1, length))]
    elif data_num < fl.shape[0] // length:
        data = np.split(fl[:length * data_num, :].reshape(-1, length), data_num, axis=0)
    else:
        raise ('data length choose wrong')

    lab = [label] * len(data)

    return data, lab


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

class CWRU(Dataset):

    def __init__(self, data_dir, normlizetype, is_train=True,ch=None,**kwargs):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.ch=self._get_ch(ch)

        list_data = get_files(self.data_dir, self.is_train,ch=ch,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.transform = None
        self.cls_num = set(list_data[1])

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
        return len(self.cls_num),self.cls_num# num, name

    def _get_ch(self,ch):
        if ch is not None:
            return ch
        else :
            raise('CW data ch is None')

def counter_dataloader(data_loader):

    label_counts = Counter()
    for batch in data_loader:
        _, labels = batch
        for label in labels:
            label_counts[label.item()] += 1  # 假设标签是一个张量，需要使用.item()将其转换为普通的Python数据类型
    print('dataloader counter',label_counts)

def get_loaders(train_dataset, seed, batch, val_ratio=0.2):
    dataset_len = len(train_dataset)
    labels = train_dataset.labels

    # 使用 StratifiedShuffleSplit 进行标签分层随机划分
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_indices, val_indices = next(sss.split(range(dataset_len), labels))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)
    counter_dataloader(val_dataloader)
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    data_dir="/home/lucian/Documents/datas/CW"
    normlizetype='mean-std'
    datasets={}
    datasets_train = CWRU(data_dir, normlizetype,is_train=True,ch=1,length=1024,data_num=20)
    ch=datasets_train.ch
    train_dataloader,val_dataloader=get_loaders(datasets_train,seed=5, batch=16)
    for id, (data,label) in enumerate(train_dataloader):
        print(id,len(data),label)
    datasets['test'] = CWRU(data_dir, normlizetype, is_train=False,ch=1)