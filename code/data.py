import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import h5py
import pandas as pd

def split(train_dataset, frac):
    '''
    Split training and validation/test set
    '''
    train_size = int(len(train_dataset) * frac)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(118))
    return train_subset, val_subset

label_dict = {
    'Negative for Pneumonia':0,
    'Typical Appearance':1,
    'Indeterminate Appearance':2,
    'Atypical Appearance':3
}

class ImageDataset(Dataset):
    def __init__(self, img_path, label_path, transforms=None):
        self.images = h5py.File(img_path, 'r')
        self.labels = pd.read_csv(label_path).iloc[:, 1:]
        self.key = list(self.images.keys())
        self.transforms = transforms
        
    def __getitem__(self, index):
        key = self.key[index]
        img = torch.Tensor(np.asarray(self.images[key])[np.newaxis, ...])
        label_row = self.labels[self.labels.id == key + '_study']
        label = label_dict[label_row.columns[(label_row==1).values.flatten().tolist()][0]]
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, label)

    def __len__(self):
        return len(self.images)


def get_dataset(data_path, label_path, args=None):
    return ImageDataset(data_path, label_path, args)


def create_dataloaders(train_set, val_set, test_set, args=None):
    if args == None:
        return DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
    batch_size = args['bz']
    shuffle_data = args['shuffle_data']
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_data, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataloader, val_dataloader, test_dataloader


def get_dataloaders(dataset_path, label_path, args=None):
    # TODO data augmentation
    # transform = transforms.Compose([...])

    dataset = get_dataset(dataset_path, label_path)

    train_set, test_set = split(dataset, 0.8)
    train_set, val_set = split(train_set, 0.8)

    dataloaders = create_dataloaders(train_set, val_set, test_set, args)
    return dataloaders

def write_to_file(path, data):
    """
    Dumps pickled data into the specified relative path.

    Args:
        path: relative path to store to
        data: data to pickle and store
    """
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)