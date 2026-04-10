import random

import numpy as np
from sklearn import utils
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from utils import reader, get_list_kmer, protein2num, neggen



def load_dataset(path, train_size=0.8, rand_neg=False, batch_size=32, num_cpu=4, device="cuda"):
    # Load the dataset from the given path
    # Split into training and testing sets based on train_size
    # If rand_neg is True, generate random negative samples
    # Return the training and testing datasets as PyTorch DataLoader objects
    manual_seed = torch.Generator().manual_seed(42)
    positive_samples = LoadEncoded(path, device=device)
    negative_samples = LoadEncoded(path, is_positive=False, fake=2, device=device)

    train_num = int(len(positive_samples) * train_size)
    val_num = int(len(positive_samples) * (1 - train_size)*0.5)
    split_size = [train_num, val_num, len(positive_samples) - train_num - val_num]
    train_pos, val_pos, test_pos = random_split(positive_samples, split_size, generator=manual_seed)
    train_neg, val_neg, test_neg = random_split(negative_samples, split_size, generator=manual_seed)

    if rand_neg:
        negative_data_rand = LoadEncoded(path, is_positive=False, fake=1, device=device)
        train_neg = ConcatDataset([train_neg, negative_data_rand])
    
    stack_datasets = [train_pos, val_pos, test_pos, train_neg, val_neg, test_neg]
    stack_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu) for dataset in stack_datasets]

    return stack_loaders


def load_data_test(path, batch_size=32, num_cpu=4, device="cuda"):
    dataset = LoadEncoded(path, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_cpu)

class LoadEncoded(Dataset):
    def __init__(self, pathpos, is_positive=True, device="cuda", fake=0, length_pro=81, divide=5, part=2):
        if is_positive and fake != 0:
            raise Exception("Cant use key word fake on positive dataset")
        self.device = device
        self.fake = fake
        self.length_pro = length_pro
        self.divide = divide
        self.part = part

        # get list of kmer
        dic = get_list_kmer(1)

        # read data from file
        self.dpos = reader(pathpos)

        # convert protein to number sequence
        self.npos = [protein2num(pro, dic) for pro in self.dpos]

        if is_positive:
            self.poslabel = torch.from_numpy(np.ones(len(self.dpos)))
        else:
            # ic("go false")
            self.poslabel = torch.from_numpy(np.zeros(len(self.dpos)))
        self.poslabel = self.poslabel.to(device)

    def __len__(self):
        return len(self.dpos)

    def __getitem__(self, idx):
        # convert data to one hot format and up to device
        pro = self.npos[idx]
        if len(pro) < self.length_pro:
            pro = pro + [0] * (self.length_pro - len(pro))
        elif len(pro) > self.length_pro:
            pro = pro[:self.length_pro]

        # random generate a fake promoter by shuffle the pro
        if self.fake == 1:
            pro = random.shuffle(pro)
        # random generate a fake promoter by replace part of pro
        elif self.fake == 2:
            pro = neggen(pro, num_part=self.divide, keep=self.part, max_class=4)

        torchpro = torch.from_numpy(np.array(pro))
        onehot = torch.nn.functional.one_hot(torchpro, num_classes=4).to(self.device)
        return onehot.float(), self.poslabel[idx]