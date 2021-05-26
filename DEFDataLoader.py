import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler
import multiprocessing

class DEFDataset(Dataset):
    def __init__(self, chromosomes, dset, zero_norms):
        ''''
        A single example in this dset is the entire set of model predictions/labels across a single chromosome
        '''
        self.ABCPreds = [[] for i in range(len(chromosomes))]
        self.labels = [[] for i in range(len(chromosomes))]
        self.zero_norms = zero_norms

        for i,chrom in enumerate(chromosomes):
            self.labels[i] = (dset[chrom][0])
            self.ABCPreds[i] = (dset[chrom][1])

    def __len__(self):
        return len(self.ABCPreds)

    #Returns the 4 channel example strings as a converted int lists, along with their label
    def __getitem__(self, idx):
        return (self.labels[idx], self.ABCPreds[idx], self.zero_norms[idx])

    def get_label(self, idx):
        return self.labels[idx]


def custom_collate_fn(batch):
    labels = [item[0] for item in batch]
    examples = [item[1] for item in batch]
    zero_norms = [item[2] for item in batch]

    return torch.Tensor(labels), torch.Tensor(examples), torch.Tensor(zero_norms)


def getLoaders(args, dset, zero_norms):
    chroms = [i for i in range(19) if i != args['model_chromosome']]    
    valid_size = 2

    data = DEFDataset(chroms, dset, zero_norms)
    data_test = DEFDataset([args['model_chromosome']], dset, zero_norms)

    # train and valid
    num_trainval_data = len(data)
    trainval_idx = list(range(num_trainval_data))

    np.random.shuffle(trainval_idx)

    train_idx = trainval_idx[valid_size:]
    valid_idx = trainval_idx[:valid_size]

    #test
    num_test_data = len(data_test)
    test_idx = list(range(num_test_data))

    train_sampler = BatchSampler(SubsetRandomSampler(train_idx), args['DEF_batch_size'], args['DEF_drop_last'])
    valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), args['DEF_batch_size'], args['DEF_drop_last'])
    test_sampler = BatchSampler(SequentialSampler(test_idx), args['DEF_batch_size'], args['DEF_drop_last'])

    train_loader = torch.utils.data.DataLoader(data, batch_sampler = train_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])
    valid_loader = torch.utils.data.DataLoader(data, batch_sampler = valid_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])
    test_loader = torch.utils.data.DataLoader(data_test, batch_sampler = test_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':  
    print("testing data loader")
    args = {'DEF_batch_size':1, 'env':'desktop', 'DEF_data_loader_workers':0, 'DEF_drop_last':False}
    
    train_loader, valid_loader, test_loader = getLoaders(1,args)

    print(len(valid_loader.batch_sampler.sampler))
    
    for label, example in train_loader:
        print(label)
