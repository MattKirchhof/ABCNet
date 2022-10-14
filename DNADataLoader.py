import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler
import multiprocessing

class DNADataset(Dataset):

    def __init__(self, chromosomes, path=None):
        '''
        Accepts a list of chromosomes to load into its dataset. This is generally a list of 1 or more chromosome indices
        '''
        self.examples = []
        self.labels = []
        self.data, self.chrom_lengths, self.zero_norms = self.loadAll(chromosomes, path)

        for chrom in self.data:
            self.labels.extend(chrom[0])
            self.examples.extend(chrom[1])

    def __len__(self):
        return len(self.examples)

    #Returns the 4 channel example strings as a converted int lists, along with their label
    def __getitem__(self, idx):
        converted_example = self.convertChannel(self.examples[idx])
        zero_norm = self.get_zero_norm(idx)
        return [self.labels[idx], converted_example, zero_norm]

    def get_zero_norm(self, idx):
        group = 0
        total = 0
        zero_norm = 0.0
        while idx >= total and group < len(self.zero_norms):
            total += self.zero_norms[group][0]
            zero_norm = self.zero_norms[group][1]
            group += 1

        if idx > total:
            print("Get Zero Norm error: Index greater than Total... idx: ", idx, " total:", total)

        return zero_norm

    def convertChannel(self, example):
        channels = []
        for i in range (len(example)):
            channels.append( list( map( int, example[i])))

        return channels

    def get_label(self, idx):
        return self.labels[idx]

    def loadAll(self, chromosomes, path = None):
        substring = "rawPCANormalizedOneHot"
        dataset = []
        zero_norms = []
        chromLengths = []
        binwiseData = []
        binwiseLabels = []

        if path is not None:
            print("Loading from custom path:", path) 
            data_path = path
        else:
            data_path = './Data/'

        for chrom_index in chromosomes:
            print("Loading: " + str(chrom_index))
            with open(data_path + str(chrom_index+1) + substring + '.fa') as processedFile:
                details = processedFile.readline().split(',')

                # skip the first X unused bins
                #Read the rest into memory
                end_of_file = False
                num_examples = 0
                while end_of_file != True:

                    dataline = processedFile.readline()
                    if not dataline:
                        end_of_file = True
                    else: 
                        dataline = dataline.strip('\n').split(',')
                        if len(dataline) > 1:
                            num_examples += 1
                            # Set dataline[0] to be our expected output as a numerical float
                            label = float(dataline[0])

                            # converts the 4 channels of '010001' strings to lists of integers
                            channels = []
                            for channel in range(1, 5):
                                channels.append( dataline[channel])

                            # Append the resulting int converted training example to the dataset
                            binwiseLabels.append(label)
                            binwiseData.append(channels)

                # Append the A/B classification cutoff to the list of zero_norms along with how many elements belong to that chrom
                zero_norms.append([num_examples, float(details[-1].strip('\n'))])
                print([num_examples, float(details[-1].strip('\n'))])

            dataset.append([binwiseLabels, binwiseData])
            chromLengths.append(len(binwiseLabels))
            binwiseData = []
            binwiseLabels = []

        return dataset, chromLengths, zero_norms


def custom_collate_fn(batch):
    labels = [[item[0]] for item in batch]
    examples = [item[1] for item in batch]
    zero_norms = [item[2] for item in batch]
    return torch.Tensor(labels), torch.Tensor(examples), torch.Tensor(zero_norms)


def getLoaders(withheld_test_chrom, args,file_path, chrom_limiter=19, shuffle_data = True, valid_size = 0.1):
    chroms = [i for i in range(chrom_limiter) if i != withheld_test_chrom]

    data = DNADataset(chroms, file_path + args['data_path'])
    data_test = DNADataset([withheld_test_chrom], file_path + args['data_path'])

    # train and valid
    num_trainval_data = len(data)
    trainval_idx = list(range(num_trainval_data))

    if shuffle_data:
        np.random.shuffle(trainval_idx)

    split_tv = int(np.floor(valid_size * len(trainval_idx)))
    train_idx = trainval_idx[split_tv:]
    valid_idx = trainval_idx[:split_tv]

    #test
    num_test_data = len(data_test)
    test_idx = list(range(num_test_data))

    if shuffle_data:
        train_sampler = BatchSampler(SubsetRandomSampler(train_idx), args['batch_size'], args['drop_last'])
        valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), args['batch_size'], args['drop_last'])
    else:
        train_sampler = BatchSampler(SequentialSampler(train_idx), args['batch_size'], args['drop_last'])
        valid_sampler = BatchSampler(SequentialSampler(valid_idx), args['batch_size'], args['drop_last'])
    test_sampler = BatchSampler(SequentialSampler(test_idx), args['batch_size'], args['drop_last'])

    print("dataloaders given " + str(args["data_loader_workers"]) + " workers..")
    train_loader = torch.utils.data.DataLoader(data, batch_sampler = train_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])
    valid_loader = torch.utils.data.DataLoader(data, batch_sampler = valid_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])
    test_loader = torch.utils.data.DataLoader(data_test, batch_sampler = test_sampler, collate_fn=custom_collate_fn, num_workers=args["data_loader_workers"])

    print("Dataloaders returned..")
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':  
    print("testing data loader")
    args = {'batch_size':10, 'env':'mist', 'data_loader_workers':4, 'drop_last':True}
    
    train_loader, valid_loader, test_loader = getLoaders(1,args)
    print(len(valid_loader.batch_sampler.sampler))
    
    for label, example in train_loader:
        print(label)
