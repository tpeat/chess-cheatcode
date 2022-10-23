from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def make_samplers(dataset, 
                  validation_split=0.04,
                  test_split=.1,
                  shuffle_dataset = True,
                  random_seed= 20):
    ''' Makes samplers to split train and test data '''
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_ind = int(np.floor(validation_split * dataset_size))
    test_ind = val_ind+int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    val_indices = indices[:val_ind]
    test_indicies = indices[val_ind:test_ind]
    train_indices = indices[test_ind:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indicies)

    return train_sampler, val_sampler, test_sampler