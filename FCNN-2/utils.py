from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

loc_str = '/cluster/home/sagraw/GNSS/Data/Pixel4/karthik_NR/withDOP/not_normal/'

def get_data(batch_train, batch_test):
    train_data   = np.load(loc_str + 'trainx.npy')
    train_labels = np.load(loc_str + 'trainy.npy')
    valid_data   = np.load(loc_str + 'validx.npy')
    valid_labels = np.load(loc_str + 'validy.npy')
    test_data    = np.load(loc_str + 'testx.npy')
    test_labels  = np.load(loc_str + 'testy.npy')

    train_data    = train_data[:,[0,3]]
    valid_data    = valid_data[:,[0,3]]
    test_data     = test_data[:,[0,3]]
    train_labels   = train_labels[:,0]
    valid_labels   = valid_labels[:,0]
    test_labels    = test_labels[:,0]

    ## Train data:
    train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).float())
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_train)

    ## Test data
    test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).float())
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)

    ## Valid data
    valid_dataset = TensorDataset(Tensor(valid_data).float(), Tensor(valid_labels).float())
    validloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_test)

    return trainloader, train_dataset, testloader, test_dataset
