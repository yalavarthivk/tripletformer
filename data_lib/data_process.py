import numpy as np
from sklearn import model_selection
import pdb

def get_dataset_2(dataset):
    x = np.load('~/data_lib/physionet_compressed.npz')
    train_data, test_data = x['train'], x['test']
    input_dim = (train_data.shape[-1]-1)//2
    print(input_dim)
    x = []
    train_data, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=42, shuffle=True
    )


    observed_vals, observed_mask, observed_tp = (
        train_data[:, :, :input_dim],
        train_data[:, :, input_dim: 2 * input_dim],
        train_data[:, :, -1],
    )

    
    print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

    if np.max(observed_tp) != 1.0:
        observed_tp = observed_tp / np.max(observed_tp)

    data_mean, data_std = [], []
    var_dict = {}
    hth = []
    lth = []
    for i in range(input_dim):
        var_dict[i] = []
    for i in range(observed_vals.shape[0]):
        for j in range(input_dim):
            indices = np.where(observed_mask[i, :, j] > 0)[0]
            var_dict[j] += observed_vals[i, indices, j].tolist()

    for i in range(input_dim):
        th1 = np.quantile(var_dict[i], 0.001)
        th2 = np.quantile(var_dict[i], 0.9995)
        observed_vals[observed_mask==0] = np.nan

        observed_mask[:,:,i][observed_vals[:,:,i] > th2] = 0
        observed_vals[:,:,i][observed_vals[:,:,i] > th2] = np.nan
        

        observed_mask[:,:,i][observed_vals[:,:,i] < th1] = 0
        observed_vals[:,:,i][observed_vals[:,:,i] < th1] = np.nan
        
        hth.append(th2)
        lth.append(th1)
        temp = []
        for val in var_dict[i]:
            if val <= th2 and val >= th1:
                temp.append(val)
        if len(np.unique(temp)) > 10:
            data_mean.append(np.mean(temp))
            data_std.append(np.std(temp))
        else:
            data_mean.append(0)
            data_std.append(1)

    # normalizing
    observed_vals[observed_mask == 0] = 0

    observed_vals = (observed_vals - data_mean) / data_std
    observed_vals[observed_mask == 0] = 0
    mask = observed_mask.sum(-1)
    observed_tp[mask==0] = 0
    # pdb.set_trace()


    train_dataset = np.concatenate(
        (observed_vals, observed_mask, observed_tp[:, :, None]), -1)
    observed_vals, observed_mask, observed_tp = [], [], []
    total_data = []
    for data in train_dataset:
        data1 = data[~np.all(data == 0, axis=1)]
        data1 = np.pad(data1, ((0, train_dataset.shape[1] - data1.shape[0]), (0, 0)), 'constant', constant_values=0)
        total_data.append(data1[None,:,:])
    train_dataset = np.concatenate(total_data, 0)
    asd = train_dataset[:,:,input_dim:input_dim*2].sum(-1).astype(bool)
    bsd = np.where(asd.sum(-1)>5)[0]
    # train_dataset = train_dataset[bsd]
    total_data = []

    ## Validation dataset
    
    observed_vals, observed_mask, observed_tp = (
        val_data[:, :, :input_dim],
        val_data[:, :, input_dim: 2 * input_dim],
        val_data[:, :, -1],
    )


    
    print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

    if np.max(observed_tp) != 1.0:
        observed_tp = observed_tp / np.max(observed_tp)

    for i in range(input_dim):
        observed_mask[:,:,i][observed_vals[:,:,i] > hth[i]] = 0
        observed_vals[:,:,i][observed_vals[:,:,i] > hth[i]] = np.nan
        

        observed_mask[:,:,i][observed_vals[:,:,i] < lth[i]] = 0
        observed_vals[:,:,i][observed_vals[:,:,i] < lth[i]] = np.nan


    # normalizing
    observed_vals[observed_mask == 0] = 0
    observed_vals = (observed_vals - data_mean) / data_std
    observed_vals[observed_mask == 0] = 0
    mask = observed_mask.sum(-1)
    observed_tp[mask==0] = 0

    val_data = np.concatenate(
        (observed_vals, observed_mask, observed_tp[:, :, None]), -1)
    observed_vals, observed_mask, observed_tp = [], [], []
    total_data = []
    for data in val_data:
        data1 = data[~np.all(data == 0, axis=1)]
        data1 = np.pad(data1, ((0, val_data.shape[1] - data1.shape[0]), (0, 0)), 'constant', constant_values=0)
        total_data.append(data1[None,:,:])
    val_data = np.concatenate(total_data, 0)
    asd = val_data[:,:,input_dim:input_dim*2].sum(-1).astype(bool)
    bsd = np.where(asd.sum(-1)>5)[0]
    # val_data = val_data[bsd]


    total_data = []

    ## Test data

    observed_vals, observed_mask, observed_tp = (
        test_data[:, :, :input_dim],
        test_data[:, :, input_dim: 2 * input_dim],
        test_data[:, :, -1],
    )


    
    print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

    if np.max(observed_tp) != 1.0:
        observed_tp = observed_tp / np.max(observed_tp)

    for i in range(input_dim):

        observed_mask[:,:,i][observed_vals[:,:,i] > hth[i]] = 0
        observed_vals[:,:,i][observed_vals[:,:,i] > hth[i]] = np.nan
        

        observed_mask[:,:,i][observed_vals[:,:,i] < lth[i]] = 0
        observed_vals[:,:,i][observed_vals[:,:,i] < lth[i]] = np.nan


    # normalizing
    observed_vals[observed_mask == 0] = 0
    observed_vals = (observed_vals - data_mean) / data_std
    observed_vals[observed_mask == 0] = 0
    mask = observed_mask.sum(-1)
    observed_tp[mask==0] = 0
    # pdb.set_trace()

    
    test_data = np.concatenate(
        (observed_vals, observed_mask, observed_tp[:, :, None]), -1)
    observed_vals, observed_mask, observed_tp = [], [], []
    total_data = []
    for data in test_data:
        data1 = data[~np.all(data == 0, axis=1)]
        data1 = np.pad(data1, ((0, test_data.shape[1] - data1.shape[0]), (0, 0)), 'constant', constant_values=0)
        total_data.append(data1[None,:,:])
    test_data = np.concatenate(total_data, 0)
    asd = test_data[:,:,input_dim:input_dim*2].sum(-1).astype(bool)
    bsd = np.where(asd.sum(-1)>5)[0]

    total_data = []
    print(train_dataset.shape, val_data.shape, test_data.shape)
    pdb.set_trace()

    np.savez('~/data_lib/physionet.npz', train=train_dataset, val=val_data, test=test_data)

if __name__ == "__main__":
    get_dataset_2(1)