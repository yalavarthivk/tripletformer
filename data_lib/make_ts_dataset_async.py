import os

import sktime
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import pdb
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.data_processing import from_nested_to_3d_numpy

from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

label_encoder = LabelEncoder()

## Download the dataset and keep in the folder with dataset name

DATA_PATH = "~/tripletformer/data_lib/"
Dataset = "PhonemeSpectra"

train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, Dataset+"/"+Dataset+"_TRAIN.ts")
)
test_x, test_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, Dataset+"/"+Dataset+"_TEST.ts")
)

train_ind = train_x.shape[0]
train_dim = train_x.shape[1]
test_ind = test_x.shape[0]
test_dim = test_x.shape[1]

train_x = from_nested_to_2d_array(train_x).values


train_val = int(train_x.shape[1]/train_dim)

train_x = np.reshape(train_x,(train_ind, train_dim, train_val))
train_x = np.transpose(train_x,(0,2,1))
train_y = label_encoder.fit_transform(train_y)

test_x = from_nested_to_2d_array(test_x).values

test_val = int(test_x.shape[1]/test_dim)

test_x = np.reshape(test_x,(test_ind, test_dim, test_val))
test_x = np.transpose(test_x,(0,2,1))
test_y = label_encoder.fit_transform(test_y)



train_mask = np.zeros_like(train_x)
test_mask = np.zeros_like(test_x)

train_x, val_x, train_mask, val_mask = train_test_split(train_x, train_mask, test_size=0.2, random_state=42)
np.random.seed(0)

n_chans = train_x.shape[-1]
for i in range(train_x.shape[0]):
	for j in range(train_x.shape[1]):
		idx = np.random.randint(n_chans)
		train_mask[i,j,idx] = 1.

for i in range(val_x.shape[0]):
	for j in range(val_x.shape[1]):
		idx = np.random.randint(n_chans)
		val_mask[i,j,idx] = 1.

for i in range(test_x.shape[0]):
	for j in range(test_x.shape[1]):
		idx = np.random.randint(n_chans)
		test_mask[i,j,idx] = 1.

train_x *= train_mask
test_x *= test_mask
val_x *= val_mask


input_dim = train_x.shape[-1]

observed_vals, observed_mask, observed_tp = (
    train_x,
    train_mask,
    np.ones_like(train_x[:,:,0]).cumsum(-1),
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
    hth.append(th2)
    lth.append(th1)
    temp = []
    for val in var_dict[i]:
        if val <= th2 and val >= th1:
            temp.append(val)
    if len(np.unique(temp)) > 10:
        data_mean.append(np.mean(temp))
        data_std.append(np.std(temp)+1e-8)
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

total_data = []
for data in train_dataset:
    data1 = data[~np.all(data == 0, axis=1)]
    data1 = np.pad(data1, ((0, train_dataset.shape[1] - data1.shape[0]), (0, 0)), 'constant', constant_values=0)
    total_data.append(data1[None,:,:])
train_dataset = np.concatenate(total_data, 0)
asd = train_dataset[:,:,input_dim:input_dim*2].sum(-1).astype(bool)
bsd = np.where(asd.sum(-1)>5)[0]
train_dataset = train_dataset[bsd]

## Validation dataset

observed_vals, observed_mask, observed_tp = (
    val_x,
    val_mask,
    np.ones_like(val_x[:,:,0]).cumsum(-1)
)

print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

if np.max(observed_tp) != 1.0:
    observed_tp = observed_tp / np.max(observed_tp)

var_dict = {}
hth = []
lth = []
for i in range(input_dim):
    var_dict[i] = []

# normalizing
observed_vals[observed_mask == 0] = 0
observed_vals = (observed_vals - data_mean) / data_std
observed_vals[observed_mask == 0] = 0
mask = observed_mask.sum(-1)
observed_tp[mask==0] = 0

val_data = np.concatenate(
    (observed_vals, observed_mask, observed_tp[:, :, None]), -1)

total_data = []
for data in val_data:
    data1 = data[~np.all(data == 0, axis=1)]
    data1 = np.pad(data1, ((0, val_data.shape[1] - data1.shape[0]), (0, 0)), 'constant', constant_values=0)
    total_data.append(data1[None,:,:])
val_data = np.concatenate(total_data, 0)
asd = val_data[:,:,input_dim:input_dim*2].sum(-1).astype(bool)
bsd = np.where(asd.sum(-1)>5)[0]
val_data = val_data[bsd]


## Test data

observed_vals, observed_mask, observed_tp = (
    test_x,
    test_mask,
    np.ones_like(test_x[:,:,0]).cumsum(-1),
)



print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

if np.max(observed_tp) != 1.0:
    observed_tp = observed_tp / np.max(observed_tp)

# normalizing
observed_vals[observed_mask == 0] = 0
observed_vals = (observed_vals - data_mean) / data_std
observed_vals[observed_mask == 0] = 0
mask = observed_mask.sum(-1)
observed_tp[mask==0] = 0

test_data = np.concatenate(
    (observed_vals, observed_mask, observed_tp[:, :, None]), -1)

total_data = []
for data in test_data:
    data1 = data[~np.all(data == 0, axis=1)]
    data1 = np.pad(data1, ((0, test_data.shape[1] - data1.shape[0]), (0, 0)), 'constant', constant_values=0)
    total_data.append(data1[None,:,:])
test_data = np.concatenate(total_data, 0)
asd = test_data[:,:,input_dim:input_dim*2].sum(-1).astype(bool)
bsd = np.where(asd.sum(-1)>5)[0]
test_data = test_data[bsd]

np.savez('~/data+lib/PhonemeSpectra.npz', train=train_dataset, val=val_data, test=test_data)