import random
import numpy as np
import sys
from visualize import visualize
from preprocess.augmentation import augment
import os

sys.path.append('dataset')

def batch_indices(batch_size=None, dataset_size=None):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    indices = list(zip(index_a, index_b))
    return indices

def train_generator(batch_size, is_augment=True):
    if is_augment:
        batch_size = int(batch_size / 2)

    # Load dataset
    directory = 'dataset/train/'
    train_x_full = np.load(os.path.join(directory, 'train_x.npy'))
    train_y_prob_full = np.load(os.path.join(directory, 'train_y_prob.npy'))
    train_y_keys_full = np.load(os.path.join(directory, 'train_y_keys.npy'))

    dataset_size = train_y_prob_full.shape[0]
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)
    print('Training Dataset Size:', dataset_size)

    while True:
        for index in indices:
            train_x = train_x_full[index[0]:index[1]]
            train_y_prob = train_y_prob_full[index[0]:index[1]]
            train_y_keys = train_y_keys_full[index[0]:index[1]]

            train_x, train_y_prob, train_y_keys = augment(train_x, train_y_prob, train_y_keys)

            train_x = train_x / 255.0
            train_y_prob = np.squeeze(train_y_prob)
            train_y_keys = train_y_keys / 128.0
            train_y_keys = np.repeat(train_y_keys[:, None], 10, axis=1)

            indices = list(range(train_x.shape[0]))
            random.shuffle(indices)

            train_x = train_x[indices]
            train_y_prob = train_y_prob[indices]
            train_y_keys = train_y_keys[indices]

            yield train_x, [train_y_prob, train_y_keys]

def valid_generator(batch_size):
    directory = 'dataset/valid/'

    valid_x_full = np.load(os.path.join(directory, 'valid_x.npy'))
    valid_y_prob_full = np.load(os.path.join(directory, 'valid_y_prob.npy'))
    valid_y_keys_full = np.load(os.path.join(directory, 'valid_y_keys.npy'))

    dataset_size = valid_y_prob_full.shape[0]
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)
    print('Validation Dataset Size:', dataset_size)

    while True:
        for index in indices:
            valid_x = valid_x_full[index[0]:index[1]]
            valid_y_prob = valid_y_prob_full[index[0]:index[1]]
            valid_y_keys = valid_y_keys_full[index[0]:index[1]]

            valid_x = valid_x / 255.0
            valid_y_prob = np.squeeze(valid_y_prob)
            valid_y_keys = valid_y_keys / 128.0
            valid_y_keys = np.repeat(valid_y_keys[:, None], 10, axis=1)

            yield valid_x, valid_y_prob, valid_y_keys

if __name__ == '__main__':
    gen = valid_generator(batch_size=32)
    for _ in range(5):  # Fetch 5 batches for demonstration
        x_batch, y_prob_batch, y_keys_batch = next(gen)

        print(x_batch.shape)
        print(y_prob_batch.shape)
        print(y_keys_batch.shape)
