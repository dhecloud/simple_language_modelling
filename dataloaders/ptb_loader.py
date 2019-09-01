import collections
import os
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_binary_mask(input, deterministic=False):
    mask=torch.rand(input.shape)
    if deterministic:
        pass
    else:
        mask[mask>0.5]=1
        mask[mask<=0.5]=0
    return mask
    
def _read_words(filename):
    with open(filename,'r',encoding='utf-8') as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict((v,k) for k,v in word_to_id.items())
  return word_to_id, id_to_word
  
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id, _ = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary


def ptb_batcher(raw_data, seq_len=35, mask=False):
    '''
    Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    seq_len : sequence length

    Returns:
    A pair of Tensors, each shaped [num_batches, batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    '''
    raw_data = torch.tensor(raw_data, dtype=torch.int32)
    data_len = raw_data.shape[0]
    # number of batches
    num_batch = data_len // seq_len
    #number of batches in an epoch

    x = torch.zeros([num_batch, seq_len])
    y = torch.zeros([num_batch, seq_len])
    if not mask:
        for i in range(num_batch):
            x[i] = raw_data[i * seq_len:(i + 1) * seq_len]
            y[i] = raw_data[(i * seq_len)+1: (i + 1) * seq_len +1]
    else:
        for i in range(num_batch):
            y[i] = raw_data[i * seq_len:(i + 1) * seq_len]
            mask = generate_binary_mask(y[i])
            x[i] = torch.mul(y[i], mask)

    return x, y
    
def ptb_loader(x_batches, y_batchers, batch_size=20, shuffle=True):
    dataset = TensorDataset(x_batches, y_batchers)
    loader = DataLoader(dataset, batch_size, shuffle=shuffle,drop_last=True)
    return loader
    

if __name__ == '__main__':
    train_data, valid_data, test_data, vocabulary = ptb_raw_data('data')
    x, y= ptb_batcher(train_data, 20)
    loader = ptb_loader(x,y,20)
    for i, batch in enumerate(loader):
        print(i, batch[0], batch[1])
        break

    # print(raw_data)
    