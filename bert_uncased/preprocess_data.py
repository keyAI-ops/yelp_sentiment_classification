import os
import pickle
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from transformers import (
    AutoTokenizer
)

def make_id_file(task, tokenizer):
    def make_data_strings(file_name):
        data_strings = []
        with open(os.path.join(file_name), 'r', encoding='utf-8') as f:
            id_file_data = [tokenizer.encode(line.lower()) for line in f.readlines()]  # id_file_data이라는 list에 [token1, token2, ...] 담긴다.
        for item in id_file_data:  
            data_strings.append(' '.join([str(k) for k in item]))  # data_strings 리스트에 띄어쓰기 단위로 string 저장
        return data_strings
    
    print('it will take some times...')
    train_pos = make_data_strings('./data/sentiment.train.1')
    train_neg = make_data_strings('./data/sentiment.train.0')
    dev_pos = make_data_strings('./data/sentiment.dev.1')
    dev_neg = make_data_strings('./data/sentiment.dev.0')

    print('make id file finished!')
    return train_pos, train_neg, dev_pos, dev_neg

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
result = make_id_file('yelp', tokenizer)
filenames = ['train_pos','train_neg', 'dev_pos', 'dev_neg']

for i, v in enumerate(result):
    filename = f"./data/{filenames[i]}"
    with open(filename, 'wb') as f:
        pickle.dump(v, f)
        