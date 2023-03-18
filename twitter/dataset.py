import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class SentimentDataset(object):
    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]    # split() 하게되면 리턴값이 list
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]  
            self.label += [[0]]
# ss
    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])
    
def collate_fn_style(samples):  # 여기서 sample로 train_dataset, dev_dataset을 받음
    
    input_ids, labels = zip(*samples)  # np array 형태
    max_len = max(len(input_id) for input_id in input_ids)    # for문을 통해 input_ids를 돌면서 가장 긴 리스트의 길이를 찾음
    # sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]   # 길이 순으로 리스트의 인덱스를 오름차순 정렬 -> [::-1]이라 역순
    sorted_indices = range(len(input_ids))
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)  # 가장 긴 길이부터 sort 해서 연산 속도를 높이기 위함
    
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])
    labels = torch.tensor(np.stack(labels, axis=0)[sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids, labels

class SentimentTestDataset(object):
    def __init__(self, tokenizer, test):
        self.tokenizer = tokenizer
        self.data = []

        for sent in test:
            self.data += [self._cast_to_int(sent.strip().split())]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample)
    
def collate_fn_style_test(samples):
    input_ids = samples
    
    max_len = max(len(input_id) for input_id in input_ids)


    sorted_indices = range(len(input_ids))
    ''' input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                              batch_first=True)      # batch_first=True      아웃풋이  B x T x * 형태로 나옴 / 이렇게 하면 안됨.'''
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids],
                             batch_first=True)

    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids