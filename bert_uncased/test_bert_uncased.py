from dataclasses import dataclass, field
import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import pandas as pd
from dataset import SentimentTestDataset, collate_fn_style_test

def make_id_file_test(tokenizer, test_dataset):
    data_strings = []
    id_file_data = [tokenizer.encode(sent.lower()) for sent in test_dataset]
    for item in id_file_data:  
        data_strings.append(' '.join([str(k) for k in item]))  # data_strings 리스트에 띄어쓰기 단위로 string 저장
    return data_strings

def test():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    test_df = pd.read_csv('test_project/data/test_no_label.csv')
    test_dataset = test_df['Id']
    
    test = make_id_file_test(tokenizer, test_dataset)
    
    test_dataset = SentimentTestDataset(tokenizer, test)
    
    test_batch_size = 32
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=test_batch_size,
                                           shuffle=False, collate_fn=collate_fn_style_test, 
                                           num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    
    model.load_state_dict(torch.load('pytorch_model.bin'))
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        predictions = []
        for input_ids, attention_mask, token_type_ids, position_ids in tqdm(test_loader,
                                                                            desc='Test',
                                                                            position=1,
                                                                            leave=None):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)

            output = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids)

            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            predictions += batch_predictions
    test_df['Category'] = predictions
    test_df.to_csv('submisson.csv', index = False)
    
if __name__ == "__main__":
    test()