import pickle
from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,  # 
    AdamW
)

import pickle
import argparse

from dataset import SentimentDataset, collate_fn_style

def train(args):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    filenames = ['train_pos','train_neg', 'dev_pos', 'dev_neg']
    temp = []
    for filename in filenames:
        with open(f"test_project/data/{filename}", 'rb') as f:
            temp.append(pickle.load(f))
    
    train_pos, train_neg, dev_pos, dev_neg = temp[0], temp[1], temp[2], temp[3]

    train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
    dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)
    
    train_batch_size = args.train_bs
    eval_batch_size = args.eval_bs
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True, collate_fn=collate_fn_style,
                                           pin_memory=True, num_workers=2)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=eval_batch_size,
                                         shuffle=False, collate_fn=collate_fn_style,
                                         num_workers=2)
    
    # random seed
    random_seed=42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(device)
    
    model.train()
    learning_rate = 5e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    def compute_acc(predictions, target_labels):
        return (np.array(predictions) == np.array(target_labels)).mean()
        
    train_epoch = 2
    lowest_valid_loss = 9999.
    for epoch in range(train_epoch):
        with tqdm(train_loader, unit="batch") as tepoch:
            for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                position_ids = position_ids.to(device)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()

                output = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            labels=labels)

                loss = output.loss
                loss.backward()

                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                if iteration != 0 and iteration % int(len(train_loader) / 5) == 0:
                    # Evaluate the model five times per epoch
                    with torch.no_grad():
                        model.eval()
                        valid_losses = []
                        predictions = []
                        target_labels = []
                        for input_ids, attention_mask, token_type_ids, position_ids, labels in tqdm(dev_loader,
                                                                                                    desc='Eval',
                                                                                                    position=1,
                                                                                                    leave=None):
                            input_ids = input_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            token_type_ids = token_type_ids.to(device)
                            position_ids = position_ids.to(device)
                            labels = labels.to(device, dtype=torch.long)

                            output = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        labels=labels)

                            logits = output.logits
                            loss = output.loss
                            valid_losses.append(loss.item())

                            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                            batch_labels = [int(example) for example in labels]

                            predictions += batch_predictions
                            target_labels += batch_labels

                    acc = compute_acc(predictions, target_labels)
                    valid_loss = sum(valid_losses) / len(valid_losses)

                    if lowest_valid_loss > valid_loss:
                        print('Acc for model which have lower valid loss: ', acc)
                        torch.save(model.state_dict(), "./pytorch_model.bin")

if __name__ == "__main__":
    
    # 프로그램을 실행시에 커맨드 라인에 인수를 받아 처리를 간단히 할 수 있도록 하는 표준 라이브러리이다.
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--eval_bs', type=int, default=32)
    parser.add_argument('--lr', type=int, default=32)
    parser.add_argument('--n-epoch', type=int, default=10)
    parser.add_argument('--proj_name', type=str, default='batch_')
    
    args = parser.parse_args()
    train(args)
    