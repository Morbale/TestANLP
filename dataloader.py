import torch
import argparse
import os
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Create train, valid, test set from aggregated conll file
def split_conll_file(args):
    input_conll_file = args.aggfile
    dest_folder = args.folder
    train_ratio= args.train
    valid_ratio= args.val 
    test_ratio= args.test

    if train_ratio + valid_ratio + test_ratio != 1:
        raise ValueError("Train, valid and test ratios must sum to 1")
    
    # Shuffle and split the data
    with open(input_conll_file, 'r') as f:
        data = f.readlines()
        random.shuffle(data)

    total_samples = len(data)
    train_split = int(train_ratio * total_samples)
    valid_split = int((train_ratio + valid_ratio) * total_samples)

    train_data = data[:train_split]
    valid_data = data[train_split:valid_split]
    test_data = data[valid_split:]

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    with open(os.path.join(dest_folder, 'train.conll'), 'w') as f:
        f.writelines(train_data)

    with open(os.path.join(dest_folder, 'val.conll'), 'w') as f:
        f.writelines(valid_data)

    with open(os.path.join(dest_folder, 'test.conll'), 'w') as f:
        f.writelines(test_data)


label2id = {'O': 0,
            'B-MethodName':1,
            'I-MethodName': 2,
            'B-HyperparameterName': 3,
            'I-HyperparameterName': 4,
            'B-HyperparameterValue': 5,
            'I-HyperparameterValue': 6,
            'B-MetricName': 7,
            'I-MetricName': 8,
            'B-MetricValue': 9,
            'I-MetricValue': 10,
            'B-TaskName': 11,
            'I-TaskName': 12,
            'B-DatasetName': 13,
            'I-DatasetName': 14
            }

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(tokenizer, token, label):
    tokenized_inputs = tokenizer(
       token, is_split_into_words=True, 
       return_offsets_mapping=True, 
       padding='max_length',
       max_length=512, 
       truncation=False
    )
    labels = [label2id[l] for l in label]

    word_ids = tokenized_inputs.word_ids()
    new_labels=align_labels_with_tokens(labels, word_ids)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# read conll file, split by space and assign first to token, second to label
def get_data(conll_file, model, batch_size):
    token = []
    label = []
    with open(conll_file, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            if line:
                token.append(line[0])
                label.append(line[1])

    # model = 'allenai/scibert_scivocab_cased'
    tokenizer = AutoTokenizer.from_pretrained(model, pad_token_id=-100)

    data = tokenize_and_align_labels(tokenizer, token, label)
    input_id, label, attention_mask = data['input_ids'], data['labels'], data['attention_mask']

    assert len(input_id) == len(label)

    dataset = SciDataset(input_id, label, attention_mask)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    
    return data_loader

# to get token back: tokenizer.convert_ids_to_tokens(data['input_ids'])


class SciDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels, attention_mask):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return in torch tensor
        # make sure 2D tensor is getting returned
        input_ids = torch.unsqueeze(torch.tensor(self.input_ids[idx]), 0)
        labels = torch.unsqueeze(torch.tensor(self.labels[idx]), 0)
        attention_mask = torch.unsqueeze(torch.tensor(self.attention_mask[idx]), 0)
        return input_ids, labels, attention_mask
        # return torch.tensor(self.input_ids[idx]), torch.tensor(self.labels[idx]), torch.tensor(self.attention_mask[idx])

    def collate_fn(self, batch):
        # batch is a list of tuples of (input_ids, label)
        # return in torch tensor
        input_ids, labels, attention_mask = zip(*batch)
        return torch.stack(input_ids, dim=0), torch.stack(labels, dim=0), torch.stack(attention_mask, dim=0)


def get_args():
    parser = argparse.ArgumentParser()
    # Args for splitting agg conll file
    parser.add_argument("--split", action='store_true', help='create data splits of train,val,test')
    parser.add_argument("--aggfile", type=str, help='aggregated data file path', default='data/aggdata.conll')
    parser.add_argument("--folder", type=str, help='destination folder to save split data, or split data to use', default='data/splits')
    parser.add_argument("--train", type=float, help='train ratio', default=0.8)
    parser.add_argument("--val", type=float, help='val ratio', default=0.1)
    parser.add_argument("--test", type=float, help='test ratio', default=0.1)


    # parser.add_argument("--folder", type=str, help='folder containing conll files to be processed')
    # parser.add_argument("--aggfile", type=str, help='file path to aggregated conll file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.split:
        split_conll_file(args)