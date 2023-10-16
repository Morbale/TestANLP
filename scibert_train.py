import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import BertForTokenClassification
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from dataloader import *

TQDM_DISABLE=False
NUM_LABELS = 15

def save_model(args, model, optimizer, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args
    }
    torch.save(save_info, filepath)
    print(f"Model saved at {filepath}")



def evaluate(data_loader, model, device):
    model.eval()

    all_labels = []
    all_preds = []
    token_ids = []

    for step, batch in enumerate(tqdm(data_loader, desc=f'eval', disable=TQDM_DISABLE)):
        input_ids = batch[0]
        labels = batch[1]

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids, labels=labels)
        logits = logits.detach().cpu().numpy()

        predictions = np.argmax(logits, dim=-1).flatten()
        labels = labels.cpu().numpy().flatten()

        all_labels.extend(labels)
        all_preds.extend(predictions)
        token_ids.extend(input_ids.cpu().numpy().flatten())

        # all_labels.extend(data['labels'].view(-1).cpu().numpy())
        # all_preds.extend(predictions.view(-1).cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return acc, f1, all_preds, all_labels, token_ids

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    #load data
    train_data = get_data(os.path.join(args.datadir, 'train.conll'), args.model)
    train_dl = DataLoader(train_data, shuffle = True, batch_size=args.batch_size, collate_fn=train_data.collate_fn)
    dev_data = get_data(os.path.join(args.datadir, 'val.conll'), args.model)
    dev_dl = DataLoader(dev_data, shuffle = False, batch_size=args.batch_size, collate_fn=dev_data.collate_fn)

    model = BertForTokenClassification.from_pretrained(
        args.model, num_labels=NUM_LABELS)
    model = model.to(device)

    if args.optimizer == "Adam":
        optimizer = optim.AdamW(model.parameters(), args.lr)
    if args.loss == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    
    best_dev_acc = 0

    # Step 6: Fine-tune the Model
    if args.option == "finetune":
        print(f'Start Training {args.model} with {args.option} mode')
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batch = 0
            for step, batch in enumerate(tqdm(train_dl, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
                input_ids = batch[0]
                labels = batch[1]

                input_ids = input_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                loss = criterion(outputs.logits.view(-1, NUM_LABELS),
                                 labels.view(-1))  # Adjust labels accordingly
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batch += 1
            
            train_loss /= num_batch

            train_acc, train_f1, *_ = evaluate(train_dl, model, device)
            dev_acc, dev_f1, *_ = evaluate(dev_dl, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(args, model, optimizer, args.modelpath)
            print(f"epoch{epoch+1}/{args.epoch}: train_loss: {train_loss :.3f}, train_acc: {train_acc :.3f}, dev_acc: {dev_acc :.3f}")




def test(args):
    # model = BertForTokenClassification.from_pretrained(args.output_path)
    # model.eval()

    # all_labels = []
    # all_preds = []
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.modelpath)
        model = BertForTokenClassification.from_pretrained(args.model, num_labels=15)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.modelpath}")
        # dev_data = get_data(os.path.join(args.datadir, 'val.conll'), args.model)
        # dev_dl = DataLoader(dev_data, shuffle = False, batch_size=args.batch_size, collate_fn=dev_data.collate_fn)

        test_data = get_data(os.path.join(args.datadir, 'test.conll'), args.model)
        test_dl = DataLoader(test_data, shuffle = False, batch_size=args.batch_size, collate_fn=test_data.collate_fn)

        # dev_acc, dev_f1, dev_pred, dev_true, dev_token = evaluate(dev_dl, model, device)
        test_acc, test_f1, test_pred, test_true, test_token = evaluate(test_dl, model, device)
        
        # to get original token back :tokenizer.convert_ids_to_tokens(data['input_ids'])

        # print(f"dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
        print(f"test acc :: {test_acc :.3f}, test f1 :: {test_f1 :.3f}")
        df = pd.DataFrame({'ID': list(range(len(test_pred))), 'TokenID': test_token, 'Label': test_pred})
        df.to_csv(args.test_out, index=None)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help='Model to use', default="allenai/scibert_scivocab_cased")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--datadir", type=str, help='folder that stores split data', default="data/splits/")
    parser.add_argument("--test_out", type=str, help='Path where test output should be stored', default="test_out.csv")
    
    parser.add_argument("--option", type=str, help='Freeze/Finetune', default="finetune")
    parser.add_argument("--batch_size", type=int, help='Batch Size', default=1)
    parser.add_argument("--optimizer", type=str, help='Adam or any other', default="Adam")
    parser.add_argument("--loss", type=str, help='Loss function to use', default="CrossEntropy")
    parser.add_argument("--epochs", type=int, help='Number of Epochs', default=10)
    parser.add_argument("--lr", type=float, help='Learning Rate', default=1e-3)
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == '__main__':
    args = get_args()
    # modelpath to contain current datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    args.modelpath = f'{current_time}_{args.option}-{args.epochs}-{args.lr}.pt'
    train(args)
    test(args)
