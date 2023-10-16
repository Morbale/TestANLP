import argparse

import torch
from torch import optim
from transformers import BertForTokenClassification
from sklearn.metrics import precision_score, recall_score, f1_score

from dataloader import *


def train(args, data_loader):
    num_labels = 15  # Define the number of unique labels for your specific task
    # Load the model
    model = BertForTokenClassification.from_pretrained(
        args.model, num_labels=15)
    # We already are getting the tokenized tokens
    # tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_cased')

    if args.optimizer == "Adam":
        optimizer = optim.AdamW(model.parameters(), args.lr)
    if args.loss == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()

    # Step 6: Fine-tune the Model
    if args.type == "finetune":
        for epoch in range(10):
            model.train()
            for batch in data_loader:
                input_ids = batch[0]
                labels = batch[1]

                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                loss = criterion(outputs.logits.view(-1, num_labels),
                                 labels.view(-1))  # Adjust labels accordingly
                loss.backward()
                optimizer.step()

    model.save_pretrained(args.output_path)


def evaluate(args, test):
    model = BertForTokenClassification.from_pretrained(args.output_path)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in test:
            outputs = model(**data)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(data['labels'].view(-1).cpu().numpy())
            all_preds.extend(predictions.view(-1).cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help='Model to use', default="allenai/scibert_scivocab_cased")
    parser.add_argument("--batch_size", type=int, help='Batch Size', default=1)

    parser.add_argument("--optimizer", type=str, help='Adam or any other', default="Adam")
    parser.add_argument("--loss", type=str, help='Loss function to use', default="CrossEntropy")
    parser.add_argument("--epochs", type=int, help='Number of Epochs', default=10)
    parser.add_argument("--lr", type=float, help='Learning Rate', default=1e-3)
    parser.add_argument("--output_path", type=str, help='Path where model weights should be stored',
                        default="finetuned/")
    
    parser.add_argument("--type", type=str, help='Freeze/Finetune', default="finetune")
    args = parser.parse_args()

    train_data = get_data('data/splits/train.conll', args.model, args.batch_size)
    test_data = get_data('data/splits/test.conll', args.model, args.batch_size)
    #
    # ipt = [{'input_ids': tensor([[100, 864, 19676, 3832, 100, 173, 14203, 100]]),
    #         'labels': tensor([[10, 14, 10, 11, 11, 11, 11, 11]])}]

    # test = [{'input_ids': tensor([[100, 864, 19676, 3832, 100, 173, 14203, 100]]),
    #          'labels': tensor([[10, 14, 10, 11, 11, 11, 11, 11]])}]


    train(args, train_data)
    precision, recall, f1_score = evaluate(args, test_data)
    print(precision,recall,f1_score)
