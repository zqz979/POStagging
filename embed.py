import utils
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from embed_def import *

EMBEDDING_DIM=8

LR=0.001
EPOCHS=20
TRAIN_BATCH=256
TEST_BATCH=1024

NUM_WORKERS=0

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
    
def train(train_loader, model, optimizer, criterion):
    model=model.train()
    losses = []
    correct=0
    incorrect=0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += torch.sum(output.argmax(axis=1) == target)
        incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def test(test_loader, model, criterion):
    model=model.eval()
    losses = []
    correct = 0
    incorrect=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            correct += torch.sum(output.argmax(axis=1) == target)
            incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def extract_features(sentences):
    return [[token['word'] for token in sentence] for sentence in sentences]

class WordDataset(torch.utils.data.Dataset):
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        y=torch.tensor(self.labels[index],dtype=torch.long)
        X=torch.tensor(self.data[index], dtype=torch.long)
        return X, y

def load_data():
    train_set = utils.load_data('./data/train.txt', 'dictionary')
    test_set = utils.load_data('./data/test.txt', 'dictionary')
    train_set, eval_set = train_test_split(train_set, test_size=0.2)
    X_train = extract_features(train_set)
    y_train = utils.extract_labels(train_set)
    X_eval = extract_features(eval_set)
    y_eval = utils.extract_labels(eval_set)
    X_test = extract_features(test_set)
    X_train_vec =[feature for sentence in X_train for feature in sentence]
    X_eval_vec =[feature for sentence in X_eval for feature in sentence]
    X_test_vec =[feature for sentence in X_test for feature in sentence]
    y_train_vec=[label for sentence in y_train for label in sentence]
    y_eval_vec=[label for sentence in y_eval for label in sentence]
    vocab = set(X_test_vec) | set(X_train_vec) | set(X_eval_vec)
    X_enc=LabelEncoder()
    X_enc=X_enc.fit(list(vocab))
    y_enc=LabelEncoder()
    y_enc=y_enc.fit(list(set(y_train_vec) | set(y_eval_vec)))
    X_train_vec=X_enc.transform(X_train_vec)
    X_eval_vec=X_enc.transform(X_eval_vec)
    y_train_vec=y_enc.transform(y_train_vec)
    y_eval_vec=y_enc.transform(y_eval_vec)
    train_loader=torch.utils.data.DataLoader(
        WordDataset(X_train_vec,y_train_vec),
        batch_size=TRAIN_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader=torch.utils.data.DataLoader(
        WordDataset(X_eval_vec,y_eval_vec),
        batch_size=TEST_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    utils.save_model(y_enc, "./custom_embeddings/y_enc.sav")
    utils.save_model(X_enc, "./custom_embeddings/X_enc.sav")
    return train_loader, test_loader, len(vocab), 1, len(y_enc.classes_)

def plt_losses(train_losses,test_losses):
    plt.figure()
    plt.plot(range(EPOCHS),train_losses, label="Train Loss")
    plt.plot(range(EPOCHS),test_losses, label="Test Loss")
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def main():
    train_loader,test_loader,vocab_size,in_features,num_classes=load_data()
    model=Embedder(vocab_size,EMBEDDING_DIM,in_features=in_features,num_classes=num_classes)
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=LR)
    train_losses=[]
    test_losses=[]
    for epoch in range(EPOCHS):
        train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        train_losses.append(train_loss)
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
    plt_losses(train_losses,test_losses)
    print(f'Accuracy: {test_acc}')
    utils.save_model(model.to("cpu"), "./custom_embeddings/mlp.pth")

main()