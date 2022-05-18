from Modelling import *


def accuracy(model, X, y):
    correct = 0
    total = 0
    preds = model(X.float())
    for pred, y_batch in zip(preds, y):
        pred = int(torch.round(pred))
        y_batch = int(y_batch)
        if pred == y_batch:
            correct += 1
        total += 1
    acc = round(correct / total, 3) * 100
    return acc
def g_loss(model,X,y,criterion):
    preds = model(X.float())
    loss = criterion(preds.view(-1),y.float().view(-1))
    return loss.item()
