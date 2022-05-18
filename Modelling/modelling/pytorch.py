from Modelling import *


class Model(Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2):
        super(Model, self).__init__()
        self.l1 = Linear(input_size, hidden_size)
        self.l2 = Linear(hidden_size, hidden_size)
        self.l3 = Linear(hidden_size, num_classes)
        self.relu = ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


class Pytorch_Modelling:
    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        all_words,
        model=Model,
        criterion=CrossEntropyLoss(),  # TODO 
        optimizer=Adam,
        epochs=100,
        batch_size=32,
        name="BaseLine",
    ):
        model = Model(input_size=len(all_words)).to(device)
        optimizer = optimizer(model.parameters(), lr=0.001)
        wandb.init(project=PROJECT_NAME, name=name)
        wandb.watch(model, log_freq=10)
        torch.cuda.empty_cache()
        for _ in tqdm(range(epochs)):
            torch.cuda.empty_cache()
            for i in range(0, len(X_train), batch_size):
                torch.cuda.empty_cache()
                try:
                    X_batch = X_train[i : i + batch_size]
                    y_batch = y_train[i : i + batch_size]
                    preds = model(X_batch.float())
                    loss = criterion(preds.float(), y_batch.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(e)
            model.eval()
            wandb.log(
                {
                    "Val Accuracy": accuracy(model, X_test, y_test),
                    "Val Loss": g_loss(model, X_test, y_test.long(),criterion),
                    "Accuracy": accuracy(model, X_train, y_train),
                    "Loss": g_loss(model, X_train, y_train.long(),criterion),
                }
            )
            model.train()
        wandb.finish()
        return model
