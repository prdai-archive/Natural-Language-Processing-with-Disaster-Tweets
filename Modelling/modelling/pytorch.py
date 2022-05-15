from Modelling import *


class Model(Module):
    def __init__(self, activation=ReLU, neurons=512):
        super().__init__()
        self.activation = activation()
        self.output_activation = Sigmoid()
        self.linear1 = Linear(len(all_words), neurons)
        self.linear2 = Linear(neurons, neurons * 2)
        self.linearbn = BatchNorm1d(neurons * 2)
        self.linear3 = Linear(neurons * 2, neurons * 2)
        self.linear4 = Linear(neurons * 2, neurons * 3)
        self.linear5 = Linear(neurons * 3, neurons * 2)
        self.output = Linear(neurons * 2, 1)

    def forward(self, X):
        preds = self.activation(self.linear1(X))
        preds = self.activation(self.linear2(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linearbn(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linear4(preds))
        preds = self.activation(self.linear5(preds))
        preds = self.output_activation(self.output(preds))
        return preds


class Pytorch_Modelling:
    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        model=Model(activation=LeakyReLU, neurons=1024).to(device),
        criterion=MSELoss(),
        optimizer=Adam(model.parameters(), lr=0.001),
        epochs=100,
        batch_size=32,
        name="BaseLine",
    ):
        wandb.init(project=PROJECT_NAME, name=name)
        wandb.watch(model, log_freq=10)
        for _ in tqdm(range(epochs)):
            for i in range(0, len(X_train), batch_size):
                try:
                    X_batch = X_train[i : i + batch_size]
                    y_batch = y_train[i : i + batch_size]
                    preds = model(X_batch.float())
                    loss = criterion(preds.float().view(-1), y_batch.float().view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except:
                    pass
            model.eval()
            wandb.log(
                {
                    "Val Accuracy": accuracy(model, X_test, y_test),
                    "Val Loss": g_loss(model, X_test, y_test),
                    "Accuracy": accuracy(model, X_train, y_train),
                    "Loss": g_loss(model, X_train, y_train),
                }
            )
            model.train()
        wandb.finish()
