from Modelling import *


class Sklearn_Modelling:
    def train(self, model, X_train, y_train, X_test, y_test, name):
        wandb.init(project=PROJECT_NAME, name=name)
        model = model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_prob = model.predict_prob(X_test)
        print(list(y_test))
        wandb.sklearn.plot_confusion_matrix(y_test, preds, y_test)
        wandb.sklearn.plot_classifier(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            preds,
            preds_prob,
            labels,
            model_name=f"{model}",
            feature_names=None,
        )
        wandb.sklearn.plot_roc(y_test, preds_prob, labels)
        wandb.sklearn.plot_precision_recall(y_test, preds_prob, labels)
        wandb.sklearn.plot_confusion_matrix(y_test, preds, labels)
        wandb.finish()
        return model, preds, preds_prob
