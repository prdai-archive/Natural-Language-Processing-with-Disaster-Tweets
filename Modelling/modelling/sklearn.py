from Modelling import *


class Sklearn_Modelling:
    def train(self, model, X_train, y_train, X_test, y_test, name):
        wandb.init(project=PROJECT_NAME, name=name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_prob = model.predict_proba(X_test)
        labels = sorted(set(list(y_test)))
        wandb.log(
            {
                "Accuracy": accuracy_score(preds, y_test),
                "F1 Score": f1_score(preds, y_test),
                "Precisions": precision_score(preds, y_test),
            }
        )
        wandb.sklearn.plot_confusion_matrix(y_test, preds, labels)
        # wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, preds, preds_prob, {0:0,1:1},
        #                                                  model_name=f'{model}')
        wandb.sklearn.plot_roc(y_test, preds_prob, labels)
        wandb.sklearn.plot_precision_recall(y_test, preds_prob, labels)
        wandb.sklearn.plot_confusion_matrix(y_test, preds, labels)
        wandb.finish()
        return model, preds, preds_prob
