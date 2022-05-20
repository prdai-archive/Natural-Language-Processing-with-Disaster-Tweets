from Modelling import *

models = ["https://tfhub.dev/google/nnlm-en-dim50/2"]
for model in models:
    tdl = TensorFlow_Data_Loader()
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = tdl.create()
    tm = TensorFlow_Modelling()
    model = tm.train(X_train, X_test, y_train, y_test, model=model)
    submission = tdl.create_submission(model)
    submission.to_csv(f"./save/{model}-TensorFlow-BaseLine.csv", index=False)

# SKlearn = 77%
# Pytorch = 60%
# Ten
