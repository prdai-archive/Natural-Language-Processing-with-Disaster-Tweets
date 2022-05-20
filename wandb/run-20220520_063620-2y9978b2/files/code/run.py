from Modelling import *

models = [
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2",
    "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2",
    "https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/2",
    "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
    "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1",
    "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    "https://tfhub.dev/google/nnlm-en-dim128/2",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
]
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
