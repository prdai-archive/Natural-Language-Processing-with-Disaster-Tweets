from Modelling import *

pdl = Pytorch_Data_Loader()
new_X, new_y, X, y, X_train, X_test, y_train, y_test, all_words = pdl.create()
pm = Pytorch_Modelling()
model = pm.train(X_train, X_test, y_train, y_test, all_words=all_words)
submission = pdl.create_submission(model)
submission.to_csv("./Pytorch-BaseLine.csv", index=False)

# SKlearn = 77%
# Pytorch =
