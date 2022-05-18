from Modelling import *

tdl = TensorFlow_Data_Loader()
X_train, X_test, y_train, y_test, = tdl.create()
tm = TensorFlow_Modelling()
model = tm.train(X_train, X_test, y_train, y_test,)
submission = tdl.create_submission(model)
submission.to_csv("./Pytorch-BaseLine.csv", index=False)

# SKlearn = 77%
# Pytorch = 60%
# Ten
