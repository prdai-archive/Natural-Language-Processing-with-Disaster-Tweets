from Modelling import *

sdl = Sklearn_Data_Loader()
X_train, X_test, y_train, y_test, vectorizer = sdl.create()
model = svm.SVC(C=16, kernel="linear", gamma="auto",probability=True)
sm = Sklearn_Modelling()
model, preds, preds_prob= sm.train(model,X_train, y_train, X_test, y_test, "BaseLine")
sdl.create_submission(model)
