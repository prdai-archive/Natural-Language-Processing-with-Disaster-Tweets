from Modelling import *

sdl = Sklearn_Data_Loader()
X_train, X_test, y_train, y_test, vectorizer = sdl.create()
sm = Sklearn_Modelling()
sm.train(X_train, X_test, y_train, y_test,"BaseLine")
