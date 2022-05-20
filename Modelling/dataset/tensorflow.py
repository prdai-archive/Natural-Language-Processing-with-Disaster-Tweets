from Modelling import *


class TensorFlow_Data_Loader:
    def __init__(
        self,
        data: pd.DataFrame = pd.read_csv("./Modelling/dataset/data/train.csv"),
        test: pd.DataFrame = pd.read_csv("./Modelling/dataset/data/test.csv"),
    ):
        self.data = data
        self.data = self.data.sample(frac=1)
        self.test = test
        self.X = np.array(self.data["keyword"])
        self.y = np.array(self.data["target"])
        print(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.0625, shuffle=True
        )

    def create(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_test(self):
        return np.array(self.test["keyword"])

    def create_submission(self, model):
        preds = model.predict(self.create_test())
        print(preds)
        ids = self.test["id"]
        submission = {"id": [], "target": []}
        for pred, id in tqdm(zip(preds, ids)):
            submission["id"].append(id)
            submission["target"].append(int(pred))
        submission = pd.DataFrame(submission)
        return submission
