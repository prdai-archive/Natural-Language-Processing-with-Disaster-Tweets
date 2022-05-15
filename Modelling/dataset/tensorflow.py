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
        self.X = np.array(self.data["text"])
        self.y = np.array(self.data["target"])

    def create(self):
        return self.X,self.y,len(self.X),len(self.y)

    def create_test(self):
        self.test = np.array(self.test)
        return self.test

    def create_submission(self, model):
        preds = model.predict(self.create_test())
        ids = self.test["id"]
        submission = {"id": [], "target": []}
        for pred, id in tqdm(zip(preds, ids)):
            submission["id"].append(id)
            submission["target"].append(int(pred))
        submission = pd.DataFrame(submission)
        return submission
