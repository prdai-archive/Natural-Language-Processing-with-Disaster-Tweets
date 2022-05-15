from Modelling import *


class Sklearn_Data_Loader:
    def __init__(
        self,
        data: pd.DataFrame = pd.read_csv("./Modelling/dataset/data/train.csv"),
        test: pd.DataFrame = pd.read_csv("./Modelling/dataset/data/test.csv"),
        all_words: list = [],
        tags: list = [],
    ):
        self.data = data
        self.data = self.data.sample(frac=1)
        self.test = test
        self.X = self.data["text"]
        self.y = self.data["target"]
        # self.y = np.array(self.y.tolist())
        # self.y = self.y.reshape(-1, 1)
        self.all_words = all_words

    def create(self, count_vectorizer=True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.125, shuffle=True
        )
        if count_vectorizer:
            self.vectorizer = CountVectorizer(binary=1)
        else:
            self.vectorizer = TfidfVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
        # self.X_train = np.array(self.X_train)
        # self.y_train = np.array(self.y_train)
        # self.X_test = np.array(self.X_test)
        # self.y_test = np.array(self.y_test)
        return self.X_train, self.X_test, self.y_train, self.y_test, self.vectorizer

    def create_test(self):
        self.test = self.vectorizer(self.test)
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
