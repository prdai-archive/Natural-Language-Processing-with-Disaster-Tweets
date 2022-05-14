from Modelling import *


class Pytorch_Data_Loader:
    def __init__(
        self,
        data:pd.DataFrame = pd.read_csv("./Modelling/dataset/data/train.csv"),
        test:pd.DataFrame = pd.read_csv("./Modelling/dataset/data/test.csv"),
        all_words:list = [],
        tags:list = []
    ):
        self.data = data
        self.data = self.data.sample(frac=1)
        self.test = test
        self.X = self.data["text"]
        self.y = self.data["target"]
        self.all_words = all_words
        self.tags = tags

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence.lower())

    def stem(self, word):
        return stemmer.stem(word.lower())

    def words_to_int(self, words, all_words):
        new_words = []
        for word in words:
            new_words.append(self.stem(word))
        list_of_os = np.zeros(len(all_words))
        for i in range(len(all_words)):
            if all_words[i] in new_words:
                list_of_os[i] = 1.0
        return list_of_os
    def create_all_words(self):
        for x_iter,y_iter in tqdm(zip(self.X,self.y)):
            x_iter = self.tokenize(x_iter)
            new_x_iter = []
            for x_iter_i in x_iter:
                new_x_iter.append(self.stem(x_iter_i))
            self.all_words.extend(new_x_iter)
            self.tags.append(y_iter)
        self.all_words = np.random.shuffle(self.all_words)
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))
        return self.all_words,self.tags
    def create(self,test_size=0.0625,shuffle=True):
        self.new_X = []
        self.new_y = []
        for X_iter,y_iter in tqdm(zip(self.X,self.y)):
            self.new_X.append(self.words_to_int(X_iter,self.all_words))
            self.new_y.append(self.tags.index(y_iter))
        self.X = np.array(self.new_X)
        self.y = np.array(self.new_y) 
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=test_size,shuffle=shuffle)
        self.X_train = torch.from_numpy(self.X_train).to(device)
        self.y_train = torch.from_numpy(self.y_train).to(device)
        self.X_test = torch.from_numpy(self.X_test).to(device)
        self.y_test = torch.from_numpy(self.y_test).to(device)
        return self.new_X,self.new_y,self.X,self.y,self.X_train,self.X_test,self.y_train,self.y_test
    def create_test(self):
        new_test = []
        for X_iter in tqdm(self.test['text']):
            new_test.append(self.words_to_int(X_iter,self.all_words))
        new_test = torch.from_numpy(np.array(new_test)).to(device)
        return new_test

    def create_submission(self,model):
        preds = model(self.create_test().float())
        ids = test['id']
        submission = {
            "id":[],
            "target":[]
        }
        for pred,id in tqdm(zip(preds,ids)):
            submission['id'].append(id)
            submission['target'].append(int(torch.round(pred)))
        submission = pd.DataFrame(submission)
        return submission
