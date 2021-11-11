import pandas as pd

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from NN_config import config
from model import model

dataset = pd.read_csv("../../Cleaned_Datasets/JobsIT_Dataset.csv")
X, y = dataset["Description"], dataset["Query"]
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
tokenizer = Tokenizer(num_words=config.params.vocabulary.value)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_matrix(x_train, mode='tfidf')
x_test = tokenizer.texts_to_matrix(x_test, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(y)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model.fit(
    x_train, y_train, 
    batch_size=config.params.batch_size.value, 
    epochs=config.params.epochs.value,
    verbose=1
)

score = model.evaluate(x_test, y_test,batch_size=config.params.batch_size.value, verbose=1)
print(score)
model.save("./jobsIT_model")