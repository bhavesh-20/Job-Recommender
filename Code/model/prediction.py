import re
import nltk
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

from NN_config import config

all_stopwords = stopwords.words('english')

fp = open("../../predictor_files/input.txt", "r")
prediction_text = fp.readlines()
fp.close()


prediction_text = ' '.join(prediction_text)
prediction_text = re.sub('[^a-zA-Z]', ' ', prediction_text)
prediction_text = prediction_text.lower()
prediction_text = prediction_text.split()
ps = PorterStemmer()
prediction_text = [ps.stem(word) for word in prediction_text if not word in set(all_stopwords)]
prediction_text = ' '.join(prediction_text)


dataset = pd.read_csv("../../Cleaned_Datasets/JobsIT_Dataset.csv")
X = dataset["Description"]
tokenizer = Tokenizer(num_words=config.params.vocabulary.value)
tokenizer.fit_on_texts(X)
prediction_text = tokenizer.texts_to_matrix([prediction_text])
model = tf.keras.models.load_model("jobsIT_model")
prediction = model.predict(prediction_text)

encoder = LabelBinarizer()
encoder.fit(dataset["Query"])
print(encoder.inverse_transform(prediction))