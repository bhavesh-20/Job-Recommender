import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


all_stopwords = stopwords.words('english')

tf = TfidfVectorizer()
jobs = pd.read_csv("../Cleaned_Datasets/JobsIT_Dataset.csv")
with open("../predictor_files/input.txt",encoding="utf8") as f:
    prediction_text = f.readlines()

prediction_text = ' '.join(prediction_text)
prediction_text = re.sub('[^a-zA-Z]', ' ', prediction_text)
prediction_text = prediction_text.lower()
prediction_text = prediction_text.split()
ps = PorterStemmer()
prediction_text = [ps.stem(word) for word in prediction_text if not word in set(all_stopwords)]
prediction_text = ' '.join(prediction_text)

tfidf_jobs = tf.fit_transform(jobs["Description"])
tfidf_prediction_text = tf.transform([prediction_text])

similarity_measure = cosine_similarity(tfidf_jobs, tfidf_prediction_text)
labels = jobs["Query"].unique()
similarity_scores = {label: {"sum": 0, "count": 0} for label in labels} 
for i in range(len(similarity_measure)):
    similarity_scores[jobs["Query"][i]]["sum"] += similarity_measure[i][0]
    similarity_scores[jobs["Query"][i]]["count"] += 1

prediction, maximum = "", -1
for label in similarity_scores:
    avg = similarity_scores[label]["sum"]/similarity_scores[label]["count"]
    if avg>maximum:
        maximum = avg
        prediction = label
print(prediction, similarity_scores[prediction])