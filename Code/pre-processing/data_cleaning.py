import re
import nltk
from nltk import data
import numpy as np
import pandas as pd
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv("../../Datasets/JobsIT_Dataset.csv")
dataset = dataset.loc[:, ["Query", "Description"]]

for i in range(len(dataset)):
    description = re.sub('[^a-zA-Z]', ' ', dataset["Description"][i])
    description = description.lower()
    description = description.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    description = [ps.stem(word) for word in description if not word in set(all_stopwords)]
    description = ' '.join(description)
    dataset["Description"][i] = description

dataset.to_csv("../../Cleaned_Datasets/JobsIT_Dataset.csv", index=False)