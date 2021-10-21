import texthero as hero
import numpy as np
import pandas as pd

dataset = pd.read_csv("../../Datasets/JobsIT_Dataset.csv")
dataset = dataset.loc[:, ["Query", "Description"]]

dataset['Description'] = hero.clean(dataset["Description"])
dataset['Description'] = hero.preprocessing.stem(dataset["Description"])

dataset.to_csv("../../Cleaned_Datasets/JobsIT_Dataset.csv", index=False)