import pandas as pd

jobsIT_regular_df = pd.read_csv("../Datasets/JobsIT_Regular_Dataset.csv")
jobsIT_software_testing_df = pd.read_csv("../Datasets/JobsIT_Software_testing_Dataset.csv")

for i in range(len(jobsIT_software_testing_df)):
    jobsIT_software_testing_df.iat[i, 0] += 10000

frames = [jobsIT_regular_df, jobsIT_software_testing_df]
jobsIT_Dataset = pd.concat(frames)
jobsIT_Dataset.to_csv("../Datasets/JobsIT_Dataset.csv", index=False)