# Job-Recommender

- Clone the repo
`git clone <SSH>`

- Change to repo's directory
`cd Job-Recommender`

## Set up Environment for the project

- Requires python 3.7 or more.
- Install virtualenv using
`pip install virtualenv`
- Set up python virtual environment for the project, using `virtualenv venv`
- Activate the virtualenv.
- If you are using windows then use the command `./venv/scripts/activate`.
- If you are using linux systems then use command `source venv/bin/activate`
- Download the required dependencies used in the project using, `pip install -r requirements.txt`

Congratulations! You have successfully set up the environment!

## Project Hierarchy
- The  `Datasets/` directory contains all the datasets downloaded for the project. It mainly consists of two job datasets one for IT and other for Non-IT, there are two IT datasets
and are merged using code written in `code/MERGE_IT_DATASETS.py`
- Data preprocessing and natural language processing is done in `code/data_cleaning.py`.
- Saved cleaned data to a new directory `Cleaned_Datasets/` to save time and reduce redundancy during runtime.

## Data cleaning
nltk library used. The following pipeline has been executed.
- remove all non alphabets regex = [^a-zA-Z], 
- remove whitespaces
- convert case to lowercase 
- tokenize words
- remove stopwords
- stemming
The result saved to `Cleaned_Dataset/` directory.
