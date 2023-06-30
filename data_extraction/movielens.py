import numpy as np
import pandas as pd

db_path = 'movielens/ml-25m/ratings.csv'
ml_db = pd.read_csv(db_path)
ml_db = ml_db[['userId', 'movieId']]  # implicit feedbacks
ml_db.to_csv('ml-25.csv')