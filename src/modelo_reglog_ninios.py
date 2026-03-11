from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import pandas as pd
import json
import isodate
from modelos_utils import download_model_dfs

#https://www.geeksforgeeks.org/machine-learning/text-classification-using-logistic-regression/

df_train, df_validation, df_test = download_model_dfs()
