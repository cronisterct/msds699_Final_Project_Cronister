import pandas as pd
import numpy as np

from category_encoders import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.compose import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
import joblib
import sys

# Support Code

def get_prediction(to_pred):
    pipe = joblib.load(f'lr_mod')

    preds = pipe.predict(to_pred)
    for_df = {'Prediction':preds}
    top_predictions = pd.DataFrame(for_df)
    if top_predictions['Prediction'][0]>=.9:
        return 'Great Retention', top_predictions, 'green'
    elif top_predictions['Prediction'][0]>=.7:
        return 'Good Retention', top_predictions, 'goldenrod'
    else:
        return 'Poor Retention', top_predictions, 'red'



def pre_process(row):

    categorical_columns = ['NCAA_CONFERENCE','SCHOOL_TYPE','NCAA_SUBDIVISION', 'SPORT_CODE']
    numerical_columns = ['FOURYEAR_ATHLETES', 'FOURYEAR_ELIGIBILITY']


    con_pipe = Pipeline([('scaler', StandardScaler()),
                          ('imputer', SimpleImputer(strategy='median', add_indicator=True))])

    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore')),
                         ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True))])

    preprocessing = ColumnTransformer([('categorical', cat_pipe,  categorical_columns),
                                       ('continuous',  con_pipe, numerical_columns),
                                       ])
    return preprocessing.transform(row)



file = sys.argv[1]
with open(file) as f:
    row = f.read()
# Process

pred = np.array(row.split(',')).reshape(1,-1)
cols = ['SCHOOL_TYPE', 'SPORT_CODE', 'NCAA_DIVISION',
       'NCAA_SUBDIVISION', 'NCAA_CONFERENCE', 'FOURYEAR_ATHLETES',
       'FOURYEAR_ELIGIBILITY']
pred_p = pd.DataFrame(pred, columns = cols)
#pred = pre_process(np.array(row).reshape(1, -1))

# Make a prediction


text = '<h1>Prediction:<br/></h1>'
values = get_prediction(pred_p)
color = values[2]
text += f'<h3 style="color:{color};">' + values[0]+'</h3>'
text += values[1].to_html()

with open('prediction.html','a+') as f:
    f.write(text)
