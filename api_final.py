import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import random as rn
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load the model
toss_model = pickle.load(open('toss_prediction.pkl','rb'))
inning_1_run = pickle.load(open('inning_1_run.pkl','rb'))
inning_2_run = pickle.load(open('inning_2_run.pkl','rb'))
over_model = pickle.load(open('over_model.pkl','rb'))

@app.get('/')
def index():
    return {'message': 'Group 23 CP-03'}

@app.post('/predict')
def predict():
    df=pd.read_csv('matches.csv')
    df.drop(['season','date', 'match_number','player_of_match', 'umpire1', 'umpire2',
       'reserve_umpire', 'match_referee', 'winner', 'winner_runs',
       'winner_wickets', 'match_type','city'],axis='columns',inplace=True)

    le=LabelEncoder()
    toss_mapping = {}

    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        toss_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    print(toss_mapping)
    upcoming_df = pd.read_csv('upcoming_matches.csv')

    #for encoding use the mapping
    upcoming_df['team1'] = upcoming_df['team1'].map(toss_mapping['team1'])
    upcoming_df['team2'] = upcoming_df['team2'].map(toss_mapping['team2'])

    toss_loser = []
    print(upcoming_df)
    for i in range(len(upcoming_df)):
        team1 = upcoming_df['team1'][i]
        team2 = upcoming_df['team2'][i]
        prob = rn.random()
        if prob < 0.5:
            toss_winner.append(team1)
            toss_loser.append(team2)
        else:
            toss_winner.append(team2)
            toss_loser.append(team1)

predict()