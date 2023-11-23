from fastapi import FastAPI
import uvicorn
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shutil
import joblib

player_scaler = joblib.load('player_scaler.pkl')
player_model = pkl.load(open('player_model.pkl', 'rb'))

app=FastAPI()

def mapping(player_name, team, opponent_team,venue):
    team_dict =  {'Afghanistan': 0,
                'Australia': 1,
                'Bangladesh': 2,
                'England': 3,
                'India': 4,
                'Netherlands': 5,
                'New Zealand': 6,
                'Pakistan': 7,
                'South Africa': 8,
                'Sri Lanka': 9}
    venue_dict = {'Arun Jaitley Stadium': 0,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 1,
    'Eden Gardens': 2,
    'Himachal Pradesh Cricket Association Stadium': 3,
    'M Chinnaswamy Stadium': 4,
    'MA Chidambaram Stadium, Chepauk': 5,
    'Maharashtra Cricket Association Stadium': 6,
    'Narendra Modi Stadium': 7,
    'Rajiv Gandhi International Stadium': 8,
    'Wankhede Stadium': 9}
    player_dict = {'A Dutt': 0,
    'A Zampa': 1,
    'AAP Atkinson': 2,
    'AD Mathews': 3,
    'AK Markram': 4,
    'AT Carey': 5,
    'AT Nidamanuru': 6,
    'AU Rashid': 7,
    'Abdullah Shafique': 8,
    'Azmatullah Omarzai': 9,
    'BA Stokes': 10,
    'BFW de Leede': 11,
    'BKG Mendis': 12,
    'Babar Azam': 13,
    'C Green': 14,
    'C Karunaratne': 15,
    'CAK Rajitha': 16,
    'CBRLS Kumara': 17,
    'CN Ackermann': 18,
    'CR Woakes': 19,
    'D Madushanka': 20,
    'DA Miller': 21,
    'DA Warner': 22,
    'DJ Malan': 23,
    'DJ Mitchell': 24,
    'DJ Willey': 25,
    'DM de Silva': 26,
    'DN Wellalage': 27,
    'DP Conway': 28,
    'FDM Karunaratne': 29,
    'Fakhar Zaman': 30,
    'Fazalhaq Farooqi': 31,
    'G Coetzee': 32,
    'GD Phillips': 33,
    'GJ Maxwell': 34,
    'H Klaasen': 35,
    'HC Brook': 36,
    'HE van der Dussen': 37,
    'HH Pandya': 38,
    'Haris Rauf': 39,
    'Hasan Ali': 40,
    'Hasan Mahmud': 41,
    'Hashmatullah Shahidi': 42,
    'Ibrahim Zadran': 43,
    'Iftikhar Ahmed': 44,
    'Ikram Alikhil': 45,
    'Imam-ul-Haq': 46,
    'Ishan Kishan': 47,
    'JC Buttler': 48,
    'JDS Neesham': 49,
    'JE Root': 50,
    'JJ Bumrah': 51,
    'JM Bairstow': 52,
    'JP Inglis': 53,
    'JR Hazlewood': 54,
    'K Rabada': 55,
    'KA Maharaj': 56,
    'KIC Asalanka': 57,
    'KL Rahul': 58,
    'KS Williamson': 59,
    'Kuldeep Yadav': 60,
    'L Ngidi': 61,
    'LH Ferguson': 62,
    'LS Livingstone': 63,
    'LV van Beek': 64,
    'M Jansen': 65,
    'M Labuschagne': 66,
    'M Pathirana': 67,
    'M Theekshana': 68,
    'MA Starc': 69,
    'MA Wood': 70,
    'MADI Hemantha': 71,
    'MD Shanaka': 72,
    'MDKJ Perera': 73,
    'MJ Henry': 74,
    'MJ Santner': 75,
    'MM Ali': 76,
    "MP O'Dowd": 77,
    'MP Stoinis': 78,
    'MR Marsh': 79,
    'MS Chapman': 80,
    'Mahedi Hasan': 81,
    'Mahmudullah': 82,
    'Mohammad Nabi': 83,
    'Mohammad Nawaz': 84,
    'Mohammad Rizwan': 85,
    'Mohammad Wasim': 86,
    'Mohammed Shami': 87,
    'Mujeeb Ur Rahman': 88,
    'Mushfiqur Rahim': 89,
    'Mustafizur Rahman': 90,
    'Najibullah Zadran': 91,
    'Nasum Ahmed': 92,
    'Naveen-ul-Haq': 93,
    'P Nissanka': 94,
    'PA van Meekeren': 95,
    'PJ Cummins': 96,
    'PVD Chameera': 97,
    'Q de Kock': 98,
    'R Klein': 99,
    'R Ravindra': 100,
    'RA Jadeja': 101,
    'RE van der Merwe': 102,
    'RG Sharma': 103,
    'RJW Topley': 104,
    'RR Hendricks': 105,
    'Rahmanullah Gurbaz': 106,
    'Rahmat Shah': 107,
    'Rashid Khan': 108,
    'S Samarawickrama': 109,
    'SA Edwards': 110,
    'SA Engelbrecht': 111,
    'SA Yadav': 112,
    'SM Curran': 113,
    'SPD Smith': 114,
    'SS Iyer': 115,
    'Saqib Zulfiqar': 116,
    'Saud Shakeel': 117,
    'Shadab Khan': 118,
    'Shaheen Shah Afridi': 119,
    'Shakib Al Hasan': 120,
    'Shariz Ahmad': 121,
    'Shoriful Islam': 122,
    'Shubman Gill': 123,
    'T Bavuma': 124,
    'T Shamsi': 125,
    'TA Boult': 126,
    'TG Southee': 127,
    'TM Head': 128,
    'TWM Latham': 129,
    'Tanzid Hasan': 130,
    'Taskin Ahmed': 131,
    'Towhid Hridoy': 132,
    'Usama Mir': 133,
    'V Kohli': 134,
    'Vikramjit Singh': 135,
    'W Barresi': 136,
    'WA Young': 137}

    player_name = player_dict[player_name]
    team = team_dict[team]
    opponent_team = team_dict[opponent_team]
    venue = venue_dict[venue]

    return player_name, team, opponent_team, venue

@app.get('/get-batsman')
def top_batsman():
    player_df = pd.read_csv('player_details.csv')
    top_run_scorer = player_df.groupby(['player_name'])['match_runs'].sum().reset_index()
    top_run_scorer = top_run_scorer.sort_values(by='match_runs', ascending=False)
    top_run_scorer = top_run_scorer.head(15)
    # print(top_run_scorer)
    player_df.drop(['total_runs','match_runs'], axis=1, inplace=True)
    player_df = player_df[player_df['player_name'].isin(top_run_scorer['player_name'])]
    # print(player_df)
    player_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    # print(top_run_scorer.columns)
    player_df = player_df.merge(top_run_scorer, on='player_name', how='left')

    player_df.rename(columns={'match_runs': 'total_runs'}, inplace=True)
    player_df = player_df.sort_values(by='total_runs', ascending=False)
    player_df.drop(['opponent_team', 'venue'], axis=1, inplace=True)

    upcoming_matches = pd.read_csv('upcoming_matches.csv')

    predict_df = pd.DataFrame(columns=['player_name','team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'])

    for index, player_row in player_df.iterrows():
        for index, match_row in upcoming_matches.iterrows():
            if player_row['team'] == match_row['team1'] or player_row['team'] == match_row['team2']:
                #in predict_df add 'player_name','team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'
                opponent_team = match_row['team1'] if player_row['team'] == match_row['team2'] else match_row['team2']
                predict_df = predict_df._append({'player_name': player_row['player_name'], 'team': player_row['team'], 'opponent_team':opponent_team, 'venue': match_row['venue'], 'total_runs': player_row['total_runs'], 'highest_score': player_row['highest_score'], 'batting_avg': player_row['batting_avg'], 'strike_rate': player_row['strike_rate'], 'bowling_runs': player_row['bowling_runs'], 'total_wickets': player_row['total_wickets'], 'bowling_avg': player_row['bowling_avg'], 'economy': player_row['economy']}, ignore_index=True)

    predict_df = predict_df.reset_index(drop=True)
    final_df = predict_df.copy()
    for index, row in predict_df.iterrows():
        predict_df.loc[index, 'player_name'], predict_df.loc[index, 'team'], predict_df.loc[index, 'opponent_team'], predict_df.loc[index,'venue'] = mapping(row['player_name'], row['team'], row['opponent_team'],row['venue'])

    predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = player_scaler.transform(predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])
    predict_df[['player_name','team','opponent_team','venue']] = predict_df[['player_name','team','opponent_team','venue']].astype(int)
    predict_df = predict_df[['player_name','team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy']]
    runs = player_model.predict(predict_df)
    runs = runs.astype(int)
    runs = runs[:,0]
    final_df['runs'] = runs
    final_df['predicted_runs'] = final_df.groupby('player_name')['runs'].transform('sum')
    final_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    final_df['total_runs'] = final_df['total_runs'] + final_df['predicted_runs']
    final_df.drop(['predicted_runs','runs','team','opponent_team','venue','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'], axis=1, inplace=True)
    final_df['answer'] = final_df['player_name'] + ' - ' + final_df['total_runs'].astype(str)
    return final_df['answer']


@app.get('/get-bowler')
def top_bowler():
    player_df = pd.read_csv('player_details.csv')

    top_wicket_scorer = player_df.groupby(['player_name'])['match_wickets'].sum().reset_index()
    top_wicket_scorer = top_wicket_scorer.sort_values(by='match_wickets', ascending=False)
    top_wicket_scorer = top_wicket_scorer.head(15)
    # print(top_run_scorer)
    player_df.drop(['total_wickets','match_wickets'], axis=1, inplace=True)
    player_df = player_df[player_df['player_name'].isin(top_wicket_scorer['player_name'])]
    # print(player_df)
    player_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    # print(top_run_scorer.columns)
    player_df = player_df.merge(top_wicket_scorer, on='player_name', how='left')

    player_df.rename(columns={'match_wickets': 'total_wickets'}, inplace=True)
    player_df = player_df.sort_values(by='total_wickets', ascending=False)
    player_df.drop(['opponent_team', 'venue'], axis=1, inplace=True)

    upcoming_matches = pd.read_csv('upcoming_matches.csv')

    predict_df = pd.DataFrame(columns=['player_name','team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'])

    for index, player_row in player_df.iterrows():
        for index, match_row in upcoming_matches.iterrows():
            if player_row['team'] == match_row['team1'] or player_row['team'] == match_row['team2']:
                opponent_team = match_row['team1'] if player_row['team'] == match_row['team2'] else match_row['team2']
                predict_df = predict_df._append({'player_name': player_row['player_name'], 'team': player_row['team'], 'opponent_team':opponent_team, 'venue': match_row['venue'], 'total_runs': player_row['total_runs'], 'highest_score': player_row['highest_score'], 'batting_avg': player_row['batting_avg'], 'strike_rate': player_row['strike_rate'], 'bowling_runs': player_row['bowling_runs'], 'total_wickets': player_row['total_wickets'], 'bowling_avg': player_row['bowling_avg'], 'economy': player_row['economy']}, ignore_index=True)

    predict_df = predict_df.reset_index(drop=True)
    final_df = predict_df.copy()
    for index, row in predict_df.iterrows():
        predict_df.loc[index, 'player_name'], predict_df.loc[index, 'team'], predict_df.loc[index, 'opponent_team'], predict_df.loc[index,'venue'] = mapping(row['player_name'], row['team'], row['opponent_team'],row['venue'])

    predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = player_scaler.transform(predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])
    predict_df[['player_name','team','opponent_team','venue']] = predict_df[['player_name','team','opponent_team','venue']].astype(int)
    predict_df = predict_df[['player_name','team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy']]
    wickets = player_model.predict(predict_df)
    wickets = wickets.astype(int)
    wickets = wickets[:,1]
    wickets = np.where(wickets < 0, 0, wickets)
    final_df['wickets'] = wickets
    final_df['predicted_wickets'] = final_df.groupby('player_name')['wickets'].transform('sum')
    final_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    final_df['total_wickets'] = final_df['total_wickets'] + final_df['predicted_wickets']
    final_df.drop(['predicted_wickets','wickets','team','opponent_team','venue','highest_score','batting_avg','strike_rate','bowling_runs','bowling_avg','economy','total_runs'], axis=1, inplace=True)
    # print(final_df)
    final_df['answer'] = final_df['player_name'] + ' - ' + final_df['total_wickets'].astype(str)
    return final_df['answer']

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)