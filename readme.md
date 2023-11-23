# CP03 Data Alchemists
## ICC World Cup 2023 Predictions
## Team Id - 23
### Team Members:
| Name              | Student ID   | Contribution |
| ----------------- | ------------ | ------------ |
| [Prarthee Desai](https://github.com/Prarthee-Desai)   | [202001257](mailto:202001257@daiict.ac.in)    | Leading wicket-taker prediction model |
| [Devansh Patel](https://github.com/dp2292)     | [202001262](mailto:202001262@daiict.ac.in)    | Semis and World Cup Final Winner and Playing 11 Team Prediction Model |
| [Dhruv Patel](https://github.com/Dhruv0348)       | [202001271](mailto:202001271@daiict.ac.in)    | Leading run scorer batsman prediction model |
| [Krish Patel](https://github.com/KrishPatel452)       | [202001452](mailto:202001452@daiict.ac.in)    | API Development, API Documentation & Testing|
| [Jaykumar Navadiya](https://github.com/jaynavadiya) | [202001465](mailto:202001465@daiict.ac.in)    | Data Scrapping, Feature Extraction, EDA and Documentation |

## Project Overview
In this project we had to make predictions regarding the ICC World Cup 2023. At the time of making this project, the world cup is ongoing and many matches are yet to be played. 

### Tasks
Task 1: \
a. Predicting the batsman who will score most runs in the tournament. \
b. Predicting the bowler who will be the leading wicket-taker in the tournament. \
Task 2: Predicting the Finalist Teams and Players \
Task 3: Predict the Winner of ICC Cricket World Cup 2023

### Result of the tasks

Task 1:
a. The batsman with the most scored runs (after 9 matches of each team - not including semi-finals and final match) - ![WhatsApp Image 2023-11-23 at 18 09 34](https://github.com/dp2292/CP03-Data-Alchemists/assets/67496808/6e9b7035-916c-4b46-b217-729e4c4ff8b9)

\
b. The leading wicket-taker bowler (after 9 matches of each team - not including semi-finals and final match) - ![WhatsApp Image 2023-11-23 at 18 09 16](https://github.com/dp2292/CP03-Data-Alchemists/assets/67496808/2a7f856c-580b-4b32-9ec2-5c742a13dc4c)
\
Task 2. Finalist Teams and Players:
Below is the screenshot from the api output for Finalist Teams: (read the API section for more information on the api)! \
![file_2023-11-14_16 11 26](https://github.com/dp2292/CP03-Data-Alchemists/assets/67496808/3ac9fbaf-d8c9-4199-9930-4cee50c16b5a)\
Below is the screenshot from the api output for Finalist Players for the winning teams: \
![file_2023-11-14_16 12 10](https://github.com/dp2292/CP03-Data-Alchemists/assets/67496808/0d9d0ee4-2327-4d92-b61f-1fbc45be27a0)\
Task 3. Winner of ICC Cricket World Cup 2023 - \
![file_2023-11-14_16 11 48](https://github.com/dp2292/CP03-Data-Alchemists/assets/67496808/fe623266-d316-4053-8c20-e872ee5c0f39)

### Data Used

1. We used [ICC Mens World Cup 2023](https://www.kaggle.com/datasets/pardeep19singh/icc-mens-world-cup-2023) from Kaggle for deliveries.csv, matches.csv and points_table.csv which is scrapped from ESPN Cricinfo and cricsheet as per description of the dataset on Kaggle.
2. We scrapped any other data used for this project from ESPN Cricinfo and obliged to the conditions set by ESPN Cricinfo for scrapping limitations. The csv files and its features are explained below. Also the scrappers can be found in EDA and Scrappers directory.

   
We have used various machine learning models for most tasks, mainly ANN (Artificial Neural Networks) & DecisionTrees and predicted the tasks. 
Mostly, the data used was from [ICC Mens World Cup 2023](https://www.kaggle.com/datasets/pardeep19singh/icc-mens-world-cup-2023). We used the first 32 ODIs out of 48 of this world cup for predicting results of the rest of the matches and other statistics!!

After creating the models, we created an api for every task which makes use of the underlying models to make predictions. The details of these apis are given below as well.


### API Installation and Run Instructions
   #### 1. Clone the Repository:
   Open a terminal or command prompt and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/dp2292/CP03-Data-Alchemists
   ```
#### 2. Navigate to the Repository

Change your current working directory to the cloned repository:

```bash
cd CP03-Data-Alchemists
```
#### 3. Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
#### 4. Run the `api.py` File

Once the dependencies are installed, you can run the `api.py` file:

```bash
python api.py
```

### API Endpoints

#### 1. Default Route

- **URL:** `/`
- **Method:** GET
- **Description:** Default route showing team members and team name.

#### 2. Predict Finalists

- **URL:** `/predict_finalists`
- **Method:** GET
- **Description:** Provides the two teams performing in the finals match. Expects a JSON object as a result with the following format:
  ```json
  {
      "final teams": [
          "Team A",
          "Team B"
      ]
  }
  ```
#### 3. Predict Winner

- **URL:** `/predict_winner`
- **Method:** GET
- **Description:** Returns the team expected to win the final match. Expects the result in JSON format with the following structure:

  ```json
  {
      "winner": [
          "Winning Team"
      ]
  }
  ```
#### 4. Get Playing 11

- **URL:** `/get_playing11`
- **Method:** GET
- **Description:** Predicts the 11 players of both teams who are going to play in the match. Expects to provide a JSON object with the following format:

  ```json
  {
      "Team A": [
          "Player 1",
          "Player 2",
          // ...
          "Player 11"
      ],
      "Team B": [
          "Player 1",
          "Player 2",
          // ...
          "Player 11"
      ]
  }
  ```

### Dataset Description

A. deliveries.csv

1. `match_id`: Identifier for a specific cricket match.

2. `season`: The cricket season during which the match took place.

3. `start_date`: The date when the match started.

4. `venue`: The location or stadium where the match was played.

5. `innings`: The inning number (1st or 2nd) in the match.

6. `ball`: The ball number in the current inning.

7. `batting_team`: The team currently batting.

8. `bowling_team`: The team currently bowling.

9. `striker`: The batsman who is currently facing the ball.

10. `non_striker`: The batsman at the non-striker's end.

11. `bowler`: The bowler delivering the ball.

12. `runs_off_bat`: The runs scored off the bat (excluding extras).

13. `extras`: Extra runs scored, including wides, no-balls, byes, leg-byes, and penalties.

14. `wides`: The run for wide deliveries bowled.

15. `noballs`: The run for no-balls bowled.

16. `byes`: Runs scored due to byes (awarded when the ball passes the batsman without touching the bat).

17. `legbyes`: Runs scored due to leg-byes (awarded for runs off the batsman's body).

18. `penalty`: Penalty runs awarded to the batting team.

19. `wicket_type`: Type of wicket taken (e.g., caught, bowled, lbw).

20. `player_dismissed`: The batsman who got dismissed (if a wicket fell).

21. `other_wicket_type`: Additional information on the type of wicket, if applicable.

22. `other_player_dismissed`: Additional information on the dismissed player, if applicable.

B. points_table.csv

1. `Ranking`: The position of the cricket team in the overall ranking.

2. `Team`: The name of the cricket team.

3. `Matches`: The total number of matches played by the team.

4. `Won`: The number of matches won by the team.

5. `Lost`: The number of matches lost by the team.

6. `Tie`: The number of matches ending in a tie for the team.

7. `No Results`: The number of matches with no conclusive result for the team.

8. `Points`: The total points earned by the team in the ranking system.

9. `Net Run Rate`: The team's net run rate, calculated based on runs scored and conceded.

10. `Series Form`: The team's recent performance trend in a series of matches.

11. `Next Match`: Information about the team's upcoming match.

12. `For`: The total runs scored by the team.

13. `Against`: The total runs conceded by the team.

C. matches.csv

1. `season`: The cricket season in which the match is played.

2. `team1`: One of the participating teams in the match.

3. `team2`: The other participating team in the match.

4. `date`: The date on which the match is scheduled.

5. `match_number`: The unique identifier for the match in the season.

6. `venue`: The location or stadium where the match will be played.

7. `city`: The city where the match is hosted.

8. `toss_winner`: The team winning the coin toss before the match.

9. `toss_decision`: The decision made by the toss winner (batting or bowling).

10. `player_of_match`: The player adjudged as the best performer in the match.

11. `umpire1`: The name of the first on-field umpire.

12. `umpire2`: The name of the second on-field umpire.

13. `reserve_umpire`: The reserve umpire designated for the match.

14. `match_referee`: The official overseeing the match and ensuring fair play.

15. `winner`: The team that emerged victorious in the match.

16. `winner_runs`: The margin of victory for the winning team in terms of runs.

17. `winner_wickets`: The number of wickets by which the winning team secured victory.

18. `match_type`: The type of cricket match (e.g., One Day International, T20, Test).

D. upcoming_matches.csv (Manually created for fixtures)

1. `team1`: One of the participating cricket teams in a match.

2. `team2`: The other participating cricket team in a match.

3. `venue`: The location or stadium where the cricket match is being played.

E. upcoming_semis.csv (Manually created)

1. `team1`: One of the participating cricket teams in a match.

2. `team2`: The other participating cricket team in a match.

3. `venue`: The location or stadium where the cricket match is being played.

F. upcoming_final.csv (Manually created after predicting the finalists by using the model)

1. `team1`: One of the participating cricket teams in a match.

2. `team2`: The other participating cricket team in a match.

3. `venue`: The location or stadium where the cricket match is being played.

G. player_details.csv (Scrapped from ESPN Cricinfo)

1. `player_name`: The name of the cricket player.
2. `team`: The team to which the player belongs.
3. `opponent_team`: The team the player is facing in a particular match.
4. `venue`: The location where the cricket match is being played.
5. `match_runs`: The total runs scored by the player in a specific match.
6. `match_wickets`: The number of wickets taken by the player in the match.
7. `total_runs`: Cumulative runs scored by the player across all matches.
8. `highest_score`: The player's highest individual score in a single inning.
9. `batting_avg`: The batting average of the player, calculated as total_runs divided by the number of times the player has been dismissed.
10. `strike_rate`: The strike rate of the player, calculated as the number of runs scored per 100 balls faced.
11. `bowling_runs`: The total runs conceded by the player while bowling in all matches.
12. `total_wickets`: The overall number of wickets taken by the player.
13. `bowling_avg`: The bowling average of the player, calculated as bowling_runs divided by the number of wickets taken.
14. `economy`: The economy rate of the player in bowling, indicating the average number of runs conceded per over.

H. previous_data_per.csv (Scrapped, Previous Data Percentages)

1. `Team`: The cricket team for which the statistics are recorded.
2. `Opposition`: The opposing team against which matches were played.
3. `Mat`: The total number of matches played by the team against the specified opposition.
4. `Won`: The number of matches won by the team against the specified opposition.
5. `Lost`: The number of matches lost by the team against the specified opposition.
6. `%`: The win percentage, calculated as the ratio of matches won to the total matches played, expressed as a percentage.

I. stadium_details.csv (Scrapped, ESPN Cricinfo)

1. `venue`: The cricket stadium or venue for which the statistics are recorded.
2. `won_after_bat_first`: The number of matches won at the venue by teams batting first.
3. `won_after_chase`: The number of matches won at the venue by teams chasing a target.
4. `first_inning_score`: The average score in the first inning at the venue across all matches.
5. `second_inning_score`: The average score in the second inning at the venue across all matches.

J. updatedTable.csv: points table, but updated after some matches (>32 matches played).

K. semis_winner.csv: 2 winners of semi finals were stored in this file.

Apart from this, we scrapped batsman and bowler stats for previous world cups and this world cup for visualizations and EDA. We used the best fitting data for making predictions via trial and error method and whatever that felt logical at the moment. For more information on the choices made on data, read comments and notes in the notebooks for the model!

### Using these dataset in various applications:

1. player_details -> playing 11
2. points_table -> finalist_prediction (just for refernce to calculate)
3. previous_data_per -> finalist/winner prediction (To predict the win rate of one team over other)
4. semis_winner -> semi final winner teams just to record the teams of the final
5. stadium_details -> final/winner prediction (By using avg runs per innings, other details)
6. upcoming_semis -> 2 matches of 4 teams of semi finals and its fixture
7. updatedTable -> updated point table

## Objectives

The primary objectives of this data mining project are as follows:

1. *Data Scrapping*: We used BeautifulSoup and Requests library for scrapping the data from ESPN Cricinfo for getting relevant data for making predictions for the ICC world cup 2023.
2. *Data Preprocessing*: Data preprocessing refers to the initial step in data analysis, where raw data is cleaned, transformed, and organized to make it suitable for further analysis and modeling. It involves handling missing values, dealing with outliers, standardizing data formats, and encoding categorical variables, ensuring the data is usable and consistent before applying data mining or machine learning techniques. Our dataset had 8,55,969 instances of data having 73 features, out of which we dropped few features.
3. *Exploratory Data Analysis (EDA)*: Exploratory Data Analysis (EDA) is visually and statistically analyzing a dataset to summarize its main characteristics, gain insights, and uncover patterns or anomalies. It involves techniques such as data visualization, summary statistics, and correlation analysis to understand better the data's structure, relationships between variables, and potential areas of further investigation in data analysis and modeling tasks. To visualize and understand the data, we have made boxplots, count plots, pie charts, bar graphs, point plots, violin plots,  etc.
4. *Feature Engineering*: Feature Engineering is the process of creating new, relevant, and informative features or transformations of existing ones from raw data to improve the performance of machine learning models. It involves selecting, combining, or transforming variables better to represent the underlying patterns and relationships in the data, ultimately enhancing the model's ability to make accurate predictions or classifications. Here we had a lot of scrapped data to work with, which required new features such as overs_left or runs_left and many more to be extracted from the available data.
5. *Model Selection:*: Model Selection is the process of choosing the most suitable machine learning algorithm or model from a set of candidate models to best address a specific problem or task based on factors like performance, accuracy, and generalization ability. We implemented Logistic Regression, Artificial Nueral Network Model, Support Vector Machine Model, SGD Classifier, Random Forest Classifier.
6. *Model Training and Tuning*: Model Training and Tuning involves training a machine learning model on a dataset to learn patterns and relationships and then optimizing the model's hyperparameters to achieve the best possible predictive performance. This iterative process aims to find the most accurate and effective model configuration for a specific task. We have used gradient descent algorithm to perform model tuning.
7. *Evaluation Metrics*: Evaluation metrics are quantitative measures used to assess the performance or accuracy of a predictive model or algorithm by quantifying the degree of its predictions' agreement with the actual observed outcomes. These metrics help understand how well a model performs its intended task, whether predicting values, classifying data, or solving other machine learning and data mining tasks. Common evaluation metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), accuracy, precision, recall, and F1-score, depending on the specific task and context.
