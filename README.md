# CP03 Data Alchemists
## ICC World Cup 2023 Predictions
## Team Id - 23
### Team Members:
| Name              | Student ID   | Contribution |
| ----------------- | ------------ | ------------ |
| [Prarthee Desai](https://github.com/Prarthee-Desai)   | [202001257](mailto:202001257@daiict.ac.in)    |  |
| [Devansh Patel](https://github.com/dp2292)     | [202001262](mailto:202001262@daiict.ac.in)    | |
| [Dhruv Patel](https://github.com/Dhruv0348)       | [202001271](mailto:202001271@daiict.ac.in)    | |
| [Krish Patel](https://github.com/KrishPatel452)       | [202001452](mailto:202001452@daiict.ac.in)    |  |
| [Jaykumar Navadiya](https://github.com/jaynavadiya) | [202001465](mailto:202001465@daiict.ac.in)    | |

## Project Overview
In this project we had to make predictions regarding the ICC World Cup 2023. At the time of making this project, the world cup is ongoing and many matches are yet to be played. 

### Tasks
1. Predicting the batsman who will score most runs in the tournament.
2. Predicting the bowler who will be the leading wicket-taker in the tournament.
3. Predicting the Finalist Teams and Players
4. Predict the Winner of ICC Cricket World Cup 2023

### Result of the tasks

1. The batsman with the most scored runs (after 9 matches of each team - not including semi-finals and final match) -
2. The leading wicket-taker bowler (after 9 matches of each team - not including semi-finals and final match) -
3. Predicting the Playing 11 for the finalists - 
4. Finalist Teams and Players -
5. Winner of ICC Cricket World Cup 2023 - 

### Data Used

1. We used [ICC Mens World Cup 2023](https://www.kaggle.com/datasets/pardeep19singh/icc-mens-world-cup-2023) from Kaggle for deliveries.csv, matches.csv and points_table.csv which is scrapped from ESPN Cricinfo and cricsheet as per description of the dataset on Kaggle.
2. We scrapped any other data used for this project from ESPN Cricinfo and obliged to the conditions set by ESPN Cricinfo for scrapping limitations. The csv files and its features are explained below. Also the scrappers can be found in EDA and Scrappers directory.

   
We have used various machine learning models for most tasks, mainly ANN (Artificial Neural Networks) & DecisionTrees and predicted the tasks. 
Mostly, the data used was from [ICC Mens World Cup 2023](https://www.kaggle.com/datasets/pardeep19singh/icc-mens-world-cup-2023). We used the first 32 ODIs out of 48 of this world cup for predicting results of the rest of the matches and other statistics!!

After creating the models, we created an api for every task which makes use of the underlying models to make predictions. The details of these apis are given below as well.

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

E. upcoming_semis.csv


## Objectives

The primary objectives of this data mining project are as follows:

1. *Data Scrapping*: We used BeautifulSoup and Requests library for scrapping the data from ESPN Cricinfo for getting relevant data for making predictions for the ICC world cup 2023.
2. *Data Preprocessing*: Data preprocessing refers to the initial step in data analysis, where raw data is cleaned, transformed, and organized to make it suitable for further analysis and modeling. It involves handling missing values, dealing with outliers, standardizing data formats, and encoding categorical variables, ensuring the data is usable and consistent before applying data mining or machine learning techniques. Our dataset had 8,55,969 instances of data having 73 features, out of which we dropped few features.
3. *Exploratory Data Analysis (EDA)*: Exploratory Data Analysis (EDA) is visually and statistically analyzing a dataset to summarize its main characteristics, gain insights, and uncover patterns or anomalies. It involves techniques such as data visualization, summary statistics, and correlation analysis to understand better the data's structure, relationships between variables, and potential areas of further investigation in data analysis and modeling tasks. To visualize and understand the data, we have made boxplots, count plots, pie charts, bar graphs, point plots, violin plots,  etc.
4. *Feature Engineering*: Feature Engineering is the process of creating new, relevant, and informative features or transformations of existing ones from raw data to improve the performance of machine learning models. It involves selecting, combining, or transforming variables better to represent the underlying patterns and relationships in the data, ultimately enhancing the model's ability to make accurate predictions or classifications. Here we had a lot of scrapped data to work with, which required new features such as overs_left or runs_left and many more to be extracted from the available data.
5. *Model Selection:*: Model Selection is the process of choosing the most suitable machine learning algorithm or model from a set of candidate models to best address a specific problem or task based on factors like performance, accuracy, and generalization ability. We implemented Logistic Regression, Artificial Nueral Network Model, Support Vector Machine Model, SGD Classifier, Random Forest Classifier.
6. *Model Training and Tuning*: Model Training and Tuning involves training a machine learning model on a dataset to learn patterns and relationships and then optimizing the model's hyperparameters to achieve the best possible predictive performance. This iterative process aims to find the most accurate and effective model configuration for a specific task. We have used gradient descent algorithm to perform model tuning.
7. *Evaluation Metrics*: Evaluation metrics are quantitative measures used to assess the performance or accuracy of a predictive model or algorithm by quantifying the degree of its predictions' agreement with the actual observed outcomes. These metrics help understand how well a model performs its intended task, whether predicting values, classifying data, or solving other machine learning and data mining tasks. Common evaluation metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), accuracy, precision, recall, and F1-score, depending on the specific task and context.
