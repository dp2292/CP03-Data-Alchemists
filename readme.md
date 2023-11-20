# CP03-Data-Alchemists

# Installation and Run Instructions
   ## 1. Clone the Repository:
   Open a terminal or command prompt and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/dp2292/CP03-Data-Alchemists
   ```
## 2. Navigate to the Repository

Change your current working directory to the cloned repository:

```bash
cd CP03-Data-Alchemists
```
## 3. Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
## 4. Run the `api.py` File

Once the dependencies are installed, you can run the `api.py` file:

```bash
python api.py
```

# API Endpoints

## 1. Default Route

- **URL:** `/`
- **Method:** GET
- **Description:** Default route showing team members and team name.

## 2. Predict Finalists

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
## 3. Predict Winner

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
## 4. Get Playing 11

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



