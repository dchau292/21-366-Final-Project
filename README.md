# 21-366-Final-Project

## Porject Overview
This project explores the intersection of sports analytics and generative AI by predicting the performance of NBA players using a combination of real-time statistics, historical data, dimensionality reduction, and language modeling.

The primary goal is to generate a predicted stat line — Points, Rebounds, and Assists — for a given player in their upcoming game, using both quantitative evidence and natural language reasoning. This is achieved through a Retrieval-Augmented Generation (RAG) pipeline that informs a GPT-4 model with:
- **Recent performance**
- **Historical performance vs upcoming opponent**
- **Opponent defensive metrics**
- **Principal Component Analysis (PCA)**
- **OpenAI's GPT-4 for natural language reasoning**

Combining traditional analytics with modern AI techniques forms a Retrieval-Augmented Generation (RAG) pipeline that enriches GPT-4's prediction capabilities with factual game data.
![image](https://github.com/user-attachments/assets/4d695074-e861-4baf-8139-034fe7108c75)


---

This script:
1. Fetches a player’s last 10 games
2. Fetches their last 10 games against the upcoming opponent (last 5 seasons)
3. Uses PCA to reduce statistical dimensions (PTS, REB, AST)
4. Builds a semantic vector search with FAISS
5. Feeds contextual summaries into GPT-4 for prediction
6. Returns a predicted stat line with reasoning

---

## Install Directions
1. Clone the repo
2. Set up environment by creating .env file in root directory to store API key securely
3. Install dependencies <code>pip install -r requirements.txt </code>
4. Modify these input variables in predictor.py to configure the analysis:
player = "Luka Doncic"       # Player to predict
opponent = "OKC"             # Opponent team abbreviation
upcoming_game = "Home"       # "Home" or "Away"
season = "2024-25"           # NBA season format


---

Modify the following variables inside project.py:
player = "Luka Doncic"        # NBA player name
opponent = "OKC"              # Opponent team abbreviation
upcoming_game = "Home"        # "Home" or "Away"
season = "2024-25"            # Season
