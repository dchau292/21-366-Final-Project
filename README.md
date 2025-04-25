# 21-366-Final-Project

## Project Overview
This project explores the intersection of sports analytics and generative AI by predicting the performance of NBA players using a combination of real-time statistics, historical data, dimensionality reduction, and language modeling.

The primary goal is to generate a predicted stat line — Points, Rebounds, and Assists — for a given player in their upcoming game, using both quantitative evidence and natural language reasoning. This is achieved through a Retrieval-Augmented Generation (RAG) pipeline that informs a GPT-4 model with:
- **Recent performance**
- **Historical performance vs upcoming opponent**
- **Opponent defensive metrics**
- **Principal Component Analysis (PCA)**
- **Memory-base Retrieval**
- **OpenAI's GPT-4 for natural language reasoning**

Combining traditional analytics with modern AI techniques forms a Retrieval-Augmented Generation (RAG) pipeline that enriches GPT-4's prediction capabilities with factual game data.
![image](https://github.com/user-attachments/assets/4d695074-e861-4baf-8139-034fe7108c75)

---

## Project Description
### RAG Pipeline Breakdown:
1. Data Retrieval
    - Recent player game logs (last 10 games)
    - Historical matchup logs vs. upcoming opponent (last 5 seasons)
    - Opponent defensive field goal %
2. Principal Component Analysis (PCA) to reduce vector size
3. Summary creation of player stat trends and matchup performance
4. Vector Indexing
    - Context summaries are embedded using <code>sentence-transformers</code>
    - Indexed and retrieved utilizing FAISS
5. Propmpt Construction
    - Summaries, defensive information, and recent trends
    - A prompt is generated to inform GPT-4 with contextual grounding
6. LLM Generation
    - GPT-4 generates a projected stat line with reasoning

---

## Install Directions
1. Clone the repo
2. Set up environment by creating .env file in root directory to store API key securely
3. Install dependencies <code>pip install -r requirements.txt </code>
4. Modify these input variables in predictor.py to configure the analysis:
</br>player = "Luka Doncic"       # Player to predict
</br>opponent = "OKC"             # Opponent team abbreviation
</br>upcoming_game = "Home"       # "Home" or "Away"
</br>season = "2024-25"           # NBA season format

---
