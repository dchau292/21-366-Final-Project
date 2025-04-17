# 21-366-Final-Project
This project predicts NBA players' stat lines (Points, Rebounds, Assists) using a combination of:
- **Recent performance**
- **Historical performance vs upcoming opponent**
- **Opponent defensive metrics**
- **Principal Component Analysis (PCA)**
- **OpenAI's GPT-4 for natural language reasoning**

---

This script:
1. Fetches a player’s last 10 games
2. Fetches their last 10 games against the upcoming opponent (last 5 seasons)
3. Uses PCA to reduce statistical dimensions
4. Builds a semantic vector search with FAISS
5. Feeds contextual summaries into GPT-4 for prediction
6. Returns a predicted stat line with reasoning

---

How to run:
In order for the code to run, create a .env file in the root directory:
  OPENAI_API_KEY=your-openai-key-here

---

Modify the following variables inside project.py:
player = "Luka Doncic"        # NBA player name
opponent = "OKC"              # Opponent team abbreviation
upcoming_game = "Home"        # "Home" or "Away"
season = "2024-25"            # Season
