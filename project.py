from nba_api.stats.endpoints import playercareerstats, PlayerGameLog, LeagueDashPtTeamDefend
from nba_api.stats.static import players
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import openai
from config import OPENAI_API_KEY

nba_api_functions = {
    "Player Stats": [
        "PlayerCareerStats", "PlayerGameLog", "PlayerGameLogs", "PlayerAwards", 
        "PlayerProfileV2", "PlayerCompare", "PlayerEstimatedMetrics", "PlayerDashboardByGeneralSplits",
        "PlayerDashboardByGameSplits", "PlayerDashboardByYearOverYear", "PlayerDashboardByShootingSplits",
        "PlayerDashboardByTeamPerformance", "PlayerDashboardByLastNGames"
    ],
    
    "Team Stats": [
        "TeamGameLog", "TeamGameLogs", "TeamYearByYearStats", "TeamDashboardByGeneralSplits",
        "TeamDashboardByShootingSplits", "TeamPlayerDashboard", "TeamPlayerOnOffSummary",
        "TeamPlayerOnOffDetails", "TeamEstimatedMetrics"
    ],

    "Box Scores": [
        "BoxScoreTraditionalV2", "BoxScoreAdvancedV2", "BoxScoreFourFactorsV2", 
        "BoxScoreMiscV2", "BoxScorePlayerTrackV2", "BoxScoreScoringV2", "BoxScoreSummaryV2",
        "BoxScoreUsageV2", "BoxScoreDefensiveV2", "BoxScoreHustleV2"
    ],

    "Game Data": [
        "PlayByPlay", "PlayByPlayV2", "PlayByPlayV3", "LeagueGameLog", "LeagueGameFinder",
        "ScoreboardV2", "GameRotation"
    ],

    "Shooting & Shot Charts": [
        "ShotChartDetail", "ShotChartLeagueWide", "ShotChartLineupDetail", "LeagueDashPlayerShotLocations",
        "LeagueDashTeamShotLocations", "LeagueDashPlayerPtShot", "LeagueDashTeamPtShot"
    ],

    "Matchups & Defensive Stats": [
        "PlayerDashPtShotDefend", "PlayerDashPtShots", "PlayerDashPtPass", "PlayerDashPtReb",
        "LeagueDashPtDefend", "LeagueDashPtStats", "LeagueDashPtTeamDefend", "TeamDashPtShots",
        "TeamDashPtPass", "TeamDashPtReb"
    ],

    "League-Wide Stats & Standings": [
        "LeagueDashPlayerStats", "LeagueDashTeamStats", "LeagueLeaders", "LeagueStandings",
        "LeagueStandingsV3", "LeagueSeasonMatchups", "LeagueDashLineups", "LeagueDashPlayerClutch",
        "LeagueDashTeamClutch", "LeagueHustleStatsPlayer", "LeagueHustleStatsTeam"
    ],

    "Draft & Rookie Data": [
        "DraftCombineDrillResults", "DraftCombinePlayerAnthro", "DraftCombineStats",
        "DraftCombineSpotShooting", "DraftCombineNonStationaryShooting", "DraftBoard",
        "DraftHistory"
    ],

    "Franchise & Team History": [
        "FranchiseHistory", "FranchiseLeaders", "FranchisePlayers", "CommonTeamYears",
        "CommonTeamRoster", "CommonAllPlayers"
    ],

    "Win Probability & Game Simulations": [
        "WinProbabilityPBP"
    ],

    "Synergy & Play Types": [
        "SynergyPlayTypes"
    ],

    "Video & Media": [
        "VideoDetails", "VideoDetailsAsset", "VideoEvents", "VideoStatus"
    ]
}
TEAM_ABBR_TO_NAME = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}
TEAM_ABBR_TO_ID = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
    'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
    'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
    'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
    'UTA': 1610612762, 'WAS': 1610612764
}

def get_last_n_games_stats(player_name, season="2024-25", n=10):
    player_info = players.find_players_by_full_name(player_name)
    if not player_info:
        raise ValueError(f"Player '{player_name}' not found.")
    player_id = player_info[0]['id']
    
    game_log = PlayerGameLog(player_id=player_id, season=season)
    df = game_log.get_data_frames()[0]

    df = df.iloc[:n]
    
    return df[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']].reset_index(drop=True)

def get_last_n_vs_team_stats(player_name, opponent_abbr, current_season="2024-25", n=10):
    player_info = players.find_players_by_full_name(player_name)
    if not player_info:
        raise ValueError(f"Player '{player_name}' not found.")
    player_id = player_info[0]['id']
    
    # Generate seasons from last 5 years
    start_year = int(current_season.split("-")[0])
    seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year - 4, start_year + 1)]

    all_games = []
    for season in seasons:
        try:
            game_log = PlayerGameLog(player_id=player_id, season=season)
            df = game_log.get_data_frames()[0]
            df['SEASON'] = season
            all_games.append(df)
        except:
            continue

    if not all_games:
        raise ValueError("No game data found.")

    combined_df = pd.concat(all_games, ignore_index=True)
    vs_team = combined_df[combined_df['MATCHUP'].str.contains(opponent_abbr, case=False)]
    vs_team = vs_team.sort_values(by='GAME_DATE', ascending=False).head(n)

    return vs_team[['SEASON', 'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']].reset_index(drop=True)

def get_combined_game_data(player_name, opponent_abbr, current_season="2024-25", n=10):
    recent_df = get_last_n_games_stats(player_name, season=current_season, n=n)
    vs_team_df = get_last_n_vs_team_stats(player_name, opponent_abbr, current_season, n=n)

    # Combine and label source
    recent_df['SOURCE'] = 'recent'
    vs_team_df['SOURCE'] = 'vs_team'

    combined = pd.concat([recent_df, vs_team_df], ignore_index=True)
    return combined

def get_pca_vector(df, n_components=3):
    features = df[['PTS', 'REB', 'AST']].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    pca_vector = pca.fit_transform(scaled).mean(axis=0)  # average over games

    return pca_vector, pca, scaler

def create_stat_summary(player_name, opponent, pca_vector, df):
    avg_stats = df[['PTS', 'REB', 'AST']].mean()
    summary = (
        f"{player_name} has averaged {avg_stats['PTS']:.1f} PTS, "
        f"{avg_stats['REB']:.1f} REB, and {avg_stats['AST']:.1f} AST in his last {len(df)} games. "
        f"PCA vector: {pca_vector}"
    )
    return summary

def build_vector_db(summaries):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(summaries)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, model, embeddings

def retrieve_similar_docs(summary, model, index, summaries, k=5):
    query_vec = model.encode([summary])
    D, I = index.search(query_vec, k)
    return [summaries[i] for i in I[0]]

# --- New Helper to fetch opponent defensive stats ---
def get_opponent_defensive_stats(opponent_abbr, season='2024-25'):
    team_id = TEAM_ABBR_TO_ID.get(opponent_abbr.upper())
    if not team_id:
        return {'d_fg_pct': 'N/A', 'ppg_allowed': 'N/A'}

    try:
        pt_def_stats = LeagueDashPtTeamDefend(season=season).get_data_frames()[0]

        # Filter for the desired team
        team_row = pt_def_stats[pt_def_stats['TEAM_ID'] == team_id]
        if team_row.empty:
            return {'d_fg_pct': 'N/A', 'ppg_allowed': 'N/A'}
        avg_dfg_pct = team_row['D_FG_PCT'].mean()

        return {
            'd_fg_pct': round(avg_dfg_pct, 3)
        }

    except Exception as e:
        print("Error fetching defensive stats from LeagueDashPtTeamDefend:", e)
        return {'d_fg_pct': 'N/A'}

# prompt generator
def generate_llm_prompt(player_name, opponent, pca_vector, retrieved_docs, home_away, last_vs_team, opp_def_stats, minutes_avg):
    docs = "\n".join(retrieved_docs)
    prompt = f"""
Player: {player_name}
Upcoming Opponent: {opponent}
Location: {'Home' if home_away == 'Home' else 'Away'}
Opponent Defensive FG%: {opp_def_stats['d_fg_pct']}

Recent Minutes/Game: {minutes_avg:.1f}
Last Performance vs {opponent}: {last_vs_team}

Performance PCA Vector: {pca_vector}

Contextual Summaries:
{docs}

Based on this statistical and contextual information, predict {player_name}'s stat line (PTS, REB, AST) for the upcoming game and explain your reasoning.
"""
    return prompt

# Integration
player = "Luka Doncic" # modify as needed
opponent = "OKC" # modify as needed
upcoming_game = "Home" # modify if upcoming game is home or away
season = "2024-25"

# Step 1: Get combined data
combined_df = get_combined_game_data(player, opponent, season, n=10)
recent_df = combined_df[combined_df['SOURCE'] == 'recent']
vs_team_df = combined_df[combined_df['SOURCE'] == 'vs_team']

# Step 2: PCA vector
pca_vector, pca_model, scaler = get_pca_vector(recent_df[['PTS', 'REB', 'AST']])

# Step 3: Stat summaries
summary = create_stat_summary(player, opponent, pca_vector, recent_df)
summaries = [summary] 

# Step 4: FAISS index
index, embedder, _ = build_vector_db(summaries)
similar_docs = retrieve_similar_docs(summary, embedder, index, summaries, k=1)

# Step 5: Extra features
minutes_avg = 0
try:
    game_log = PlayerGameLog(player_id=players.find_players_by_full_name(player)[0]['id'], season=season)
    minutes_avg = game_log.get_data_frames()[0].head(10)['MIN'].astype(float).mean()
except:
    minutes_avg = 30

opp_def_stats = get_opponent_defensive_stats(opponent, season)

last_game_vs_opp = vs_team_df.iloc[0] if not vs_team_df.empty else {'PTS': '-', 'REB': '-', 'AST': '-'}
last_vs_line = f"{last_game_vs_opp['PTS']} PTS, {last_game_vs_opp['REB']} REB, {last_game_vs_opp['AST']} AST"

# Step 6: Prompt
prompt = generate_llm_prompt(
    player, opponent, pca_vector, similar_docs,
    upcoming_game, last_vs_line, opp_def_stats, minutes_avg
)

# # Step 7: LLM Call
client = openai.OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an NBA analyst who predicts player stat lines."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content.encode('utf-8', errors='ignore').decode('utf-8'))