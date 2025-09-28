import numpy as np
import pandas as pd
import json
import ast
import os
import unicodedata
import itertools
import streamlit as st
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

chrome_options = Options()
chrome_options.add_argument("--headless")  # required for Streamlit Cloud
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

#driver = webdriver.Chrome()

root = os.getcwd() + '/'

#df = pd.read_csv(f'/Users/scini/Documents - Local/PL Event Data.csv')
df = pd.read_parquet(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/PL_Event_Data.parquet')
#formation_df = pd.read_csv(f'/Users/scini/Documents - Local/PL Formations.csv')
formation_df = pd.read_parquet(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/PL_Formations.parquet')
#referee_df = pd.read_csv(f'/Users/scini/Documents - Local/RefereeStatistics.csv')
referee_df = pd.read_csv(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/RefereeStatistics.csv')
#team_data_df = pd.read_csv(f'/Users/scini/Documents - Local/PL Team Data.csv')
team_data_df = pd.read_csv(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/PL%20Team%20Data.csv')
#teamId_mapping_df = pd.read_csv(f'/Users/scini/Documents - Local/TeamIDMapping.csv')
teamId_mapping_df = pd.read_csv(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/TeamIDMapping.csv')
#metrics_file = pd.read_csv(f'{root}Final FBRef All Leagues.csv')
metrics_file = pd.read_parquet(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/Final_FBRef_All_Leagues.parquet')
#fixture_df = pd.read_csv(f'/Users/scini/Documents - Local/PL Upcoming Fixtures.csv')
fixture_df = pd.read_csv(f'https://github.com/ITFCAnalytics/Fouls-Model/raw/8adcbe0b4615610dfff0cb27d8f1c9cf220be881/PL%20Upcoming%20Fixtures.csv')

# select fixture to scrape lineups from
unique_fixtures = fixture_df['Match'].sort_values().unique()
fixture_filter = st.selectbox('Select a fixture:', unique_fixtures, index=0)

match_id = fixture_df[fixture_df['Match'] == fixture_filter]['match_id'].values[0]
season = '2025-2026'

### WhoScored Season Scraper with retries and incremental updates

def load_formation_ref_data(match_id, season, max_retries=1, retry_delay=10):
    """
    Scrape WhoScored formation and referee data with retries and incremental updates.
    """
    full_df = pd.DataFrame()

    url = f'https://www.whoscored.com/matches/{match_id}/live/england-premier-league-{season}'
    print(f'Currently working on {url}')

    success = False

    for attempt in range(1, max_retries + 1):
        try:
            driver.set_page_load_timeout(300)
            driver.get(url)

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            element = soup.select_one('script:-soup-contains("matchCentreData")')

            match_dict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])

            # Extract formation data
            home = match_dict.get("home", {})
            away = match_dict.get("away", {})

            home_formation_df = pd.json_normalize(home.get("formations", []))
            away_formation_df = pd.json_normalize(away.get("formations", []))

            home_formation_df["team"] = "home"
            home_formation_df["match_id"] = match_id

            away_formation_df["team"] = "away"
            away_formation_df["match_id"] = match_id

            referee_list = match_dict.get("referee", {})
            referee_df = pd.json_normalize(referee_list)
            referee_df['match_id'] = match_id

            # Merge formation data
            formation_df = pd.merge(home_formation_df, away_formation_df, on='match_id', how='left')
            merged_df = pd.merge(formation_df, referee_df, on='match_id', how='left')

            full_df = pd.concat([full_df, merged_df], ignore_index=True)
            full_df['Season'] = season

            print(f"Successfully processed match {match_id}")
            success = True
            break  # ✅ success, exit retry loop

        except Exception as e:
            print(f"Attempt {attempt} failed for match {match_id}: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Skipping match {match_id} after {max_retries} failed attempts")

    # 🚨 Handle failure after all retries
    if not success:
        match_row = fixture_df[fixture_df["match_id"] == match_id]
        if not match_row.empty and "start_time" in match_row.columns:
            start_time = pd.to_datetime(match_row["start_time"].iloc[0])
            check_back_time = start_time - timedelta(hours=1)
            st.error(f"Lineups not available yet. Please check back at {check_back_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.error("Lineups not available yet. Please check back closer to kickoff.")
        st.stop()  # ❌ stop the whole app here

    return full_df

full_df = load_formation_ref_data(match_id, season, max_retries=3, retry_delay=10)

def remove_accents(text):
    if pd.isna(text):
        return text  # keep NaN as is
    text = str(text)  # convert to string if not
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

player_df = df[['playerId', 'playerName']]
player_df = player_df[player_df['playerName'].notna()]

player_team_df = df[['playerId', 'playerName', 'teamId', 'teamName']]
player_team_df = df[['playerId', 'playerName', 'teamId', 'teamName']].drop_duplicates()
player_team_df = player_team_df[player_team_df['playerName'].notna()]

player_team_df["playerId"] = player_team_df["playerId"].astype(int)

reduced_df = df[['eventId', 'minute', 'second', 'teamId', 'teamName', 'half_id', 'half', 'type_id', 'type_name', 'outcome_id', 'outcome', 'x', 'y', 'endX', 'endY', 'playerId', 'playerName','isTouch', 'cardType', 'match_id', 'Season', 'InPossession_Ind', 'OutOfPossession_Ind']]

p_avgX = 100 - reduced_df[reduced_df['OutOfPossession_Ind'] == 1].groupby(['teamId', 'teamName', 'playerName', 'Season'])['x'].mean()
p_avgY = 100 - reduced_df[reduced_df['OutOfPossession_Ind'] == 1].groupby(['teamId', 'teamName', 'playerName', 'Season'])['y'].mean()

avg_pressing_position_df = pd.concat([p_avgX, p_avgY], axis=1).reset_index()
avg_pressing_position_df.columns = ['teamId', 'teamName', 'playerName', 'Season', 'Pressing_avgX', 'Pressing_avgY']

bu_avgX = reduced_df[reduced_df['InPossession_Ind'] == 1].groupby(['teamId', 'teamName', 'playerName', 'Season'])['x'].mean()
bu_avgY = reduced_df[reduced_df['InPossession_Ind'] == 1].groupby(['teamId', 'teamName', 'playerName', 'Season'])['y'].mean()

avg_buildup_position_df = pd.concat([bu_avgX, bu_avgY], axis=1).reset_index()
avg_buildup_position_df.columns = ['teamId', 'teamName', 'playerName', 'Season', 'BuildUp_avgX', 'BuildUp_avgY']

metrics_file['Player'] = metrics_file['Player'].apply(remove_accents)

# Find the max (latest) season
latest_season = metrics_file["Season"].max()

# Apply different thresholds depending on season
metrics_file = metrics_file[
    ((metrics_file["Season"] == latest_season) & (metrics_file["Min"] > 45)) |
    ((metrics_file["Season"] != latest_season) & (metrics_file["Min"] > 360))
]

defensive_metrics_df = metrics_file[['Player', 'Squad', 'Season', 'FlsPer90', 'TklWinPossPer90', 'DrbPastPer90', 'DrbTkl%']]
offensive_metrics_df = metrics_file[['Player', 'Squad', 'Season', 'FldPer90', 'AttDrbPer90']]
minutes_df = metrics_file[['Player', 'Squad', 'Season', 'MP', 'Min']]

#def calc_ewma(series):
    #return series.ewm(span=3, adjust=False).mean().iloc[-1]

# defensive_metrics_df = (
#     defensive_metrics_df
#     .sort_values(["Player", "Season"])
#     .groupby("Player")
#     .agg({
#         "FlsPer90": calc_ewma,
#         "TklWinPossPer90": calc_ewma,
#         "DrbPastPer90": calc_ewma,
#         "DrbTkl%": calc_ewma,
#         "Season": "last"   # keep latest season
#     })
#     .reset_index()
# )
#
# offensive_metrics_df = (
#     offensive_metrics_df
#     .sort_values(["Player", "Season"])
#     .groupby("Player")
#     .agg({
#         "FldPer90": calc_ewma,
#         "AttDrbPer90": calc_ewma,
#         "Season": "last"   # keep latest season
#     })
#     .reset_index()
# )
#
# team_data_df = (
#     team_data_df
#     .sort_values(["teamName", "Season"])
#     .groupby(["teamName", "teamId"])
#     .agg({
#         "Fouls_Per90": calc_ewma,
#         "Fouled_Per90": calc_ewma,
#         "Season": "last"   # keep latest season
#     })
#     .reset_index()
# )

# referee_df = (
#     referee_df
#     .sort_values(["Referee", "Season"])
#     .groupby("Referee")
#     .agg({
#         "Fouls pg": calc_ewma,
#         "Fouls/Tackles": calc_ewma,
#         "Yel pg": calc_ewma,
#         "Red pg": calc_ewma,
#         "Season": "last"   # keep latest season
#     })
#     .reset_index()
# )

def calc_weighted_features(df, group_cols, season_col="Season", 
                           weight_current=0.6, weight_prev=0.4):
    """
    Calculate weighted rolling averages per group, giving more weight to the current season.

    Parameters:
        df (pd.DataFrame): Input dataframe
        group_cols (list[str]): Columns to group by (e.g., ["Player"], ["teamName","teamId"], ["Referee"])
        season_col (str): Column name for seasons
        weight_current (float): Weight for current season
        weight_prev (float): Weight for previous season(s)

    Returns:
        pd.DataFrame: Weighted aggregated dataframe
    """

    results = []

    # detect numeric columns (metrics only)
    exclude_cols = set(group_cols + [season_col])
    metrics = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]

    # loop per group
    for _, g in df.sort_values([*group_cols, season_col]).groupby(group_cols):
        seasons = g[season_col].unique()

        for i, season in enumerate(seasons):
            current_vals = g.loc[g[season_col] == season, metrics].mean()

            if i == 0:
                # only current season available
                agg = current_vals
            else:
                prev_vals = g.loc[g[season_col] == seasons[i-1], metrics].mean()
                agg = weight_prev * prev_vals + weight_current * current_vals

            # add identifiers
            for col in group_cols:
                agg[col] = g[col].iloc[0]
            agg[season_col] = season

            results.append(agg)

    return pd.DataFrame(results).reset_index(drop=True)

defensive_metrics_df = calc_weighted_features(defensive_metrics_df, group_cols=["Player"])
offensive_metrics_df = calc_weighted_features(offensive_metrics_df, group_cols=["Player"])
team_data_df = calc_weighted_features(team_data_df, group_cols=["teamName", "teamId"])
referee_df = calc_weighted_features(referee_df, group_cols=["Referee"])

team_data_df = team_data_df.rename(columns={"Fouls_Per90": "Team_Fouls_Per90", "Fouled_Per90": "Team_Fouled_Per90"})

referee_df = referee_df[['Apps', 'Fouls pg', 'Fouls/Tackles', 'Yel pg', 'Red pg', 'Referee', 'Season']]

referee_df = referee_df.rename(columns={'Referee': 'refereeName'})

referee_df['refereeName'] = referee_df['refereeName'].astype("string")

opp_team_data_df = team_data_df.copy()

opp_team_data_df = opp_team_data_df.rename(columns={"teamId": "oppTeamId"})

def closest_players_avg_positions(teamId_1, lineup_teamId_1, teamId_2, lineup_teamId_2, team_mapping_df, current_season):
    """
    Finds closest build-up players to pressing players, using only the current and previous season.

    Args:
        teamId_1: home team ID
        lineup_teamId_1: lineup players for team 1
        teamId_2: away team ID
        lineup_teamId_2: lineup players for team 2
        team_mapping_df: DataFrame mapping teamId -> teamName
        current_season: season string (e.g. '2025-2026')
    """

    from scipy.spatial import cKDTree

    # lookup team names
    teamName_1 = team_mapping_df.loc[team_mapping_df['teamId'] == teamId_1, 'teamName'].iloc[0]
    teamName_2 = team_mapping_df.loc[team_mapping_df['teamId'] == teamId_2, 'teamName'].iloc[0]

    # determine valid seasons (current + previous)
    seasons = sorted(avg_buildup_position_df['Season'].unique())
    if current_season not in seasons:
        raise ValueError(f"{current_season} not found in data")
    current_idx = seasons.index(current_season)
    valid_seasons = seasons[max(0, current_idx - 1): current_idx + 1]

    # --- filter dataframes to valid seasons ---
    team1_buildup_df = avg_buildup_position_df[
        (avg_buildup_position_df['teamId'] == teamId_1) &
        (avg_buildup_position_df['Season'].isin(valid_seasons))
    ]
    team2_buildup_df = avg_buildup_position_df[
        (avg_buildup_position_df['teamId'] == teamId_2) &
        (avg_buildup_position_df['Season'].isin(valid_seasons))
    ]

    team1_pressing_df = avg_pressing_position_df[
        (avg_pressing_position_df['teamId'] == teamId_1) &
        (avg_pressing_position_df['Season'].isin(valid_seasons))
    ]
    team2_pressing_df = avg_pressing_position_df[
        (avg_pressing_position_df['teamId'] == teamId_2) &
        (avg_pressing_position_df['Season'].isin(valid_seasons))
    ]

    # --- collapse to seasonal averages (keep season info) ---
    team1_buildup_df = team1_buildup_df.groupby(["playerName", "teamId"], as_index=False)[["BuildUp_avgX", "BuildUp_avgY"]].mean()
    team2_buildup_df = team2_buildup_df.groupby(["playerName", "teamId"], as_index=False)[["BuildUp_avgX", "BuildUp_avgY"]].mean()
    team1_pressing_df = team1_pressing_df.groupby(["playerName", "teamId"], as_index=False)[["Pressing_avgX", "Pressing_avgY"]].mean()
    team2_pressing_df = team2_pressing_df.groupby(["playerName", "teamId"], as_index=False)[["Pressing_avgX", "Pressing_avgY"]].mean()

    # --- only include players in current lineups ---
    team1_buildup_df = team1_buildup_df[team1_buildup_df['playerName'].isin(lineup_teamId_1)]
    team1_pressing_df = team1_pressing_df[team1_pressing_df['playerName'].isin(lineup_teamId_1)]
    team2_buildup_df = team2_buildup_df[team2_buildup_df['playerName'].isin(lineup_teamId_2)]
    team2_pressing_df = team2_pressing_df[team2_pressing_df['playerName'].isin(lineup_teamId_2)]

    # --- team 2 pressing team 1 ---
    if not team1_buildup_df.empty and not team2_pressing_df.empty:
        build_coords_1 = team1_buildup_df[['BuildUp_avgX', 'BuildUp_avgY']].to_numpy()
        tree_1 = cKDTree(build_coords_1)
    
        press_coords_1 = team2_pressing_df[['Pressing_avgX', 'Pressing_avgY']].to_numpy()
        distances_1, indices_1 = tree_1.query(press_coords_1, k=1)
    
        team2_pressing_df['closest_build_player'] = team1_buildup_df.iloc[indices_1]['playerName'].values
        team2_pressing_df['distance'] = distances_1
    else:
        team2_pressing_df['closest_build_player'] = np.nan
        team2_pressing_df['distance'] = np.nan

    team2_pressing_df['teamId'] = teamId_2
    team2_pressing_df['teamName'] = teamName_2
    team2_pressing_df['oppTeamId'] = teamId_1
    team2_pressing_df['oppTeamName'] = teamName_1
    team2_pressing_df['Season'] = current_season  # force season column for join

    # --- team 1 pressing team 2 ---

    if not team2_buildup_df.empty and not team1_pressing_df.empty:
        build_coords_2 = team2_buildup_df[['BuildUp_avgX', 'BuildUp_avgY']].to_numpy()
        tree_2 = cKDTree(build_coords_2)
    
        press_coords_2 = team1_pressing_df[['Pressing_avgX', 'Pressing_avgY']].to_numpy()
        distances_2, indices_2 = tree_2.query(press_coords_2, k=1)
    
        team1_pressing_df['closest_build_player'] = team2_buildup_df.iloc[indices_2]['playerName'].values
        team1_pressing_df['distance'] = distances_2
    else:
        # if empty, create empty columns to keep structure
        team1_pressing_df['closest_build_player'] = np.nan
        team1_pressing_df['distance'] = np.nan

    team1_pressing_df['teamId'] = teamId_1
    team1_pressing_df['teamName'] = teamName_1
    team1_pressing_df['oppTeamId'] = teamId_2
    team1_pressing_df['oppTeamName'] = teamName_2
    team1_pressing_df['Season'] = current_season  # force season column for join

    # --- combine ---
    pressing_df = pd.concat([team1_pressing_df, team2_pressing_df], ignore_index=True)

    pressing_df['closest_build_player'] = pressing_df['closest_build_player'].astype("string")

    return pressing_df

def reverse_closest_players_avg_positions(teamId_1, lineup_teamId_1, teamId_2, lineup_teamId_2, team_mapping_df, current_season):
    """
    Finds closest pressing players to build up players, using only the current and previous season.

    Args:
        teamId_1: home team ID
        lineup_teamId_1: lineup players for team 1
        teamId_2: away team ID
        lineup_teamId_2: lineup players for team 2
        team_mapping_df: DataFrame mapping teamId -> teamName
        current_season: season string (e.g. '2025-2026')
    """

    from scipy.spatial import cKDTree

    # lookup team names
    teamName_1 = team_mapping_df.loc[team_mapping_df['teamId'] == teamId_1, 'teamName'].iloc[0]
    teamName_2 = team_mapping_df.loc[team_mapping_df['teamId'] == teamId_2, 'teamName'].iloc[0]

    # determine valid seasons (current + previous)
    seasons = sorted(avg_buildup_position_df['Season'].unique())
    if current_season not in seasons:
        raise ValueError(f"{current_season} not found in data")
    current_idx = seasons.index(current_season)
    valid_seasons = seasons[max(0, current_idx - 1): current_idx + 1]

    # --- filter dataframes to valid seasons ---
    team1_buildup_df = avg_buildup_position_df[
        (avg_buildup_position_df['teamId'] == teamId_1) &
        (avg_buildup_position_df['Season'].isin(valid_seasons))
    ]
    team2_buildup_df = avg_buildup_position_df[
        (avg_buildup_position_df['teamId'] == teamId_2) &
        (avg_buildup_position_df['Season'].isin(valid_seasons))
    ]

    team1_pressing_df = avg_pressing_position_df[
        (avg_pressing_position_df['teamId'] == teamId_1) &
        (avg_pressing_position_df['Season'].isin(valid_seasons))
    ]
    team2_pressing_df = avg_pressing_position_df[
        (avg_pressing_position_df['teamId'] == teamId_2) &
        (avg_pressing_position_df['Season'].isin(valid_seasons))
    ]

    # --- collapse to seasonal averages (keep season info) ---
    team1_buildup_df = team1_buildup_df.groupby(["playerName", "teamId"], as_index=False)[["BuildUp_avgX", "BuildUp_avgY"]].mean()
    team2_buildup_df = team2_buildup_df.groupby(["playerName", "teamId"], as_index=False)[["BuildUp_avgX", "BuildUp_avgY"]].mean()
    team1_pressing_df = team1_pressing_df.groupby(["playerName", "teamId"], as_index=False)[["Pressing_avgX", "Pressing_avgY"]].mean()
    team2_pressing_df = team2_pressing_df.groupby(["playerName", "teamId"], as_index=False)[["Pressing_avgX", "Pressing_avgY"]].mean()

    # --- only include players in current lineups ---
    team1_buildup_df = team1_buildup_df[team1_buildup_df['playerName'].isin(lineup_teamId_1)]
    team1_pressing_df = team1_pressing_df[team1_pressing_df['playerName'].isin(lineup_teamId_1)]
    team2_buildup_df = team2_buildup_df[team2_buildup_df['playerName'].isin(lineup_teamId_2)]
    team2_pressing_df = team2_pressing_df[team2_pressing_df['playerName'].isin(lineup_teamId_2)]

    # --- team 2 building up team 1 ---
    if not team2_buildup_df.empty and not team1_pressing_df.empty:
        press_coords_1 = team1_pressing_df[['Pressing_avgX', 'Pressing_avgY']].to_numpy()
        tree_1 = cKDTree(press_coords_1)
    
        buildup_coords_1 = team2_buildup_df[['BuildUp_avgX', 'BuildUp_avgY']].to_numpy()
        distances_1, indices_1 = tree_1.query(buildup_coords_1, k=1)
    
        team2_buildup_df['closest_pressing_player'] = team1_pressing_df.iloc[indices_1]['playerName'].values
        team2_buildup_df['distance'] = distances_1
    else:
        team2_buildup_df['closest_pressing_player'] = np.nan
        team2_buildup_df['distance'] = np.nan

    team2_buildup_df['teamId'] = teamId_2
    team2_buildup_df['teamName'] = teamName_2
    team2_buildup_df['oppTeamId'] = teamId_1
    team2_buildup_df['oppTeamName'] = teamName_1
    team2_buildup_df['Season'] = current_season  # force season column for join

    # --- team 1 building up team 2 ---
    if not team1_buildup_df.empty and not team2_pressing_df.empty:
        press_coords_2 = team2_pressing_df[['Pressing_avgX', 'Pressing_avgY']].to_numpy()
        tree_2 = cKDTree(press_coords_2)
    
        buildup_coords_2 = team1_buildup_df[['BuildUp_avgX', 'BuildUp_avgY']].to_numpy()
        distances_2, indices_2 = tree_2.query(buildup_coords_2, k=1)
    
        team1_buildup_df['closest_pressing_player'] = team2_pressing_df.iloc[indices_2]['playerName'].values
        team1_buildup_df['distance'] = distances_2
    else:
        team1_buildup_df['closest_pressing_player'] = np.nan
        team1_buildup_df['distance'] = np.nan

    team1_buildup_df['teamId'] = teamId_1
    team1_buildup_df['teamName'] = teamName_1
    team1_buildup_df['oppTeamId'] = teamId_2
    team1_buildup_df['oppTeamName'] = teamName_2
    team1_buildup_df['Season'] = current_season  # force season column for join

    # --- combine ---
    buildup_df = pd.concat([team1_buildup_df, team2_buildup_df], ignore_index=True)

    buildup_df['closest_pressing_player'] = buildup_df['closest_pressing_player'].astype("string")

    return buildup_df

formation_mapping = {
    '4231': [
        'GK', 'RB', 'LB', 'LCM', 'RCB', 'LCB', 'RW', 'RCM', 'ST', 'AM', 'LW'
    ],
    '433': [
        'GK', 'RB', 'LB', 'DM', 'RCB', 'LCB', 'RCM', 'LCM', 'ST', 'RW', 'LW'
    ],
    '442': [
        'GK', 'RB', 'LB', 'RCM', 'RCB', 'LCB', 'RM', 'LCM', 'LST', 'RST', 'LM'
    ],
    '4411': [
        'GK', 'RB', 'LB', 'RCM', 'RCB', 'LCB', 'RM', 'LCM', 'ST', 'AM', 'LM'
    ],
    '4141': [
        'GK', 'RB', 'LB', 'DM', 'RCB', 'LCB', 'RM', 'RCM', 'ST', 'LCM', 'LM'
    ],
    '451': [
        'GK', 'RB', 'LB', 'RCM', 'RCB', 'LCB', 'RM', 'LCM', 'ST', 'CM', 'LM'
    ],
    '4132': [
        'GK', 'RB', 'LB', 'DM', 'RCB', 'LCB', 'RM', 'CM', 'RST', 'LST', 'LM'
    ],
    '4312': [
        'GK', 'RB', 'LB', 'CM', 'RCB', 'LCB', 'RCM', 'AM', 'RST', 'LST', 'LCM'
    ],
    '41212': [
        'GK', 'RB', 'LB', 'DM', 'RCB', 'LCB', 'RCM', 'AM', 'RST', 'LST', 'AM'
    ],
    '4222': [
        'GK', 'RB', 'LB', 'RCM', 'RCB', 'LCB', 'RAM', 'LCM', 'RST', 'LST', 'LAM'
    ],
    '424': [
        'GK', 'RB', 'LB', 'RCM', 'RCB', 'LCB', 'LCM', 'RW', 'RST', 'LST', 'LW'
    ],
    '4321': [
        'GK', 'RB', 'LB', 'CM', 'RCB', 'LCB', 'RCM', 'LCM', 'ST', 'RAM', 'LAM'
    ],
    '343': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'ST', 'RW', 'LW'
    ],
    '3421': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'ST', 'RAM', 'LAM'
    ],
    '3241': [
        'GK', 'RCM', 'LCM', 'LCB', 'CB', 'RCB', 'RAM', 'LAM', 'ST', 'RM', 'LM'
    ],
    '3412': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'AM', 'RST', 'LST', 'LCM'
    ],
    '3142': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'DM', 'RCM', 'RST', 'LST', 'LCM'
    ],
    '352': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'LST', 'RST', 'CM'
    ],
    '3511': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'AM', 'ST', 'CM'
    ],
    '523': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'ST', 'RW', 'LW'
    ],
    '5221': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'ST', 'RAM', 'LAM'
    ],
    '541': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RW', 'RCM', 'ST', 'LCM', 'LW'
    ],
    '5212': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'AM', 'RST', 'LST', 'LCM'
    ],
    '5122': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'DM', 'RCM', 'RST', 'LST', 'LCM'
    ],
    '532': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'RST', 'LST', 'CM'
    ],
    '5311': [
        'GK', 'RWB', 'LWB', 'LCB', 'CB', 'RCB', 'RCM', 'LCM', 'AM', 'ST', 'CM'
    ]
}

import ast

def add_team_ids_to_formations(formation_df, player_team_df):
    """
    Adds teamId_x and teamId_y to formation_df using player_team_df.
    Ensures consistent dtypes (integers) and removes whitespace issues.
    """

    # Ensure playerId is int
    player_team_df = player_team_df.copy()
    player_team_df["playerId"] = player_team_df["playerId"].astype(int)

    def parse_player_ids(value):
        """Ensure playerIds_x/y are lists of ints, even if stored as strings."""
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)  # safely parse "[1,2,3]"
                return [int(x) for x in parsed]
            except Exception:
                return []
        elif isinstance(value, list):
            return [int(x) for x in value]
        else:
            return []

    def get_team_id(player_ids):
        """Look up the teamId of the first player in the list."""
        try:
            if not player_ids:  # empty list
                return None
            first_pid = player_ids[0]  # already int
            match = player_team_df.loc[player_team_df["playerId"] == first_pid, "teamId"]
            if not match.empty:
                return int(match.iloc[0])
            return None
        except Exception as e:
            print("Error in get_team_id:", e, player_ids)
            return None

    # Normalize columns first
    formation_df = formation_df.copy()
    formation_df["playerIds_x"] = formation_df["playerIds_x"].apply(parse_player_ids)
    formation_df["playerIds_y"] = formation_df["playerIds_y"].apply(parse_player_ids)

    # Assign teamId_x and teamId_y
    formation_df["teamId_x"] = formation_df["playerIds_x"].apply(get_team_id)
    formation_df["teamId_y"] = formation_df["playerIds_y"].apply(get_team_id)

    # Clean formation names
    formation_df["formationName_x"] = formation_df["formationName_x"].astype(str).str.strip()
    formation_df["formationName_y"] = formation_df["formationName_y"].astype(str).str.strip()

    # Ensure IDs are integers (drop rows where lookup failed)
    formation_df = formation_df.dropna(subset=["teamId_x", "teamId_y"])
    formation_df["teamId_x"] = formation_df["teamId_x"].astype(int)
    formation_df["teamId_y"] = formation_df["teamId_y"].astype(int)

    return formation_df

def closest_players_formations(row, formation_mapping, player_team_df):
    """
    Given one row of the formations dataframe, find closest opponent matchups
    between the two teams based on base formation positions.
    Returns a DataFrame of both teams' players, their closest opponent, and team/opponent info.
    """

    from scipy.spatial import cKDTree
    import ast

    def parse_positions(player_ids_input, positions_input, formation_name, teamId, referee_name, match_id, Season, flip=False):
        """Convert playerIds + formationPositions into a DataFrame with coords + team info."""
    
        # Ensure player_ids_input is a list
        if isinstance(player_ids_input, str):
            player_ids = ast.literal_eval(player_ids_input)
        else:
            player_ids = player_ids_input  # already a list
    
        # Same for positions
        if isinstance(positions_input, str):
            positions = ast.literal_eval(positions_input)
        else:
            positions = positions_input
        
        pos_labels = formation_mapping[str(formation_name)]
    
        records = []
        for idx, (pid, pos) in enumerate(zip(player_ids, positions)):
            x, y = float(pos["horizontal"]), float(pos["vertical"])
            if flip:
                y = 10 - y
                x = 10 - x
    
            # look up player/team info
            player_info = player_team_df[
                (player_team_df["playerId"] == pid) & (player_team_df["teamId"] == teamId)
            ]
            if player_info.empty:
                player_info = player_team_df[player_team_df["playerId"] == pid]
                if player_info.empty:
                    print(f"⚠️ Skipping playerId {pid} (not found in player_team_df) "
                          f"[match={match_id}, team={teamId}, season={Season}]")
                    continue  # skip this player and move on
                player_info = player_info.iloc[0]
    
            records.append((pid, pos_labels[idx], x, y,
                            player_info["teamId"], player_info["teamName"],
                            referee_name, match_id, Season))
    
        return pd.DataFrame(records, 
                            columns=["playerId", "position", "x", "y", "teamId", "teamName", "refereeName", "match_id", "Season"])


    ref_name = row["name"]

    match_id = row["match_id"]

    Season = row["Season"]
    
    team_x_df = parse_positions(row["playerIds_x"], row["formationPositions_x"], row["formationName_x"], row["teamId_x"], ref_name, match_id, Season, flip=False)
    team_y_df = parse_positions(row["playerIds_y"], row["formationPositions_y"], row["formationName_y"], row["teamId_y"], ref_name, match_id, Season, flip=True)
    
    # Remove goalkeepers
    team_x_df = team_x_df[team_x_df["position"] != "GK"].copy()
    team_y_df = team_y_df[team_y_df["position"] != "GK"].copy()

    # KDTree: find closest opponents
    tree_y = cKDTree(team_y_df[["x", "y"]].to_numpy())
    distances_y, indices_y = tree_y.query(team_x_df[["x", "y"]].to_numpy(), k=1)

    team_x_df["closest_opponent_playerID"] = team_y_df.iloc[indices_y]["playerId"].values
    team_x_df["opponent_position"] = team_y_df.iloc[indices_y]["position"].values
    team_x_df["distance"] = distances_y
    team_x_df["oppTeamId"] = team_y_df.iloc[indices_y]["teamId"].values
    team_x_df["oppTeamName"] = team_y_df.iloc[indices_y]["teamName"].values

    tree_x = cKDTree(team_x_df[["x", "y"]].to_numpy())
    distances_x, indices_x = tree_x.query(team_y_df[["x", "y"]].to_numpy(), k=1)

    team_y_df["closest_opponent_playerID"] = team_x_df.iloc[indices_x]["playerId"].values
    team_y_df["opponent_position"] = team_x_df.iloc[indices_x]["position"].values
    team_y_df["distance"] = distances_x
    team_y_df["oppTeamId"] = team_x_df.iloc[indices_x]["teamId"].values
    team_y_df["oppTeamName"] = team_x_df.iloc[indices_x]["teamName"].values

    # Combine both teams
    combined_df = pd.concat([team_x_df, team_y_df], ignore_index=True)

    # Merge player names for both playerId and closest opponent
    combined_df = combined_df.merge(
        player_team_df[["playerId", "playerName"]], on="playerId", how="left"
    ).merge(
        player_team_df.rename(columns={"playerId": "closest_opponent_playerID", "playerName": "closest_opponent_playerName"}),
        on="closest_opponent_playerID",
        how="left"
    )

    combined_df = combined_df[['match_id', 'Season', 'teamId_x', 'teamName_x', 'playerId', 'playerName', 'position', 'x', 'y', 'distance', 'oppTeamId', 'oppTeamName', 'closest_opponent_playerID', 'closest_opponent_playerName', 'opponent_position', 'refereeName']]
    
    for col in combined_df.select_dtypes(include='object').columns:
        combined_df[col] = combined_df[col].apply(lambda x: x if not isinstance(x, pd.Series) else x.iloc[0])
    
    combined_df = combined_df.drop_duplicates()

    combined_df['refereeName'] = combined_df['refereeName'].astype("string")
    
    return combined_df
    
def build_player_matchup_df(
    avg_positions_df,
    formations_df,
    defensive_metrics_df,
    offensive_metrics_df,
    team_data_df,
    opp_team_data_df,
    referee_df):
    """
    Combine positional matchups, defensive metrics, offensive metrics, team metrics, and referee metrics into one DataFrame.
    Joins happen on both playerName and Season (or teamId/refereeName and Season).
    """

    combined = avg_positions_df[
        [
            "teamId",
            "teamName",
            "playerName",
            "Pressing_avgX",
            "Pressing_avgY",
            "closest_build_player",
            "distance",
            "oppTeamId",
            "oppTeamName",
            "Season",
        ]
    ]

    # --- Step 1: Merge defensive metrics for the main player (playerName + Season) ---
    combined = combined.merge(
        defensive_metrics_df.rename(columns={"Player": "playerName"})[
            ["playerName", "Season", "FlsPer90", "TklWinPossPer90", "DrbPastPer90", "DrbTkl%"]
        ],
        on=["playerName", "Season"],
        how="left",
    )

    # --- Step 2: Merge offensive metrics for the closest buildup player (playerName + Season) ---
    combined = combined.merge(
        offensive_metrics_df.rename(columns={"Player": "closest_build_player"})[
            ["closest_build_player", "Season", "FldPer90", "AttDrbPer90"]
        ],
        on=["closest_build_player", "Season"],
        how="left",
    )

    # --- Step 3: Merge offensive metrics for the closest opponent (playerName + Season) ---
    closest_opponent_df = formations_df.merge(
        offensive_metrics_df.rename(columns={"Player": "closest_opponent_playerName"})[
            ["closest_opponent_playerName", "Season", "FldPer90", "AttDrbPer90"]
        ],
        on=["closest_opponent_playerName", "Season"],
        how="left",
    )

    #print(closest_opponent_df.columns.tolist())

    # --- Step 4: Merge closest opponent info back into combined (playerName + Season) ---
    combined = combined.merge(
        closest_opponent_df[
            [
                "match_id",
                "playerName",
                "playerId",
                "position",
                "closest_opponent_playerName",
                "opponent_position",
                "FldPer90",
                "AttDrbPer90",
                "refereeName",
                "Season",
            ]
        ],
        on=["playerName", "Season"],
        how="left",
    )

    # --- Step 5: Merge team foul metrics (teamId + Season) ---
    combined = combined.merge(
        team_data_df[["teamId", "Season", "Team_Fouls_Per90"]],
        on=["teamId", "Season"],
        how="left",
    )

    # --- Step 6: Merge opponent fouled metrics (oppTeamId + Season) ---
    combined = combined.merge(
        opp_team_data_df[["oppTeamId", "Season", "Team_Fouled_Per90"]],
        on=["oppTeamId", "Season"],
        how="left",
    )

    # --- Step 7: Merge referee metrics (refereeName + Season) ---
    combined = combined.merge(
        referee_df[["refereeName", "Season", "Fouls pg", "Fouls/Tackles"]],
        on=["refereeName", "Season"],
        how="left",
    )

    #print(combined.columns.tolist())

    return combined

def build_fouled_player_matchup_df(
    avg_positions_df,
    formations_df,
    defensive_metrics_df,
    offensive_metrics_df,
    team_data_df,
    opp_team_data_df,
    referee_df):
    """
    Combine positional matchups, defensive metrics, offensive metrics, team metrics, and referee metrics into one DataFrame.
    Joins happen on both playerName and Season (or teamId/refereeName and Season).
    """

    combined = avg_positions_df[
        [
            "teamId",
            "teamName",
            "playerName",
            "BuildUp_avgX",
            "BuildUp_avgY",
            "closest_pressing_player",
            "distance",
            "oppTeamId",
            "oppTeamName",
            "Season",
        ]
    ]

    # --- Step 1: Merge offensive metrics for the main player (playerName + Season) ---
    combined = combined.merge(
        offensive_metrics_df.rename(columns={"Player": "playerName"})[
            ["playerName", "Season", "FldPer90", "AttDrbPer90"]
        ],
        on=["playerName", "Season"],
        how="left",
    )

    # --- Step 2: Merge defensive metrics for the closest pressing player (playerName + Season) ---
    combined = combined.merge(
        defensive_metrics_df.rename(columns={"Player": "closest_pressing_player"})[
            ["closest_pressing_player", "Season", "FlsPer90", "TklWinPossPer90", "DrbPastPer90", "DrbTkl%"]
        ],
        on=["closest_pressing_player", "Season"],
        how="left",
    )

    # --- Step 3: Merge defensive metrics for the closest opponent (playerName + Season) ---
    closest_opponent_df = formations_df.merge(
        defensive_metrics_df.rename(columns={"Player": "closest_opponent_playerName"})[
            ["closest_opponent_playerName", "Season", "FlsPer90", "TklWinPossPer90", "DrbPastPer90", "DrbTkl%"]
        ],
        on=["closest_opponent_playerName", "Season"],
        how="left",
    )

    #print(closest_opponent_df.columns.tolist())

    # --- Step 4: Merge closest opponent info back into combined (playerName + Season) ---
    combined = combined.merge(
        closest_opponent_df[
            [
                "match_id",
                "playerName",
                "playerId",
                "position",
                "closest_opponent_playerName",
                "opponent_position",
                "FlsPer90",
                "TklWinPossPer90",
                "DrbPastPer90",
                "DrbTkl%",
                "refereeName",
                "Season",
            ]
        ],
        on=["playerName", "Season"],
        how="left",
    )

    # --- Step 5: Merge team fouled metrics (teamId + Season) ---
    combined = combined.merge(
        team_data_df[["teamId", "Season", "Team_Fouled_Per90"]],
        on=["teamId", "Season"],
        how="left",
    )

    # --- Step 6: Merge opponent foul metrics (oppTeamId + Season) ---
    combined = combined.merge(
        opp_team_data_df[["oppTeamId", "Season", "Team_Fouls_Per90"]],
        on=["oppTeamId", "Season"],
        how="left",
    )

    # --- Step 7: Merge referee metrics (refereeName + Season) ---
    combined = combined.merge(
        referee_df[["refereeName", "Season", "Fouls pg", "Fouls/Tackles"]],
        on=["refereeName", "Season"],
        how="left",
    )

    #print(combined.columns.tolist())

    return combined

foul_actions = ['Foul', 'Fouled']

fouls_df = reduced_df[reduced_df['type_name'].isin(foul_actions)]

fouls_df['Type Count'] = fouls_df.groupby(
    ['Season', 'teamId', 'teamName', 'match_id', 'playerName', 'type_name']
)['type_name'].transform('count')

fouls_df = fouls_df[['Season', 'match_id', 'teamId', 'teamName', 'playerName', 'type_name', 'Type Count']].sort_values(['Season', 'match_id', 'teamName', 'type_name', 'Type Count', 'playerName']).drop_duplicates(['Season', 'match_id', 'teamName', 'type_name', 'Type Count', 'playerName'])

########################################################

# Get list of unique team IDs
# historical match up builder for fouls

# Get list of unique team IDs
team_ids = teamId_mapping_df['teamId'].unique()

# Generate all possible matchups (team1 vs team2)
all_matchups = list(itertools.combinations(team_ids, 2))
print(f"Total matchups: {len(all_matchups)}")

all_seasons = ['2022-2023', '2023-2024', '2024-2025', '2025-2026']

# Now loop through each matchup
all_matches_combined = []

all_formation_df = add_team_ids_to_formations(formation_df, player_team_df)

for teamId_1, teamId_2 in all_matchups:
    for season in all_seasons:
        # Filter formations for this matchup
        edited_formation_df = all_formation_df[
            ((all_formation_df["teamId_x"] == teamId_1) & (all_formation_df["teamId_y"] == teamId_2) & (all_formation_df["Season"] == season))
        ]
            
        if edited_formation_df.empty:
            continue  # skip if no formation data for this matchup
    
        row_formation = edited_formation_df.iloc[0]
    
        # Get closest players formations for this matchup
        formations_df = closest_players_formations(row_formation, formation_mapping, player_team_df)

        lineup_teamId_1 = formations_df[formations_df['teamId_x'] == teamId_1]['playerName']
        lineup_teamId_2 = formations_df[formations_df['teamId_x'] == teamId_2]['playerName']
    
        # Compute avg positions for the matchup
        avg_positions_df = closest_players_avg_positions(teamId_1, lineup_teamId_1,
                                                         teamId_2, lineup_teamId_2,
                                                         teamId_mapping_df, season)
        
        avg_positions_df = avg_positions_df[["Season", "teamId", "teamName", "playerName", "Pressing_avgX",
                                             "Pressing_avgY", "closest_build_player", "distance",
                                             "oppTeamId", "oppTeamName"]]
    
        # Build the matchup DataFrame with metrics
        combined_df_match = build_player_matchup_df(avg_positions_df, formations_df,
                                                    defensive_metrics_df, offensive_metrics_df,
                                                    team_data_df, opp_team_data_df, referee_df)
        
        # Rename columns and select final features
        combined_df_match = combined_df_match.rename(columns={
            "FldPer90_x": "closest_build_FldPer90",
            "AttDrbPer90_x": "closest_build_AttDrbPer90",
            "opponent_position": "closest_opponent_position",
            "FldPer90_y": "closest_opponent_FldPer90",
            "AttDrbPer90_y": "closest_opponent_AttDrbPer90",
            "Fouls pg": "Referee_FoulsPerGame",
            "Fouls/Tackles": "Referee_FoulsPerTackle"
        })[
            ['match_id', 'Season', 'teamId', 'teamName', 'Team_Fouls_Per90', 'playerId', 'playerName',
             'position', 'FlsPer90', 'TklWinPossPer90', 'DrbPastPer90', 'DrbTkl%',
             'oppTeamId', 'oppTeamName', 'Team_Fouled_Per90', 'closest_build_player',
             'closest_build_FldPer90', 'closest_build_AttDrbPer90', 'closest_opponent_playerName',
             'closest_opponent_position', 'closest_opponent_FldPer90', 'closest_opponent_AttDrbPer90',
             'refereeName', 'Referee_FoulsPerGame', 'Referee_FoulsPerTackle']
        ]
    
        all_matches_combined.append(combined_df_match)

# Concatenate all matchups into one big DataFrame
combined_df_all_matches = pd.concat(all_matches_combined, ignore_index=True)

#combined_df_all_matches.head(30)

# historical match up builder for fouled

# Get list of unique team IDs
reverse_team_ids = teamId_mapping_df['teamId'].unique()

# Generate all possible matchups (team1 vs team2)
reverse_all_matchups = list(itertools.combinations(reverse_team_ids, 2))
print(f"Total matchups: {len(reverse_all_matchups)}")

reverse_all_seasons = ['2022-2023', '2023-2024', '2024-2025', '2025-2026']

# Now loop through each matchup
reverse_all_matches_combined = []

reverse_all_formation_df = add_team_ids_to_formations(formation_df, player_team_df)

for teamId_1, teamId_2 in reverse_all_matchups:
    for season in reverse_all_seasons:
        # Filter formations for this matchup
        reverse_edited_formation_df = reverse_all_formation_df[
            ((reverse_all_formation_df["teamId_x"] == teamId_1) & (reverse_all_formation_df["teamId_y"] == teamId_2) & (reverse_all_formation_df["Season"] == season))
        ]
            
        if reverse_edited_formation_df.empty:
            continue  # skip if no formation data for this matchup
    
        reverse_row_formation = reverse_edited_formation_df.iloc[0]
    
        # Get closest players formations for this matchup
        reverse_formations_df = closest_players_formations(reverse_row_formation, formation_mapping, player_team_df)

        reverse_lineup_teamId_1 = reverse_formations_df[reverse_formations_df['teamId_x'] == teamId_1]['playerName']
        reverse_lineup_teamId_2 = reverse_formations_df[reverse_formations_df['teamId_x'] == teamId_2]['playerName']
    
        # Compute avg positions for the matchup
        reverse_avg_positions_df = reverse_closest_players_avg_positions(teamId_1, reverse_lineup_teamId_1,
                                                         teamId_2, reverse_lineup_teamId_2,
                                                         teamId_mapping_df, season)
        
        reverse_avg_positions_df = reverse_avg_positions_df[["Season", "teamId", "teamName", "playerName", "BuildUp_avgX",
                                             "BuildUp_avgY", "closest_pressing_player", "distance",
                                             "oppTeamId", "oppTeamName"]]
    
        # Build the matchup DataFrame with metrics
        reverse_combined_df_match = build_fouled_player_matchup_df(reverse_avg_positions_df, reverse_formations_df,
                                                    defensive_metrics_df, offensive_metrics_df,
                                                    team_data_df, opp_team_data_df, referee_df)
        
        # Rename columns and select final features
        reverse_combined_df_match = reverse_combined_df_match.rename(columns={
            "FlsPer90_x": "closest_pressing_FlsPer90",
            "TklWinPossPer90_x": "closest_pressing_TklWinPossPer90",
            "DrbPastPer90_x": "closest_pressing_DrbPastPer90",
            "DrbTkl%_x": "closest_pressing_DrbTkl%",
            "opponent_position": "closest_opponent_position",
            "FlsPer90_y": "closest_opponent_FlsPer90",
            "TklWinPossPer90_y": "closest_opponent_TklWinPossPer90",
            "DrbPastPer90_y": "closest_opponent_DrbPastPer90",
            "DrbTkl%_y": "closest_opponent_DrbTkl%",
            "Fouls pg": "Referee_FoulsPerGame",
            "Fouls/Tackles": "Referee_FoulsPerTackle"
        })[
            ['match_id', 'Season', 'teamId', 'teamName', 'Team_Fouled_Per90', 'playerId', 'playerName',
             'position', 'FldPer90', 'AttDrbPer90', 'oppTeamId', 'oppTeamName', 'Team_Fouls_Per90', 'closest_pressing_player',
             'closest_pressing_FlsPer90', 'closest_pressing_TklWinPossPer90', 'closest_pressing_DrbPastPer90',
             'closest_pressing_DrbTkl%', 'closest_opponent_playerName',
             'closest_opponent_position', 'closest_opponent_FlsPer90', 'closest_opponent_TklWinPossPer90',
             'closest_opponent_DrbPastPer90', 'closest_opponent_DrbTkl%',
             'refereeName', 'Referee_FoulsPerGame', 'Referee_FoulsPerTackle']
        ]
    
        reverse_all_matches_combined.append(reverse_combined_df_match)

# Concatenate all matchups into one big DataFrame
reverse_combined_df_all_matches = pd.concat(reverse_all_matches_combined, ignore_index=True)

#reverse_combined_df_all_matches.head(30)

def prepare_future_features_and_X(future_df, trained_feature_columns, verbose=True):
    """
    Prepares a future/unseen match DataFrame for prediction with the trained model.
    - Adds dummy target columns (0) to match training structure.
    - Applies any feature transformations used during training.
    - Ensures feature matrix matches trained model's columns.
    
    Returns:
        X_future: feature matrix ready for model.predict()
        future_df_prepared: the input DataFrame with dummy targets and transformations
    """
    import pandas as pd

    future_df_prepared = future_df.copy()

    # --- Step 2: Apply any transformations from training ---
    if 'DrbTkl%' in future_df_prepared.columns:
        future_df_prepared['DrbTkl%'] = 100 - future_df_prepared['DrbTkl%']

    # --- Step 3: Build feature matrix aligned with training ---
    X_future = future_df_prepared.copy()

    # Add missing columns (set to 0)
    for col in trained_feature_columns:
        if col not in X_future.columns:
            X_future[col] = 0

    # Drop extra columns not used in training
    X_future = X_future[trained_feature_columns]

    if verbose:
        print(f"[prepare_future_features_and_X] Prepared {X_future.shape[0]} rows x {X_future.shape[1]} features")
    
    return X_future, future_df_prepared

def reverse_prepare_future_features_and_X(future_df, trained_feature_columns, verbose=True):
    """
    Prepares a future/unseen match DataFrame for prediction with the trained model.
    - Adds dummy target columns (0) to match training structure.
    - Applies any feature transformations used during training.
    - Ensures feature matrix matches trained model's columns.
    
    Returns:
        X_future: feature matrix ready for model.predict()
        future_df_prepared: the input DataFrame with dummy targets and transformations
    """
    import pandas as pd

    future_df_prepared = future_df.copy()

    # --- Step 2: Apply any transformations from training ---
    if 'closest_pressing_DrbTkl%' in future_df_prepared.columns:
        future_df_prepared['closest_pressing_DrbTkl%'] = 100 - future_df_prepared['closest_pressing_DrbTkl%']
    if 'closest_opponent_DrbTkl%' in future_df_prepared.columns:
        future_df_prepared['closest_opponent_DrbTkl%'] = 100 - future_df_prepared['closest_opponent_DrbTkl%']

    # --- Step 3: Build feature matrix aligned with training ---
    X_future = future_df_prepared.copy()

    # Add missing columns (set to 0)
    for col in trained_feature_columns:
        if col not in X_future.columns:
            X_future[col] = 0

    # Drop extra columns not used in training
    X_future = X_future[trained_feature_columns]

    if verbose:
        print(f"[reverse_prepare_future_features_and_X] Prepared {X_future.shape[0]} rows x {X_future.shape[1]} features")
    
    return X_future, future_df_prepared

def merge_features_and_target(combined_df, fouls_df, verbose=True):
    """
    Merge feature table (combined_df) with foul targets (fouls_df).
    - For type_name == 'Foul': merge on playerId/match_id → Type_Count_Foul
    - For type_name == 'Fouled': merge separately onto closest_build_playerName and closest_opponent_playerName
      → Type_Count_Fouled_Build and Type_Count_Fouled_Opp
    """
    import pandas as pd

    X = combined_df.copy()
    y = fouls_df.copy()

    if 'match_id' in X.columns and 'match_id' in y.columns:
        try:
            X['match_id'] = X['match_id'].astype(int)
            y['match_id'] = y['match_id'].astype(int)
        except Exception:
            X['match_id'] = X['match_id'].astype(str).str.strip()
            y['match_id'] = y['match_id'].astype(str).str.strip()

    # --- Foul counts (main player) ---
    foul_df = y[y['type_name'] == 'Foul']
    if {'playerId','match_id'}.issubset(X.columns) and {'playerId','match_id'}.issubset(foul_df.columns):
        X = X.merge(
            foul_df[['playerId','match_id','Type Count']].rename(columns={'Type Count':'Type_Count_Foul'}),
            on=['playerId','match_id'], how='left'
        )
    elif 'playerName' in X.columns and 'playerName' in foul_df.columns:
        X = X.merge(
            foul_df[['playerName','match_id','Type Count']].rename(columns={'Type Count':'Type_Count_Foul'}),
            on=['playerName','match_id'], how='left'
        )

    # --- Fouled counts for closest_build_player ---
    fouled_df = y[y['type_name'] == 'Fouled']
    if 'closest_build_player' in X.columns and 'playerName' in fouled_df.columns:
        X = X.merge(
            fouled_df[['playerName','match_id','Type Count']].rename(columns={
                'playerName':'closest_build_player','Type Count':'Type_Count_Fouled_Build'
            }),
            on=['closest_build_player','match_id'], how='left'
        )

    # --- Fouled counts for closest_opponent_player ---
    if 'closest_opponent_playerName' in X.columns and 'playerName' in fouled_df.columns:
        X = X.merge(
            fouled_df[['playerName','match_id','Type Count']].rename(columns={
                'playerName':'closest_opponent_playerName','Type Count':'Type_Count_Fouled_Opp'
            }),
            on=['closest_opponent_playerName','match_id'], how='left'
        )

    # --- Ensure all target columns exist ---
    for col in ['Type_Count_Foul','Type_Count_Fouled_Build','Type_Count_Fouled_Opp']:
        if col not in X.columns:
            X[col] = 0
        else:
            X[col] = X[col].fillna(0).astype(int)

    if verbose:
        print(f"[merge_features_and_target] after merge: {len(X)} rows, "
              f"{X[['Type_Count_Foul','Type_Count_Fouled_Build','Type_Count_Fouled_Opp']].notna().sum().to_dict()} rows with targets")

    return X

def reverse_merge_features_and_target(combined_df, fouls_df, verbose=True):
    """
    Merge feature table (combined_df) with foul targets (fouls_df).
    - For type_name == 'Foul': merge on playerId/match_id → Type_Count_Foul
    - For type_name == 'Fouled': merge separately onto closest_build_playerName and closest_opponent_playerName
      → Type_Count_Fouled_Build and Type_Count_Fouled_Opp
    """
    import pandas as pd

    X = combined_df.copy()
    y = fouls_df.copy()

    if 'match_id' in X.columns and 'match_id' in y.columns:
        try:
            X['match_id'] = X['match_id'].astype(int)
            y['match_id'] = y['match_id'].astype(int)
        except Exception:
            X['match_id'] = X['match_id'].astype(str).str.strip()
            y['match_id'] = y['match_id'].astype(str).str.strip()

    # --- Foul counts (main player) ---
    fouled_df = y[y['type_name'] == 'Fouled']
    if {'playerId','match_id'}.issubset(X.columns) and {'playerId','match_id'}.issubset(fouled_df.columns):
        X = X.merge(
            fouled_df[['playerId','match_id','Type Count']].rename(columns={'Type Count':'Type_Count_Fouled'}),
            on=['playerId','match_id'], how='left'
        )
    elif 'playerName' in X.columns and 'playerName' in fouled_df.columns:
        X = X.merge(
            fouled_df[['playerName','match_id','Type Count']].rename(columns={'Type Count':'Type_Count_Fouled'}),
            on=['playerName','match_id'], how='left'
        )

    # --- Fouled counts for closest_build_player ---
    foul_df = y[y['type_name'] == 'Foul']
    if 'closest_pressing_player' in X.columns and 'playerName' in foul_df.columns:
        X = X.merge(
            foul_df[['playerName','match_id','Type Count']].rename(columns={
                'playerName':'closest_pressing_player','Type Count':'Type_Count_Foul_Build'
            }),
            on=['closest_pressing_player','match_id'], how='left'
        )

    # --- Fouled counts for closest_opponent_player ---
    if 'closest_opponent_playerName' in X.columns and 'playerName' in foul_df.columns:
        X = X.merge(
            foul_df[['playerName','match_id','Type Count']].rename(columns={
                'playerName':'closest_opponent_playerName','Type Count':'Type_Count_Foul_Opp'
            }),
            on=['closest_opponent_playerName','match_id'], how='left'
        )

    # --- Ensure all target columns exist ---
    for col in ['Type_Count_Fouled','Type_Count_Foul_Build','Type_Count_Foul_Opp']:
        if col not in X.columns:
            X[col] = 0
        else:
            X[col] = X[col].fillna(0).astype(int)

    if verbose:
        print(f"[merge_features_and_target] after merge: {len(X)} rows, "
              f"{X[['Type_Count_Fouled','Type_Count_Foul_Build','Type_Count_Foul_Opp']].notna().sum().to_dict()} rows with targets")

    return X

def prepare_training_data(merged_df, verbose=True):
    """
    Prepares feature matrix X and target matrix y for model training.
    - X contains all input features from combined_df
    - y contains three separate foul count targets:
        ['Type_Count_Foul', 'Type_Count_Fouled_Build', 'Type_Count_Fouled_Opp']
    """
    import pandas as pd

    df = merged_df.copy()

    # --- Target variables ---
    target_cols = ['Type_Count_Foul', 'Type_Count_Fouled_Build', 'Type_Count_Fouled_Opp']
    y = df[target_cols]

    # --- Feature matrix (drop identifiers and targets) ---
    drop_cols = ['Season','playerId','playerName','match_id',
                 'closest_opponent_playerName','teamName',
                 'oppTeamName','teamId','position',
                 'oppTeamId','closest_build_player',
                 'closest_opponent_position','refereeName'] + target_cols

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    if verbose:
        print(f"[prepare_training_data] features: {len(feature_cols)} cols, "
              f"targets: {target_cols}, n_samples={len(y)}")
        print(f"Sample features: {feature_cols[:10]}")

    return X, y

def reverse_prepare_training_data(merged_df, verbose=True):
    """
    Prepares feature matrix X and target matrix y for model training.
    - X contains all input features from combined_df
    - y contains three separate foul count targets:
        ['Type_Count_Fouled', 'Type_Count_Foul_Build', 'Type_Count_Foul_Opp']
    """
    import pandas as pd

    df = merged_df.copy()

    # --- Target variables ---
    target_cols = ['Type_Count_Fouled', 'Type_Count_Foul_Build', 'Type_Count_Foul_Opp']
    y = df[target_cols]

    # --- Feature matrix (drop identifiers and targets) ---
    drop_cols = ['Season','playerId','playerName','match_id',
                 'closest_opponent_playerName','teamName',
                 'oppTeamName','teamId','position',
                 'oppTeamId','closest_pressing_player',
                 'closest_opponent_position','refereeName'] + target_cols

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    if verbose:
        print(f"[prepare_training_data] features: {len(feature_cols)} cols, "
              f"targets: {target_cols}, n_samples={len(y)}")
        print(f"Sample features: {feature_cols[:10]}")

    return X, y

def train_and_evaluate_model(X, y, test_size=0.2, random_state=42, verbose=True):
    """
    Trains a multi-output regression model to predict:
      - Type_Count_Foul
      - Type_Count_Fouled_Build
      - Type_Count_Fouled_Opp

    Returns the trained model and evaluation metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    import numpy as np

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Define base model ---
    base_model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    # --- Wrap in multi-output regressor ---
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test)

    # --- Metrics for each target ---
    metrics = {}
    for i, col in enumerate(y.columns):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        metrics[col] = {"RMSE": rmse, "R2": r2}

    if verbose:
        print("[train_and_evaluate_model] Evaluation metrics:")
        for col, vals in metrics.items():
            print(f"  {col}: RMSE={vals['RMSE']:.3f}, R2={vals['R2']:.3f}")

    return model, metrics

### run the model using test data and get a predicted value for each target column ###

merged_df = merge_features_and_target(combined_df_all_matches, fouls_df, verbose=True)

merged_df['DrbTkl%'] = 100 - merged_df['DrbTkl%']

X, y = prepare_training_data(merged_df, verbose=True)

# --- Train the model ---
model, metrics = train_and_evaluate_model(X, y, test_size=0.2, random_state=42, verbose=True)

# --- Generate predictions ---
y_pred = model.predict(X)

# --- Convert predictions to DataFrame ---
y_pred_df = pd.DataFrame(
    y_pred,
    columns=['Pred_Type_Count_Foul', 'Pred_Type_Count_Fouled_Build', 'Pred_Type_Count_Fouled_Opp']
)

# --- Add predictions and differences to merged_df ---
merged_with_preds = merged_df.copy().reset_index(drop=True)
merged_with_preds = pd.concat([merged_with_preds, y_pred_df.reset_index(drop=True)], axis=1)

merged_with_preds['Diff_Type_Count_Foul'] = merged_with_preds['Pred_Type_Count_Foul'] - merged_with_preds['Type_Count_Foul']
merged_with_preds['Diff_Type_Count_Fouled_Build'] = merged_with_preds['Pred_Type_Count_Fouled_Build'] - merged_with_preds['Type_Count_Fouled_Build']
merged_with_preds['Diff_Type_Count_Fouled_Opp'] = merged_with_preds['Pred_Type_Count_Fouled_Opp'] - merged_with_preds['Type_Count_Fouled_Opp']

### fouled model training

reverse_merged_df = reverse_merge_features_and_target(reverse_combined_df_all_matches, fouls_df, verbose=True)

reverse_merged_df['closest_pressing_DrbTkl%'] = 100 - reverse_merged_df['closest_pressing_DrbTkl%']
reverse_merged_df['closest_opponent_DrbTkl%'] = 100 - reverse_merged_df['closest_opponent_DrbTkl%']

reverse_X, reverse_y = reverse_prepare_training_data(reverse_merged_df, verbose=True)

# --- Train the model ---
reverse_model, reverse_metrics = train_and_evaluate_model(reverse_X, reverse_y, test_size=0.2, random_state=42, verbose=True)

# --- Generate predictions ---
reverse_y_pred = reverse_model.predict(reverse_X)

# --- Convert predictions to DataFrame ---
reverse_y_pred_df = pd.DataFrame(
    reverse_y_pred,
    columns=['Pred_Type_Count_Fouled', 'Pred_Type_Count_Foul_Build', 'Pred_Type_Count_Foul_Opp']
)

# --- Add predictions and differences to merged_df ---
reverse_merged_with_preds = reverse_merged_df.copy().reset_index(drop=True)
reverse_merged_with_preds = pd.concat([reverse_merged_with_preds, reverse_y_pred_df.reset_index(drop=True)], axis=1)

reverse_merged_with_preds['Diff_Type_Count_Fouled'] = reverse_merged_with_preds['Pred_Type_Count_Fouled'] - reverse_merged_with_preds['Type_Count_Fouled']
reverse_merged_with_preds['Diff_Type_Count_Foul_Build'] = reverse_merged_with_preds['Pred_Type_Count_Foul_Build'] - reverse_merged_with_preds['Type_Count_Foul_Build']
reverse_merged_with_preds['Diff_Type_Count_Foul_Opp'] = reverse_merged_with_preds['Pred_Type_Count_Foul_Opp'] - reverse_merged_with_preds['Type_Count_Foul_Opp']

# single match up builder for fouls

teamId_1 = fixture_df[fixture_df['match_id'] == match_id]['homeTeamId'].values[0]
teamId_2 = fixture_df[fixture_df['match_id'] == match_id]['awayTeamId'].values[0]

formation_df = add_team_ids_to_formations(full_df, player_team_df)

row_formation = formation_df.iloc[0]

# Get closest players formations for this matchup
formations_df = closest_players_formations(row_formation, formation_mapping, player_team_df)

lineup_teamId_1 = formations_df[formations_df['teamId_x'] == teamId_1]['playerName']
lineup_teamId_2 = formations_df[formations_df['teamId_x'] == teamId_2]['playerName']

# Compute avg positions for the matchup
avg_positions_df = closest_players_avg_positions(teamId_1, lineup_teamId_1,
                                                 teamId_2, lineup_teamId_2,
                                                 teamId_mapping_df, season)

avg_positions_df = avg_positions_df[["Season", "teamId", "teamName", "playerName", "Pressing_avgX",
                                     "Pressing_avgY", "closest_build_player", "distance",
                                     "oppTeamId", "oppTeamName"]]

# Build the matchup DataFrame with metrics
combined_df_match = build_player_matchup_df(avg_positions_df, formations_df,
                                            defensive_metrics_df, offensive_metrics_df,
                                            team_data_df, opp_team_data_df, referee_df)

# Rename columns and select final features
combined_df_match = combined_df_match.rename(columns={
    "FldPer90_x": "closest_build_FldPer90",
    "AttDrbPer90_x": "closest_build_AttDrbPer90",
    "opponent_position": "closest_opponent_position",
    "FldPer90_y": "closest_opponent_FldPer90",
    "AttDrbPer90_y": "closest_opponent_AttDrbPer90",
    "Fouls pg": "Referee_FoulsPerGame",
    "Fouls/Tackles": "Referee_FoulsPerTackle"
})[
    ['match_id', 'Season', 'teamId', 'teamName', 'Team_Fouls_Per90', 'playerId', 'playerName',
     'position', 'FlsPer90', 'TklWinPossPer90', 'DrbPastPer90', 'DrbTkl%',
     'oppTeamId', 'oppTeamName', 'Team_Fouled_Per90', 'closest_build_player',
     'closest_build_FldPer90', 'closest_build_AttDrbPer90', 'closest_opponent_playerName',
     'closest_opponent_position', 'closest_opponent_FldPer90', 'closest_opponent_AttDrbPer90',
     'refereeName', 'Referee_FoulsPerGame', 'Referee_FoulsPerTackle']
]

single_df_match = []

single_df_match = combined_df_match

# single match up builder for fouled


# Compute avg positions for the matchup
reverse_avg_positions_df = reverse_closest_players_avg_positions(teamId_1, lineup_teamId_1,
                                                 teamId_2, lineup_teamId_2,
                                                 teamId_mapping_df, season)

reverse_avg_positions_df = reverse_avg_positions_df[["Season", "teamId", "teamName", "playerName", "BuildUp_avgX",
                                     "BuildUp_avgY", "closest_pressing_player", "distance",
                                     "oppTeamId", "oppTeamName"]]

# Build the matchup DataFrame with metrics
reverse_combined_df_match = build_fouled_player_matchup_df(reverse_avg_positions_df, formations_df,
                                            defensive_metrics_df, offensive_metrics_df,
                                            team_data_df, opp_team_data_df, referee_df)

# Rename columns and select final features
reverse_combined_df_match = reverse_combined_df_match.rename(columns={
    "FlsPer90_x": "closest_pressing_FlsPer90",
    "TklWinPossPer90_x": "closest_pressing_TklWinPossPer90",
    "DrbPastPer90_x": "closest_pressing_DrbPastPer90",
    "DrbTkl%_x": "closest_pressing_DrbTkl%",
    "opponent_position": "closest_opponent_position",
    "FlsPer90_y": "closest_opponent_FlsPer90",
    "TklWinPossPer90_y": "closest_opponent_TklWinPossPer90",
    "DrbPastPer90_y": "closest_opponent_DrbPastPer90",
    "DrbTkl%_y": "closest_opponent_DrbTkl%",
    "Fouls pg": "Referee_FoulsPerGame",
    "Fouls/Tackles": "Referee_FoulsPerTackle"
})[
    ['match_id', 'Season', 'teamId', 'teamName', 'Team_Fouled_Per90', 'playerId', 'playerName',
     'position', 'FldPer90', 'AttDrbPer90', 'oppTeamId', 'oppTeamName', 'Team_Fouls_Per90', 'closest_pressing_player',
     'closest_pressing_FlsPer90', 'closest_pressing_TklWinPossPer90', 'closest_pressing_DrbPastPer90',
     'closest_pressing_DrbTkl%', 'closest_opponent_playerName',
     'closest_opponent_position', 'closest_opponent_FlsPer90', 'closest_opponent_TklWinPossPer90',
     'closest_opponent_DrbPastPer90', 'closest_opponent_DrbTkl%',
     'refereeName', 'Referee_FoulsPerGame', 'Referee_FoulsPerTackle']
]

reverse_single_df_match = []

reverse_single_df_match = reverse_combined_df_match

### predict fouls

# Call the new function
X_future, future_with_dummy_targets = prepare_future_features_and_X(
    single_df_match,
    trained_feature_columns=X.columns,  # ensure same order as training
    verbose=True
)

# Generate predictions
y_future_pred = model.predict(X_future)

# Convert predictions to DataFrame
y_future_pred_df = pd.DataFrame(
    y_future_pred,
    columns=['Pred_Type_Count_Foul', 'Pred_Type_Count_Fouled_Build', 'Pred_Type_Count_Fouled_Opp']
)

# Merge predictions back with original future match info
future_with_preds = pd.concat([future_with_dummy_targets.reset_index(drop=True),
                               y_future_pred_df.reset_index(drop=True)], axis=1)

minimised_future_with_preds = future_with_preds[['teamName', 'playerName', 'position', 'Pred_Type_Count_Foul', 'oppTeamName', 'closest_build_player', 'Pred_Type_Count_Fouled_Build', 'closest_opponent_playerName', 'closest_opponent_position', 'Pred_Type_Count_Fouled_Opp']]
minimised_future_with_preds = future_with_preds.sort_values('Pred_Type_Count_Foul', ascending=False)

### predict fouled

# --- Step 1: Prepare features for the future game ---
reverse_X_future, reverse_future_with_dummy_targets = reverse_prepare_future_features_and_X(
    reverse_single_df_match,
    trained_feature_columns=reverse_X.columns,  # use correct feature schema
    verbose=True
)

# --- Step 2: Generate predictions with the correct model ---
reverse_y_future_pred = reverse_model.predict(reverse_X_future)

# --- Step 3: Convert predictions to DataFrame ---
reverse_y_future_pred_df = pd.DataFrame(
    reverse_y_future_pred,
    columns=['Pred_Type_Count_Fouled', 'Pred_Type_Count_Foul_Build', 'Pred_Type_Count_Foul_Opp']
)

# --- Step 4: Merge predictions with match info ---
reverse_future_with_preds = pd.concat(
    [reverse_future_with_dummy_targets.reset_index(drop=True),
     reverse_y_future_pred_df.reset_index(drop=True)], axis=1
)

minimised_reverse_future_with_preds = reverse_future_with_preds[['teamName', 'playerName', 'position', 'Pred_Type_Count_Fouled', 'oppTeamName', 'closest_pressing_player', 'Pred_Type_Count_Foul_Build', 'closest_opponent_playerName', 'closest_opponent_position', 'Pred_Type_Count_Foul_Opp']]
minimised_reverse_future_with_preds = reverse_future_with_preds.sort_values('Pred_Type_Count_Fouled', ascending=False)

# call the foul and fouled df's

st.subheader(f'Fouls Model - {fixture_filter}')
st.table(minimised_future_with_preds)
with st.expander('Full Fouls Model:', expanded=False):
    st.table(future_with_preds)

st.subheader(f'Fouled Model - {fixture_filter}')
st.table(minimised_reverse_future_with_preds)
with st.expander('Full Fouled Model:', expanded=False):
    st.table(reverse_future_with_preds)