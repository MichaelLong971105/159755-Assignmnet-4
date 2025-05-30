import pandas as pd
import sqlite3
import glob
import os

base_path = os.path.dirname(os.path.abspath(__file__))
def save_match_data(connection, cursor):
    create_match_stats = """CREATE TABLE match_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        season TEXT NOT NULL,
        ftr TEXT,
        b365h REAL,
        b365d REAL,
        b365a REAL,
        b365_prob_h REAL,
        b365_prob_d REAL,
        b365_prob_a REAL
    );"""
    cursor.execute("DROP TABLE IF EXISTS match_stats")
    cursor.execute(create_match_stats)

    team_name = {
        # England
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Newcastle": "Newcastle United",
        "West Brom": "West Bromwich Albion",
        "Sheff Utd": "Sheffield United",
        "Nott'm Forest": "Nottingham Forest",
        "Leeds": "Leeds United",
        "Brighton": "Brighton & Hove Albion",
        "Birmingham": "Birmingham City",
        "Blackburn": "Blackburn Rovers",
        "Bolton": "Bolton Wanderers",
        "Stoke": "Stoke City",
        "Wigan": "Wigan Athletic",
        "West Ham": "West Ham United",

        # France
        "PSG": "Paris Saint-Germain",
        "AS Nancy": "AS Nancy-Lorraine",
        "Arles": "AC Arles-Avignon",
        "Auxerre": "AJ Auxerre",
        "Brest": "Stade Brestois 29",
        "Caen": "Stade Malherbe Caen",
        "Lorient": "FC Lorient",
        "Nice": "OGC Nice",
        "Saint Etienne": "AS Saint-Étienne",
        "Sochaux": "FC Sochaux-Montbéliard",

        # Germany
        "Bayern Munich": "FC Bayern München",
        "Wolfsburg": "VfL Wolfsburg",
        "Monchengladbach": "Borussia Mönchengladbach",
        "Nurnberg": "1. FC Nürnberg",
        "St Pauli": "FC St. Pauli",
        "Hoffenheim": "TSG 1899 Hoffenheim",

        # Italy
        "AC Milan": "Milan",
        "Inter": "Internazionale",
        "Lazio": "SS Lazio",
        "Cagliari": "Cagliari Calcio",
        "Cesena": "AC Cesena",
        "Lecce": "US Lecce",

        # Spain
        "Atletico": "Atlético Madrid",
        "Mallorca": "RCD Mallorca",
        "Hercules": "Hércules CF",
        "Racing": "Racing Santander",
        "Sporting": "Sporting Gijón",
        "Zaragoza": "Real Zaragoza",
        "Real Madrid": "Real Madrid CF",
        "Barcelona": "FC Barcelona",
    }

    def map_team_name(name):
        return team_name.get(name, name)

    files = os.path.join(base_path, '..', 'data/match_data', '*.csv')

    dataframes = []

    for file in files:
        try:
            # Extract and format season
            raw_season = os.path.splitext(os.path.basename(file))[0].split('_')[1]  # e.g. '2020-2021'
            start_year, end_year = raw_season.split('-')
            season = f"{start_year}/{end_year[2:]}"  # e.g. '2020/21'
            df_match = pd.read_csv(file)
            df_match['Season'] = season
            df_match = df_match[['HomeTeam', 'AwayTeam', 'Season', 'FTR', 'B365H', 'B365D', 'B365A']]
            inverse = 1 / df_match[['B365H', 'B365D', 'B365A']]
            total = inverse.sum(axis=1)
            df_match['b365_prob_H'] = inverse['B365H'] / total
            df_match['b365_prob_D'] = inverse['B365D'] / total
            df_match['b365_prob_A'] = inverse['B365A'] / total
            dataframes.append(df_match)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df['HomeTeam'] = merged_df['HomeTeam'].apply(map_team_name)
    merged_df['AwayTeam'] = merged_df['AwayTeam'].apply(map_team_name)
    merged_df.columns = ['home_team', 'away_team', 'season', 'ftr', 'b365h', 'b365d', 'b365a', 'b365_prob_h',
                         'b365_prob_d', 'b365_prob_a']
    merged_df.dropna(inplace=True)
    merged_df.to_sql('match_stats', connection, if_exists='append', index=False)
    connection.commit()


def save_team_data(connection, cursor):
    files = os.path.join(base_path, '..', 'team_data', '*.csv')

    create_team_stats = """CREATE TABLE team_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        team TEXT NOT NULL,
        league TEXT NOT NULL,
        year TEXT NOT NULL,
        overall INTEGER,
        attack INTEGER,
        midfield INTEGER,
        defence INTEGER,
        players INTEGER,
        starting_xi_avg_age REAL);"""

    cursor.execute("DROP TABLE IF EXISTS team_stats")
    cursor.execute(create_team_stats)

    dataframes = []

    for file in files:
        df_team = pd.read_csv(file)
        df_team[['Team', 'League']] = df_team['Name'].str.split('\n', expand=True)
        df_team['Year'] = os.path.basename(file)[-8:-4]
        df_team.drop(columns=['Name'], inplace=True)
        df_team.rename(columns={'Starting XI average age': 'starting_xi_avg_age'}, inplace=True)
        df_team = df_team[['Team', 'League', 'Year', 'Overall', 'Attack', 'Midfield', 'Defence', 'Players', 'starting_xi_avg_age']]
        dataframes.append(df_team)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_sql('team_stats', connection, if_exists='append', index=False)
    connection.commit()


def save_player_data(connection, cursor):
    files = os.path.join(base_path, '..', 'player_data', '*.csv')

    create_player_stats = """CREATE TABLE player_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        age INTEGER,
        year TEXT NOT NULL,
        overall_rating INTEGER,
        potential INTEGER,
        best_position TEXT,
        team TEXT,
        height_cm REAL,
        weight_kg REAL,
        value REAL,
        wage REAL,
        short_passing INTEGER,
        dribbling INTEGER,
        stamina INTEGER,
        total_goalkeeping INTEGER
        );"""

    cursor.execute("DROP TABLE IF EXISTS player_stats")
    cursor.execute(create_player_stats)

    dataframes = []

    for file in files:
        df_player = pd.read_csv(file)
        df_player['name'] = df_player['Name'].str.split('\n').str[0]
        df_player['year'] = os.path.basename(file)[-8:-4]

        df_player['team'] = df_player['Team & Contract'].str.split('\n').str[0]
        df_player['contract'] = df_player['Team & Contract'].str.split('\n').str[1]

        df_player['height_cm'] = df_player['Height'].str.extract(r'(\d+)cm').astype(float)

        df_player['weight_kg'] = df_player['Weight'].str.extract(r'(\d+)kg').astype(float)

        for col in ['Stamina', 'Dribbling', 'Short passing']:
            df_player[col] = df_player[col].astype(str).str.extract(r'(\d+)').astype(float)

        df_player.rename(columns={
            'Full Name': 'full_name',
            'Age': 'age',
            'Overall rating': 'overall_rating',
            'Potential': 'potential',
            'Best position': 'best_position',
            'Value': 'value',
            'Wage': 'wage',
            'Short passing': 'short_passing',
            'Dribbling': 'dribbling',
            'Stamina': 'stamina',
            'Total goalkeeping': 'total_goalkeeping'
        }, inplace=True)

        cleaned_df = df_player[[
            'full_name', 'age', 'year', 'overall_rating', 'potential', 'best_position',
            'team', 'height_cm', 'weight_kg',
            'value', 'wage', 'short_passing', 'dribbling', 'stamina', 'total_goalkeeping'
        ]]
        dataframes.append(cleaned_df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_sql('player_stats', connection, if_exists='append', index=False)
    connection.commit()

if __name__ == '__main__':

    database_path = os.path.join(base_path, '..', 'data', 'allData.sl3')
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    save_match_data(connection, cursor)
    save_team_data(connection, cursor)
    save_player_data(connection, cursor)
    cursor.close()
    connection.close()
