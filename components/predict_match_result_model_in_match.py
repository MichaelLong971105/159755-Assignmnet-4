import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns  # Load Datasets

folder_path = '../data/in_match_predict'
save_path = "../models"

csv_files = [file for file in os.listdir(folder_path) if file.endswith('E0.csv')]

dataframes = []
column_sets = []
all_cols = set()

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        # Extract and format season
        raw_season = file.split(' ')[1]  # e.g. '2020-2021'
        start_year, end_year = raw_season.split('-')
        season = f"{start_year}/{end_year[2:]}"  # e.g. '2020/21'

        df = pd.read_csv(file_path)
        df['Season'] = season
        dataframes.append(df)

        header_df = pd.read_csv(file_path, nrows=0)
        cols = set(header_df.columns)
        column_sets.append(cols)
        all_cols.update(cols)
    except Exception as e:
        print(f"Error reading {file}: {e}")

merged_df = pd.concat(dataframes, ignore_index=True)

common_cols = set.intersection(*column_sets)
non_shared_cols = all_cols - common_cols

# Save the csv file to dataset folder
# merged_df.to_csv(os.path.join(folder_path, 'E0_merged.csv'), index=False)
# print(f"\n Merged dataset saved to: {os.path.join(folder_path, 'E0_merged.csv')}")

# merged_df.to_csv("E0_merged.csv", index=False)
print(f"The size of merged dataset is : {merged_df.shape}")

# Sort the dataset based on Division and Year

merged_df['Date'] = pd.to_datetime(merged_df['Date'], dayfirst=True)
merged_df['Year'] = merged_df['Date'].dt.year

merged_df['Div_num'] = merged_df['Div'].str.extract(r'E(\d)').astype(int)

merged_df = merged_df.sort_values(by=['Year', 'Div_num'], ascending=[True, False])

merged_df.drop(columns='Div_num', inplace=True)

# Load club stats
club_df = pd.read_csv(os.path.join(folder_path, 'club_stats.csv'))

# club_df.head()

# Check Unquie Team Name
# print("\n------Unique HomeTeams in merged_df------\n")
# print(merged_df['HomeTeam'].unique())
#
# print("\n------ Unique AwayTeams in merged_df-------\n")
# print(merged_df['AwayTeam'].unique())
#
# print("\n------Unique Clubs in club_df------\n")
# print(club_df['Club'].unique())

# Mapping dictionary from short names to full club names
team_name = {
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Sheffield United': 'Sheffield United',
    'West Ham': 'West Ham United',
    'West Brom': 'West Bromwich Albion',
    'Brighton': 'Brighton & Hove Albion',
    'Wolves': 'Wolverhampton Wanderers',
    'Spurs': 'Tottenham Hotspur',
    'Tottenham': 'Tottenham Hotspur',
    'Newcastle': 'Newcastle United',
    "Nott'm Forest": 'Nottingham Forest',
    'Leeds': 'Leeds United',
    'Norwich': 'Norwich City',
    'Luton': 'Luton Town',
    'Ipswich': 'Ipswich Town',
    'Leicester': 'Leicester City',
}


def map_team_name(name):
    return team_name.get(name, name)


# Map to full club names for HomeTeam and AwayTeam
merged_df['HomeTeam'] = merged_df['HomeTeam'].apply(map_team_name)
merged_df['AwayTeam'] = merged_df['AwayTeam'].apply(map_team_name)

# Strip any leading/trailing spaces from column names
club_df.columns = club_df.columns.str.strip()

# Prepare home stats with renamed columns
home_stats = club_df.rename(columns={
    'Club': 'HomeTeam',
    'Position': 'HPos',
    'Played': 'HPlayed',
    'Won': 'HWon',
    'Drawn': 'HDrawn',
    'Lost': 'HLost'
})

# Prepare away stats with renamed columns
away_stats = club_df.rename(columns={
    'Club': 'AwayTeam',
    'Position': 'APos',
    'Played': 'APlayed',
    'Won': 'AWon',
    'Drawn': 'ADrawn',
    'Lost': 'ALost'
})

# Merge df with the home_stats and away_stats
merged_df = merged_df.merge(home_stats, how='left', on=['Season', 'HomeTeam'])
merged_df = merged_df.merge(away_stats, how='left', on=['Season', 'AwayTeam'])
# Defines the columns

cols_to_keep = [
    'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',  # Fouls / Corners
    'HY', 'AY', 'HR', 'AR',  # Yellow/Red Cards
    'B365H', 'B365D', 'B365A',  # B356
    'BWH', 'BWD', 'BWA',  # Pinnacle
    'PSH', 'PSD', 'PSA',  # Bet&Win
    'WHH', 'WHD', 'WHA',  # William Hill
    'Season',
    'HPos', 'HPlayed', 'HWon', 'HDrawn', 'HLost',
    'APos', 'APlayed', 'AWon', 'ADrawn', 'ALost'
]

# List of columns to convert
numeric_cols = [
    'FTHG', 'FTAG', 'HTHG', 'HTAG',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',  # Fouls / Corners
    'HY', 'AY', 'HR', 'AR',  # Yellow/Red Cards
    'B365H', 'B365D', 'B365A',  # B365
    'BWH', 'BWD', 'BWA',  # Pinnacle
    'PSH', 'PSD', 'PSA',  # Bet&Win
    'WHH', 'WHD', 'WHA',  # William Hill
    'HPos', 'HPlayed', 'HWon', 'HDrawn', 'HLost',
    'APos', 'APlayed', 'AWon', 'ADrawn', 'ALost'
]

betting_cols = [
    'B365H', 'B365D', 'B365A',
    'BWH', 'BWD', 'BWA',
    'PSH', 'PSD', 'PSA',
    'WHH', 'WHD', 'WHA'
]

# Impute betting columns
merged_df[betting_cols] = merged_df[betting_cols].fillna(0)

# avoid division by zero
eps = 1e-6

# Pre-match form & ranking features
merged_df['HWinRate'] = merged_df['HWon'] / (merged_df['HPlayed'] + eps)
merged_df['AWinRate'] = merged_df['AWon'] / (merged_df['APlayed'] + eps)
merged_df['HDrawRate'] = merged_df['HDrawn'] / (merged_df['HPlayed'] + eps)
merged_df['ADrawRate'] = merged_df['ADrawn'] / (merged_df['APlayed'] + eps)
merged_df['HLossRate'] = merged_df['HLost'] / (merged_df['HPlayed'] + eps)
merged_df['ALossRate'] = merged_df['ALost'] / (merged_df['APlayed'] + eps)

merged_df['PosDiff'] = merged_df['APos'] - merged_df['HPos']
merged_df['PosRatio'] = merged_df['HPos'] / (merged_df['APos'] + eps)

# Pre-match betting odds features
merged_df['AvgH'] = merged_df[['B365H', 'BWH', 'PSH', 'WHH']].mean(axis=1)
merged_df['AvgD'] = merged_df[['B365D', 'BWD', 'PSD', 'WHD']].mean(axis=1)
merged_df['AvgA'] = merged_df[['B365A', 'BWA', 'PSA', 'WHA']].mean(axis=1)

merged_df['MaxH'] = merged_df[['B365H', 'BWH', 'PSH', 'WHH']].max(axis=1)
merged_df['MaxD'] = merged_df[['B365D', 'BWD', 'PSD', 'WHD']].max(axis=1)
merged_df['MaxA'] = merged_df[['B365A', 'BWA', 'PSA', 'WHA']].max(axis=1)

merged_df['HDRatio'] = merged_df['AvgH'] / (merged_df['AvgD'] + eps)
merged_df['HARatio'] = merged_df['AvgH'] / (merged_df['AvgA'] + eps)
merged_df['DARatio'] = merged_df['AvgD'] / (merged_df['AvgA'] + eps)

# Save the csv file to dataset folder
merged_df.to_csv(os.path.join(folder_path, 'final.csv'), index=False)
# print(f"\n Merged dataset saved to: {os.path.join(folder_path, 'final.csv')}")

# Features
feature_cols = [
    'HTHG', 'HTAG',
    'B365H', 'B365D', 'B365A',
    'BWH', 'BWD', 'BWA',
    'PSH', 'PSD', 'PSA',
    'WHH', 'WHD', 'WHA',
    'AvgH', 'AvgD', 'AvgA',
    'MaxH', 'MaxD', 'MaxA',
    'HPos', 'HPlayed', 'HWon', 'HDrawn', 'HLost',
    'APos', 'APlayed', 'AWon', 'ADrawn', 'ALost',
    'HWinRate', 'AWinRate',
    'HDrawRate', 'ADrawRate',
    'HLossRate', 'ALossRate',
    'PosDiff', 'PosRatio',
    'HDRatio', 'HARatio', 'DARatio'
]

X = merged_df[feature_cols]
y = merged_df['FTR']  # Target: H, D, A

# Features Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search for Best Parameters

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']  # Handle imbalanced classes
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_scaled, y)

# Model Evaluation
best_model = grid_search.best_estimator_
print("\n ------ Best Parameters ------ \n")
print(grid_search.best_params_)

# Evaluate using cross-validated predictions
y_pred = cross_val_predict(best_model, X_scaled, y, cv=cv)

print("\n ------ Confusion Matrix ------ \n")
print(confusion_matrix(y, y_pred))

print("\n ------ Classification Report ------ \n ")
print(classification_report(y, y_pred))

joblib.dump(best_model, os.path.join(save_path, "in_match_result_model.pkl"))
joblib.dump(scaler, os.path.join(save_path, "in_match_result_scaler.pkl"))
