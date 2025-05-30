import os
import requests

start_season = 7
end_season = 25

ENGLAND = 'E0'
GERMANY = 'D1'
ITALY = 'I1'
SPAIN = 'SP1'
FRANCE = 'F1'

leagues = [ENGLAND, GERMANY, ITALY, SPAIN, FRANCE]


def download_season_data(league, season_code, year, save_dir, name):
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league}.csv"
    file_path = os.path.join(save_dir, f"{name}_20{year:02d}-20{year + 1:02d}.csv")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {file_path}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")

for league in leagues:
    if league == ENGLAND:
        save_dir = "../data/match_data/England"
        name = 'England'
    elif league == GERMANY:
        save_dir = "../data/match_data/Germany"
        name = 'Germany'
    elif league == ITALY:
        save_dir = "../data/match_data/Italy"
        name = 'Italy'
    elif league == SPAIN:
        save_dir = "../data/match_data/Spain"
        name = 'Spain'
    else:
        save_dir = "../data/match_data/France"
        name = 'France'
    os.makedirs(save_dir, exist_ok=True)
    for year in range(start_season, end_season):
        season_code = f"{year:02d}{year + 1:02d}"  # e.g., '0304', '0405', ..., '2324'
        download_season_data(league, season_code, year, save_dir, name)
