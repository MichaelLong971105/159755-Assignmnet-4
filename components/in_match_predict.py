import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Data and model loading =====
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data', 'in_match_predict')
    club_stats = pd.read_csv(os.path.join(data_path, "club_stats.csv"))
    win_rate = pd.read_csv(os.path.join(data_path, "team_win_rates.csv"))
    return club_stats, win_rate

@st.cache_data
def load_merged_match_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(base_path, '..', 'data', 'in_match_predict', 'final.csv')
    df = pd.read_csv(df_path, parse_dates=['Date'])
    if df.empty:
        st.error("❌ Failed to load match data. Please check CSV path or content.")
        return
    return df

def load_model_and_scaler():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, '..', 'models', 'in_match_result_model.pkl')
    scaler_path = os.path.join(base_path, '..', 'models', 'in_match_result_scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ===== Betting suggestion function =====
def betting_recommendation(pred_result, home_winrate, away_winrate, draw_gap=0.05):
    if pred_result == 'H':
        return '✅ Bet' if home_winrate > away_winrate else '❌ No Bet'
    elif pred_result == 'A':
        return '✅ Bet' if away_winrate > home_winrate else '❌ No Bet'
    elif pred_result == 'D':
        return '✅ Bet' if abs(home_winrate - away_winrate) <= draw_gap else '❌ No Bet'
    return 'Unknown'

# ===== Visualization =====
def plot_team_season_stats(df, team, season):
    st.markdown(f"### 📈 {team} - Performance in {season}")
    team_df = df[(df['Season'] == season) & ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))].copy()
    if team_df.empty:
        st.warning("No match data found.")
        return

    team_df['Goals Scored'] = team_df.apply(lambda row: row['FTHG'] if row['HomeTeam'] == team else row['FTAG'], axis=1)
    team_df['Goals Conceded'] = team_df.apply(lambda row: row['FTAG'] if row['HomeTeam'] == team else row['FTHG'], axis=1)
    team_df['Match Result'] = team_df.apply(
        lambda row: 'Win' if (row['HomeTeam'] == team and row['FTR'] == 'H') or (row['AwayTeam'] == team and row['FTR'] == 'A')
        else ('Draw' if row['FTR'] == 'D' else 'Loss'), axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=team_df, x='Date', y='Goals Scored', label='Scored', ax=axs[0])
    sns.lineplot(data=team_df, x='Date', y='Goals Conceded', label='Conceded', ax=axs[0])
    axs[0].set_title("Goals Over Time")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend()

    result_counts = team_df['Match Result'].value_counts()
    sns.barplot(x=result_counts.index, y=result_counts.values, palette='Set2', ax=axs[1])
    axs[1].set_title("Match Result Distribution")

    st.pyplot(fig)

def plot_head_to_head(df, team1, team2, season):
    st.markdown(f"### 🤝 {team1} vs {team2} - Head to Head ({season})")
    h2h_df = df[(df['Season'] == season) & (
        ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
        ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
    )].copy()
    if h2h_df.empty:
        st.warning("No head-to-head matches found.")
        return

    def team_result(row):
        if row['HomeTeam'] == team1:
            return 'Win' if row['FTR'] == 'H' else ('Loss' if row['FTR'] == 'A' else 'Draw')
        else:
            return 'Win' if row['FTR'] == 'A' else ('Loss' if row['FTR'] == 'H' else 'Draw')

    h2h_df['Team1_Result'] = h2h_df.apply(team_result, axis=1)
    avg1 = h2h_df.apply(lambda r: r['FTHG'] if r['HomeTeam'] == team1 else r['FTAG'], axis=1).mean()
    avg2 = h2h_df.apply(lambda r: r['FTHG'] if r['HomeTeam'] == team2 else r['FTAG'], axis=1).mean()

    st.markdown(f"- `{team1}` Avg Goals: **{avg1:.2f}** | `{team2}` Avg Goals: **{avg2:.2f}**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Team1_Result', data=h2h_df, palette='Set1', ax=ax)
    ax.set_title(f"{team1} Match Outcomes vs {team2}")
    st.pyplot(fig)

# ===== Main interface =====
def main():
    st.title("🏟️ In-Match Result Prediction")
    club_stats, winrates = load_data()
    merged_df = load_merged_match_data()

    team_names = sorted(winrates["HomeTeam"].unique())
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", team_names)
        hthg = st.number_input("⚽ Half-time Goals (Home)", min_value=0, max_value=10, value=1)
        odd_h = st.number_input("💰 Odds - Home Win", min_value=1.0, value=2.1)
    with col2:
        away_team = st.selectbox("🛫 Away Team", [t for t in team_names if t != home_team])
        htag = st.number_input("⚽ Half-time Goals (Away)", min_value=0, max_value=10, value=1)
        odd_a = st.number_input("💰 Odds - Away Win", min_value=1.0, value=3.1)

    odd_d = st.number_input("💰 Odds - Draw", min_value=1.0, value=3.0)
    season_options = sorted(merged_df["Season"].dropna().unique())

    if season_options:
        season = st.selectbox("📅 Select Season", season_options, index=len(season_options) - 1)
    else:
        st.warning("⚠️ No season data available.")
        return

    if st.button("🔮 Predict"):
        try:
            recent_home = club_stats[club_stats["Club"] == home_team].sort_values("Season", ascending=False).iloc[0]
            recent_away = club_stats[club_stats["Club"] == away_team].sort_values("Season", ascending=False).iloc[0]
            win_row = winrates[winrates["HomeTeam"] == home_team].iloc[0]
            away_win_row = winrates[winrates["HomeTeam"] == away_team].iloc[0]
        except IndexError:
            st.error("❌ No data found for selected teams.")
            return

        home_winrate, away_winrate = win_row["HomeWinRate"], away_win_row["AwayWinRate"]
        eps = 1e-6

        df_input = pd.DataFrame([{
            'HTHG': hthg, 'HTAG': htag,
            'B365H': odd_h, 'B365D': odd_d, 'B365A': odd_a,
            'BWH': odd_h, 'BWD': odd_d, 'BWA': odd_a,
            'PSH': odd_h, 'PSD': odd_d, 'PSA': odd_a,
            'WHH': odd_h, 'WHD': odd_d, 'WHA': odd_a,
            'AvgH': odd_h, 'AvgD': odd_d, 'AvgA': odd_a,
            'MaxH': odd_h, 'MaxD': odd_d, 'MaxA': odd_a,
            'HPos': recent_home["Position"], 'HPlayed': recent_home["Played"],
            'HWon': recent_home["Won"], 'HDrawn': recent_home["Drawn"], 'HLost': recent_home["Lost"],
            'APos': recent_away["Position"], 'APlayed': recent_away["Played"],
            'AWon': recent_away["Won"], 'ADrawn': recent_away["Drawn"], 'ALost': recent_away["Lost"],
            'HWinRate': home_winrate, 'AWinRate': away_winrate,
            'HDrawRate': recent_home["Drawn"] / (recent_home["Played"] + eps),
            'ADrawRate': recent_away["Drawn"] / (recent_away["Played"] + eps),
            'HLossRate': recent_home["Lost"] / (recent_home["Played"] + eps),
            'ALossRate': recent_away["Lost"] / (recent_away["Played"] + eps),
            'PosDiff': recent_away["Position"] - recent_home["Position"],
            'PosRatio': recent_home["Position"] / (recent_away["Position"] + eps),
            'HDRatio': odd_h / (odd_d + eps),
            'HARatio': odd_h / (odd_a + eps),
            'DARatio': odd_d / (odd_a + eps)
        }])

        model, scaler = load_model_and_scaler()
        X_scaled = scaler.transform(df_input)
        pred_proba = model.predict_proba(X_scaled)[0]
        label_order = model.classes_
        pred_label = label_order[np.argmax(pred_proba)]

        label_text = {'H': '🏠 Home Win', 'D': '⚖️ Draw', 'A': '🏟️ Away Win'}
        readable = label_text[pred_label]
        bet_suggestion = betting_recommendation(pred_label, home_winrate, away_winrate)

        st.success(f"🎯 Prediction: **{readable}**")
        st.markdown("### 📊 Probability")
        st.markdown(f"- Home Win: **{pred_proba[label_order.tolist().index('H')]*100:.1f}%**")
        st.markdown(f"- Draw: **{pred_proba[label_order.tolist().index('D')]*100:.1f}%**")
        st.markdown(f"- Away Win: **{pred_proba[label_order.tolist().index('A')]*100:.1f}%**")
        st.markdown("---")
        st.markdown(f"### 💡 Betting Suggestion: **{bet_suggestion}**")
        st.markdown(f"🏠 Home Win Rate: **{home_winrate:.2f}**  🛫 Away Win Rate: **{away_winrate:.2f}**")

        # Plots
        st.divider()
        plot_team_season_stats(merged_df, home_team, season)
        st.divider()
        plot_team_season_stats(merged_df, away_team, season)
        st.divider()
        plot_head_to_head(merged_df, home_team, away_team, season)

# ===== Exported for app.py to call =====
def render_in_match_predict_section(mode):
    if mode == "Match Predict (In-match)":
        main()
