import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ===== Data and model loading =====
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data', 'in_match_predict')
    club_stats = pd.read_csv(os.path.join(data_path, "club_stats.csv"))
    win_rate = pd.read_csv(os.path.join(data_path, "team_win_rates.csv"))
    return club_stats, win_rate

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

# ===== Main interface =====
def main():
    st.title("🏟️ In-Match Result Prediction")

    club_stats, winrates = load_data()
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

    if st.button("🔮 Predict"):
        try:
            # 提取每支队伍最新的赛季记录
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

# ===== Exported for app.py to call =====
def render_in_match_predict_section(mode):
    if mode == "Match Predict (In-match)":
        main()
