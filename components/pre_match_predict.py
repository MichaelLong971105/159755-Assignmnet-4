# components/pre_match_predict.py
import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from components.predict_match_result_model_pre_match import predict_match_result

def show_all_teams(mode):
    if mode == "Match Predict (Pre-match)":
        st.markdown("## 🧾 All Teams from Database")
        base_path = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_path, '..', 'data', 'allData.sl3')

        if not os.path.exists(db_path):
            st.error(f"❌ Database not found: {db_path}")
            return

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM team_stats WHERE year = '2025'", conn)
        conn.close()

        if df.empty:
            st.warning("⚠️ No team data found for 2025.")
            return

        # Display all team data
        st.dataframe(df)

        # Let user selects two teams
        st.markdown("### ⚔️ Predict a Match Between Two Teams")
        team_names = df['team'].tolist()
        team_home = st.selectbox("🏠 Select Home Team", team_names, key="home_team")
        team_away = st.selectbox("🛫 Select Away Team", [t for t in team_names if t != team_home], key="away_team")

        # Odds input area
        st.markdown("### 🎯 Optional: Enter Odds (Leave blank to auto-generate)")
        col1, col2 = st.columns(2)

        with col1:
            input_b365_h = st.text_input("Home Win Odds", "")
            input_b365_a = st.text_input("Away Win Odds", "")

        if st.button("🔮 Predict Match Result"):
            row_home = df[df['team'] == team_home].iloc[0]
            row_away = df[df['team'] == team_away].iloc[0]

            try:
                b365_h = float(input_b365_h) if input_b365_h else round(np.random.uniform(1.3, 3.5), 2)
            except ValueError:
                st.error("Invalid input for Home Win Odds")
                return

            try:
                b365_a = float(input_b365_a) if input_b365_a else round(np.random.uniform(1.5, 4.0), 2)
            except ValueError:
                st.error("Invalid input for Away Win Odds")
                return

            # b365_h = np.round(np.random.uniform(1.3, 3.5), 2)  # Home win odds
            b365_d = np.round(np.random.uniform(2.8, 4.5), 2)  # Draw odds
            # b365_a = np.round(np.random.uniform(1.5, 4.0), 2)  # Away win odds

            inverse = 1 / np.array([b365_h, b365_d, b365_a])
            total = inverse.sum()
            prob_h = inverse[0] / total
            prob_d = inverse[1] / total
            prob_a = inverse[2] / total

            # Constructing prediction input
            df_match = pd.DataFrame([{
                "b365_prob_h": prob_h,
                "b365_prob_d": prob_d,
                "b365_prob_a": prob_a,
                "overall_diff": row_home['overall'] - row_away['overall'],
                "attack_diff": row_home['attack'] - row_away['attack'],
                "midfield_diff": row_home['midfield'] - row_away['midfield'],
                "defence_diff": row_home['defence'] - row_away['defence'],
                "age_diff": row_home['starting_xi_avg_age'] - row_away['starting_xi_avg_age']
            }])

            result = predict_match_result(df_match)

            st.success(f"🏆 Predicted Result: **{team_home} vs {team_away} → {result['prediction']}**")
            st.markdown(f"""
                - Home Win Probability: **{result['probabilities']['Home Win']}**
                - Home win odds: **{b365_h}**
                - Away Win Probability: **{result['probabilities']['Away Win']}**
                - Away win odds: **{b365_a}**
                """)
