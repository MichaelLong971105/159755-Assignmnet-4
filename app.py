# app.py (主程序入口)
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Manager Simulator", layout="centered")
from components.player_input import handle_player_input
from components.recruit import render_recruit_section
from components.team_manage import render_team_section
from components.match import run_season_simulation
from components.pre_match_predict import show_all_teams
from components.in_match_predict import render_in_match_predict_section

st.title("🎮 Virtual Football Manager")

# Initialization state
if 'budget' not in st.session_state:
    st.session_state['budget'] = 100_000_000
if 'team' not in st.session_state:
    st.session_state['team'] = []

# Function Selection
mode = st.radio("Select which one you want", ["Create New Player", "Choose Preset Player",
                                              "Match Predict (Pre-match)", "Match Predict (In-match)"])
st.session_state['mode'] = mode

# Processing player input logic (sidebar + model predictions)
handle_player_input(mode)

# Recruitment module (showing Player to Recruit + button)
render_recruit_section(mode)

# Team Presentation + Management
render_team_section(mode)

# Predict Pre_match
show_all_teams(mode)

# Predict In_match
render_in_match_predict_section(mode)

# Simulation Match
run_season_simulation()

# Reset button
if st.sidebar.button("Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
