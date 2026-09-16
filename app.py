"""
App Streamlit — Sistema de Rating CBLOL (Elo + Glicko-2)
==========================================================
Deploy: mesmo fluxo do projeto de VaR (GitHub -> Streamlit Community Cloud)

Rodar localmente:
    pip install streamlit pandas numpy matplotlib plotly
    streamlit run app.py

O app espera um CSV no formato Oracle's Elixir. O usuário faz upload
direto na interface (não precisa vir hardcoded no repositório).
"""

import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="CBLOL Rating System", page_icon="🎮", layout="wide")


# ----------------------------------------------------------------------
# RATING SYSTEMS (mesma lógica do script original)
# ----------------------------------------------------------------------

class EloRatingSystem:
    def __init__(self, k=24, base_rating=1500):
        self.k = k
        self.base_rating = base_rating
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, self.base_rating)

    def win_probability(self, team_a, team_b):
        ra, rb = self.get(team_a), self.get(team_b)
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, team_a, team_b, a_won: bool):
        pa = self.win_probability(team_a, team_b)
        score_a = 1.0 if a_won else 0.0
        ra, rb = self.get(team_a), self.get(team_b)
        self.ratings[team_a] = ra + self.k * (score_a - pa)
        self.ratings[team_b] = rb + self.k * ((1 - score_a) - (1 - pa))


class Glicko2Team:
    def __init__(self, rating=1500, rd=350, vol=0.06):
        self.rating = rating
        self.rd = rd
        self.vol = vol


class Glicko2RatingSystem:
    TAU = 0.5
    SCALE = 173.7178

    def __init__(self):
        self.teams = {}

    def _get(self, name):
        if name not in self.teams:
            self.teams[name] = Glicko2Team()
        return self.teams[name]

    def _to_scale(self, team):
        return (team.rating - 1500) / self.SCALE, team.rd / self.SCALE

    def win_probability(self, team_a, team_b):
        ta, tb = self._get(team_a), self._get(team_b)
        mu_a, phi_a = self._to_scale(ta)
        mu_b, phi_b = self._to_scale(tb)
        g_phi = 1 / math.sqrt(1 + 3 * phi_b ** 2 / math.pi ** 2)
        return 1 / (1 + math.exp(-g_phi * (mu_a - mu_b)))

    def update(self, team_a, team_b, a_won: bool):
        self._update_one(team_a, team_b, 1.0 if a_won else 0.0)
        self._update_one(team_b, team_a, 0.0 if a_won else 1.0)

    def _update_one(self, name, opp_name, score):
        team, opp = self._get(name), self._get(opp_name)
        mu, phi = self._to_scale(team)
        mu_j, phi_j = self._to_scale(opp)
        g_j = 1 / math.sqrt(1 + 3 * phi_j ** 2 / math.pi ** 2)
        E_j = 1 / (1 + math.exp(-g_j * (mu - mu_j)))
        v = 1 / (g_j ** 2 * E_j * (1 - E_j) + 1e-10)
        delta = v * g_j * (score - E_j)
        a = math.log(team.vol ** 2)
        A = a
        eps = 1e-6
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while self._f(a - k * self.TAU, delta, phi, v, a) < 0:
                k += 1
            B = a - k * self.TAU
        fA, fB = self._f(A, delta, phi, v, a), self._f(B, delta, phi, v, a)
        while abs(B - A) > eps:
            C = A + (A - B) * fA / (fB - fA)
            fC = self._f(C, delta, phi, v, a)
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC
        new_vol = math.exp(A / 2)
        phi_star = math.sqrt(phi ** 2 + new_vol ** 2)
        new_phi = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
        new_mu = mu + new_phi ** 2 * g_j * (score - E_j)
        team.rating = new_mu * self.SCALE + 1500
        team.rd = new_phi * self.SCALE
        team.vol = new_vol

    def _f(self, x, delta, phi, v, a):
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2 * (phi ** 2 + v + ex) ** 2
        return (num / den) - (x - a) / self.TAU ** 2


# ----------------------------------------------------------------------
# DADOS
# ----------------------------------------------------------------------

@st.cache_data
def load_matches(file, league):
    df = pd.read_csv(file, low_memory=False)
    if "position" in df.columns:
        df = df[df["position"].str.lower() == "team"]
    df = df[df["league"] == league].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    matches = []
    for gid, g in df.groupby("gameid"):
        if len(g) != 2:
            continue
        row_blue = g[g["side"].str.lower() == "blue"]
        row_red = g[g["side"].str.lower() == "red"]
        if row_blue.empty or row_red.empty:
            continue
        row_blue, row_red = row_blue.iloc[0], row_red.iloc[0]
        matches.append({
            "date": row_blue["date"],
            "team_blue": row_blue["teamname"],
            "team_red": row_red["teamname"],
            "blue_won": int(row_blue["result"]),
        })
    return pd.DataFrame(matches).sort_values("date").reset_index(drop=True)


@st.cache_data
def run_ratings(matches):
    elo = EloRatingSystem()
    g2 = Glicko2RatingSystem()
    elo_history, g2_history = [], []
    elo_preds, g2_preds = [], []

    for _, row in matches.iterrows():
        elo_preds.append(elo.win_probability(row["team_blue"], row["team_red"]))
        g2_preds.append(g2.win_probability(row["team_blue"], row["team_red"]))

        elo.update(row["team_blue"], row["team_red"], bool(row["blue_won"]))
        g2.update(row["team_blue"], row["team_red"], bool(row["blue_won"]))

        for team in (row["team_blue"], row["team_red"]):
            elo_history.append({"date": row["date"], "team": team, "rating": elo.get(team)})
            g2_history.append({"date": row["date"], "team": team, "rating": g2._get(team).rating})

    return elo, g2, pd.DataFrame(elo_history), pd.DataFrame(g2_history), np.array(elo_preds), np.array(g2_preds)


def metrics(preds, actual):
    eps = 1e-10
    logloss = -np.mean(actual * np.log(preds + eps) + (1 - actual) * np.log(1 - preds + eps))
    acc = np.mean((preds > 0.5).astype(int) == actual)
    return logloss, acc


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🎮 CBLOL Rating System")
st.caption("Elo + Glicko-2 aplicado ao CBLOL, com dados do Oracle's Elixir")

uploaded = st.file_uploader("Envie o CSV do Oracle's Elixir", type="csv")
league = st.sidebar.text_input("Liga (código Oracle's Elixir)", value="CBLOL")

if uploaded is None:
    st.info("⬆️ Envie um CSV do Oracle's Elixir pra começar (oracleselixir.com/tools/downloads).")
    st.stop()

matches = load_matches(uploaded, league)

if matches.empty:
    st.error(f"Nenhuma partida encontrada pra liga '{league}'. Confira o código da liga ou o arquivo.")
    st.stop()

st.success(f"{len(matches)} partidas carregadas de {league}.")

elo, g2, elo_hist, g2_hist, elo_preds, g2_preds = run_ratings(matches)
actual = matches["blue_won"].values
elo_ll, elo_acc = metrics(elo_preds, actual)
g2_ll, g2_acc = metrics(g2_preds, actual)
baseline_acc = max(actual.mean(), 1 - actual.mean())

tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Evolução de rating", "⚔️ Simular confronto"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia — Baseline", f"{baseline_acc:.1%}")
    col2.metric("Acurácia — Elo", f"{elo_acc:.1%}", f"{(elo_acc - baseline_acc):+.1%}")
    col3.metric("Acurácia — Glicko-2", f"{g2_acc:.1%}", f"{(g2_acc - baseline_acc):+.1%}")
    st.caption(f"Log loss — Elo: {elo_ll:.4f} | Glicko-2: {g2_ll:.4f} (quanto menor, melhor)")

    st.subheader("Ranking atual dos times (Elo)")
    ranking = pd.DataFrame(
        [(t, r) for t, r in elo.ratings.items()], columns=["Time", "Rating Elo"]
    ).sort_values("Rating Elo", ascending=False).reset_index(drop=True)
    ranking.index += 1
    st.dataframe(ranking, use_container_width=True)

with tab2:
    system_choice = st.radio("Sistema", ["Elo", "Glicko-2"], horizontal=True)
    hist = elo_hist if system_choice == "Elo" else g2_hist
    teams = st.multiselect("Times pra mostrar", options=sorted(hist["team"].unique()),
                            default=sorted(hist["team"].unique())[:5])
    fig = go.Figure()
    for team in teams:
        team_data = hist[hist["team"] == team]
        fig.add_trace(go.Scatter(x=team_data["date"], y=team_data["rating"],
                                  mode="lines+markers", name=team))
    fig.update_layout(xaxis_title="Data", yaxis_title="Rating", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    teams_list = sorted(elo.ratings.keys())
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Time (lado azul)", teams_list, index=0)
    team_b = c2.selectbox("Time (lado vermelho)", teams_list, index=min(1, len(teams_list) - 1))

    if team_a == team_b:
        st.warning("Escolha dois times diferentes.")
    else:
        p_elo = elo.win_probability(team_a, team_b)
        p_g2 = g2.win_probability(team_a, team_b)
        c1.metric(f"{team_a} vence (Elo)", f"{p_elo:.1%}")
        c1.metric(f"{team_a} vence (Glicko-2)", f"{p_g2:.1%}")
        c2.metric(f"{team_b} vence (Elo)", f"{1 - p_elo:.1%}")
        c2.metric(f"{team_b} vence (Glicko-2)", f"{1 - p_g2:.1%}")

st.divider()
st.caption("Backtesting walk-forward: cada previsão usa só dados anteriores à partida, sem olhar o futuro.")
