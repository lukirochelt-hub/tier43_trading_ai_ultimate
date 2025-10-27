# dashboard_heatmap.py — Tier 4.3+ Trading AI (Interactive Dashboard)
# Minimaler Dash-Server, der heatmap_agg.json lädt und als Heatmap + Tabelle zeigt.
import os
import json
import pandas as pd

from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go

DATA_DIR = os.getenv("DATA_DIR", "data")
INPUT_PATH = os.path.join(DATA_DIR, "heatmap_agg.json")
REFRESH_MS = int(os.getenv("HEATMAP_REFRESH_MS", "5000"))  # 5s


def load_df():
    if not os.path.exists(INPUT_PATH):
        return pd.DataFrame(
            columns=[
                "metric",
                "pnl_corr",
                "vol_corr",
                "sentiment_corr",
                "macro_corr",
                "timestamp",
                "records",
            ]
        )
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    # nur letzte Werte pro metric
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("ts").groupby("metric", as_index=False).last()
    return df


def fig_heatmap(df: pd.DataFrame):
    if df.empty:
        z = [[0, 0, 0, 0]]
        y = ["no data"]
    else:
        z = df[["pnl_corr", "vol_corr", "sentiment_corr", "macro_corr"]].to_numpy().tolist()
        dtype=float
        
        y = df["metric"].astype(str).tolist()
    x = ["pnl_corr", "vol_corr", "sentiment_corr", "macro_corr"]
    fig = go.Figure(
        data=go.Heatmap(z=z, x=x, y=y, zmin=-1, zmax=1, colorbar_title="Spearman")
    )
    fig.update_layout(
        margin=dict(l=80, r=20, t=40, b=40), title="Tier 4.3+ — Correlation Heatmap"
    )
    return fig


def fig_topbars(df: pd.DataFrame, target="pnl_corr", top_k=20):
    if df.empty or target not in df.columns:
        return go.Figure()
    d = df[["metric", target]].dropna()
    d = d[d["metric"].str.lower() != target.replace("_corr", "")]
    d["abs"] = d[target].abs()
    d = d.sort_values("abs", ascending=False).head(top_k)
    fig = go.Figure(go.Bar(x=d[target], y=d["metric"], orientation="h"))
    fig.update_layout(
        margin=dict(l=120, r=20, t=40, b=40), title=f"Top {len(d)} vs {target}"
    )
    return fig


app = Dash(__name__)
app.title = "Tier 4.3+ Heatmap"

app.layout = html.Div(
    [
        html.H1("Tier 4.3+ — Heatmap Dashboard", style={"margin": "16px 0"}),
        html.Div(id="meta", style={"opacity": 0.7, "marginBottom": "8px"}),
        dcc.Dropdown(
            id="target",
            options=[
                {"label": l, "value": l}
                for l in ["pnl_corr", "vol_corr", "sentiment_corr", "macro_corr"]
            ],
            value="pnl_corr",
            clearable=False,
            style={"width": "240px", "marginBottom": "8px"},
        ),
        dcc.Graph(id="heatmap"),
        dcc.Graph(id="topbars"),
        dash_table.DataTable(
            id="table",
            columns=[
                {"name": c, "id": c}
                for c in [
                    "metric",
                    "pnl_corr",
                    "vol_corr",
                    "sentiment_corr",
                    "macro_corr",
                    "timestamp",
                    "records",
                ]
            ],
            style_table={"overflowX": "auto"},
            page_size=20,
        ),
        dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "12px"},
)


@app.callback(
    Output("heatmap", "figure"),
    Output("topbars", "figure"),
    Output("table", "data"),
    Output("meta", "children"),
    Input("tick", "n_intervals"),
    Input("target", "value"),
)
def _update(_, target):
    df = load_df()
    heatmap = fig_heatmap(df)
    top = fig_topbars(df, target=target)
    meta = "keine Daten"
    if not df.empty and "timestamp" in df.columns:
        latest = pd.to_datetime(df["timestamp"]).max()
        meta = f"Letztes Update: {latest} | Metrics: {len(df)} | Datei: {INPUT_PATH}"
    return heatmap, top, df.to_dict("records"), meta


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=False)
