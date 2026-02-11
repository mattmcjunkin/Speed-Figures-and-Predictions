import io
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

US_THOROUGHBRED_TRACKS = {
    "AQU": "Aqueduct",
    "BEL": "Belmont Park",
    "SAR": "Saratoga",
    "CD": "Churchill Downs",
    "KEE": "Keeneland",
    "SA": "Santa Anita",
    "DMR": "Del Mar",
    "GP": "Gulfstream Park",
    "TAM": "Tampa Bay Downs",
    "FG": "Fair Grounds",
    "OP": "Oaklawn Park",
    "PIM": "Pimlico",
    "LRL": "Laurel Park",
    "MTH": "Monmouth Park",
    "WO": "Woodbine",
    "PRX": "Parx Racing",
    "BAQ": "Belmont at the Big A",
    "GG": "Golden Gate Fields",
    "HOU": "Sam Houston",
    "LRC": "Los Alamitos",
    "ELP": "Ellis Park",
    "IND": "Horseshoe Indianapolis",
    "CT": "Charles Town",
    "PEN": "Penn National",
    "DEL": "Delaware Park",
    "LS": "Lone Star Park",
    "MVR": "Mahoning Valley",
    "PID": "Presque Isle Downs",
    "RP": "Remington Park",
    "TUP": "Turf Paradise",
    "FL": "Finger Lakes",
    "CBY": "Canterbury",
    "EVD": "Evangeline Downs",
    "LAD": "Louisiana Downs",
    "GGF": "Golden Gate",
    "SR": "Santa Rosa",
    "MNR": "Mountaineer",
    "RUI": "Ruidoso Downs",
    "AP": "Arlington Park",
    "BTP": "Belterra Park",
}

DEFAULT_COLUMNS = [
    "track",
    "race",
    "horse",
    "post",
    "surface",
    "distance",
    "early_pace",
    "middle_pace",
    "late_pace",
    "last_speed",
    "avg_speed",
    "class_rating",
    "days_since",
    "run_style",
]


@st.cache_data(show_spinner=False)
def parse_drf(file_bytes: bytes, filename: str) -> pd.DataFrame:
    text = file_bytes.decode(errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    raw = "\n".join(lines)

    if "|" in raw:
        df = pd.read_csv(io.StringIO(raw), sep="|", engine="python")
    elif "\t" in raw:
        df = pd.read_csv(io.StringIO(raw), sep="\t", engine="python")
    else:
        try:
            df = pd.read_csv(io.StringIO(raw), engine="python")
        except Exception:
            recs = []
            for ln in lines:
                if len(ln) < 70:
                    continue
                recs.append(
                    {
                        "track": ln[0:3].strip(),
                        "race": ln[3:6].strip(),
                        "horse": ln[6:30].strip(),
                        "post": ln[30:32].strip(),
                        "surface": ln[32:33].strip(),
                        "distance": ln[33:37].strip(),
                        "early_pace": ln[37:40].strip(),
                        "middle_pace": ln[40:43].strip(),
                        "late_pace": ln[43:46].strip(),
                        "last_speed": ln[46:49].strip(),
                        "avg_speed": ln[49:52].strip(),
                        "class_rating": ln[52:55].strip(),
                        "days_since": ln[55:58].strip(),
                        "run_style": ln[58:60].strip(),
                    }
                )
            df = pd.DataFrame(recs)

    rename_map = {
        "trk": "track",
        "track_code": "track",
        "race_num": "race",
        "r": "race",
        "horse_name": "horse",
        "program": "post",
        "post_position": "post",
        "dist": "distance",
        "ep": "early_pace",
        "mp": "middle_pace",
        "lp": "late_pace",
        "speed": "last_speed",
        "avgspd": "avg_speed",
        "class": "class_rating",
        "layoff": "days_since",
        "style": "run_style",
    }
    lower_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=lower_map).rename(columns=rename_map)

    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[DEFAULT_COLUMNS].copy()
    df["source_file"] = filename

    for num_col in [
        "race",
        "post",
        "distance",
        "early_pace",
        "middle_pace",
        "late_pace",
        "last_speed",
        "avg_speed",
        "class_rating",
        "days_since",
    ]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df["track"] = df["track"].astype(str).str.upper().str.strip()
    df["surface"] = df["surface"].astype(str).str.upper().str.strip().replace({"NAN": ""})
    df["run_style"] = df["run_style"].astype(str).str.upper().str.strip().replace({"NAN": "UNK"})
    df["horse"] = df["horse"].astype(str).str.strip()
    return df.dropna(subset=["track", "race", "horse"], how="any")


@st.cache_data(show_spinner=False)
def parse_history(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        hist = pd.read_csv(io.BytesIO(file_bytes))
    else:
        hist = pd.read_excel(io.BytesIO(file_bytes))

    hist.columns = [c.lower().strip() for c in hist.columns]
    required = ["track", "post", "finish_position", "run_style"]
    for col in required:
        if col not in hist.columns:
            hist[col] = np.nan

    hist["track"] = hist["track"].astype(str).str.upper().str.strip()
    hist["post"] = pd.to_numeric(hist["post"], errors="coerce")
    hist["finish_position"] = pd.to_numeric(hist["finish_position"], errors="coerce")
    hist["run_style"] = hist["run_style"].astype(str).str.upper().str.strip()
    hist["won"] = (hist["finish_position"] == 1).astype(int)
    return hist


def compute_track_bias(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["track", "bias_post", "bias_speed", "bias_style"])

    grp = history.groupby("track", dropna=False)
    bias = grp.agg(
        avg_win_post=("post", lambda s: float(np.nanmean(s[history.loc[s.index, "won"] == 1])) if (history.loc[s.index, "won"] == 1).any() else np.nan),
        overall_post=("post", "mean"),
        style_win_rate=("won", "mean"),
    ).reset_index()

    bias["bias_post"] = (bias["overall_post"] - bias["avg_win_post"]).fillna(0).clip(-2, 2)
    bias["bias_style"] = (bias["style_win_rate"] - bias["style_win_rate"].mean()).fillna(0).clip(-0.1, 0.1)
    bias["bias_speed"] = (bias["bias_post"] * -1.2 + bias["bias_style"] * 20).clip(-3, 3)
    return bias[["track", "bias_post", "bias_speed", "bias_style"]]


def score_horses(card: pd.DataFrame, track_bias: pd.DataFrame, use_bias: bool = True) -> pd.DataFrame:
    scored = card.copy()
    for col in ["early_pace", "middle_pace", "late_pace", "last_speed", "avg_speed", "class_rating"]:
        scored[col] = scored[col].fillna(scored[col].median())

    scored["days_since"] = scored["days_since"].fillna(scored["days_since"].median()).clip(lower=0)
    scored["post"] = scored["post"].fillna(scored["post"].median())

    scored["pace_composite"] = scored[["early_pace", "middle_pace", "late_pace"]].mean(axis=1)
    scored["form_score"] = 0.65 * scored["last_speed"] + 0.35 * scored["avg_speed"]
    scored["recency_adj"] = np.exp(-scored["days_since"] / 80) * 10

    scored = scored.merge(track_bias, on="track", how="left")
    scored[["bias_post", "bias_speed", "bias_style"]] = scored[["bias_post", "bias_speed", "bias_style"]].fillna(0)

    style_bonus = {
        "E": 1.2,
        "EP": 0.9,
        "P": 0.4,
        "S": -0.2,
        "UNK": 0,
    }
    scored["style_bonus"] = scored["run_style"].map(style_bonus).fillna(0)

    bias_term = (scored["bias_speed"] + scored["bias_post"] * (8 - scored["post"]) * 0.1 + scored["bias_style"] * 10) if use_bias else 0

    scored["proprietary_speed_figure"] = (
        0.35 * scored["form_score"]
        + 0.25 * scored["pace_composite"]
        + 0.20 * scored["class_rating"]
        + 0.10 * scored["recency_adj"]
        + 0.05 * scored["style_bonus"]
        + 0.05 * bias_term
    )

    scored["win_probability"] = (
        scored.groupby(["track", "race"])["proprietary_speed_figure"].transform(lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum())
    )
    scored["rank"] = scored.groupby(["track", "race"])["proprietary_speed_figure"].rank(ascending=False, method="first")
    return scored.sort_values(["track", "race", "rank"])


def pace_chart(race_df: pd.DataFrame):
    melt = race_df.melt(
        id_vars=["horse"],
        value_vars=["early_pace", "middle_pace", "late_pace"],
        var_name="segment",
        value_name="pace",
    )
    segment_order = ["early_pace", "middle_pace", "late_pace"]
    chart = (
        alt.Chart(melt)
        .mark_line(point=True)
        .encode(
            x=alt.X("segment:N", sort=segment_order, title="Race Segment"),
            y=alt.Y("pace:Q", title="Pace Rating"),
            color=alt.Color("horse:N", title="Horse"),
            tooltip=["horse", "segment", "pace"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def build_track_tabs(scored: pd.DataFrame):
    available_tracks = sorted(scored["track"].dropna().unique())
    if not available_tracks:
        st.info("No races available for analysis.")
        return

    tabs = st.tabs([f"{t} • {US_THOROUGHBRED_TRACKS.get(t, 'Track Analyzer')}" for t in available_tracks])

    for tab, track in zip(tabs, available_tracks):
        with tab:
            st.subheader(f"{track} Analyzer")
            track_df = scored[scored["track"] == track]
            race_numbers = sorted(track_df["race"].dropna().astype(int).unique())
            if not race_numbers:
                st.write("No races loaded for this track.")
                continue

            for race in race_numbers:
                race_df = track_df[track_df["race"].astype(int) == race].copy()
                if race_df.empty:
                    continue
                st.markdown(f"#### Race {race} most likely winner")
                top_pick = race_df.nsmallest(1, "rank")[
                    ["horse", "post", "proprietary_speed_figure", "win_probability"]
                ]
                st.dataframe(
                    top_pick.assign(
                        win_probability=lambda d: (d["win_probability"] * 100).round(1).astype(str) + "%",
                        proprietary_speed_figure=lambda d: d["proprietary_speed_figure"].round(2),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption("Pace predictor graphic")
                pace_chart(race_df)

                st.caption("Full contender ranking")
                view = race_df[
                    [
                        "rank",
                        "horse",
                        "post",
                        "run_style",
                        "proprietary_speed_figure",
                        "win_probability",
                        "early_pace",
                        "middle_pace",
                        "late_pace",
                        "last_speed",
                        "class_rating",
                    ]
                ].copy()
                view["win_probability"] = (view["win_probability"] * 100).round(1)
                st.dataframe(view.sort_values("rank"), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Thoroughbred Race Analyzer", layout="wide")
    st.title("Proprietary Thoroughbred Speed Figures + Race Winner Analyzer")
    st.write(
        "Upload Brisnet `.drf` files for a race card, then optionally upload historical results "
        "to detect track bias and blend it into the current race analysis."
    )

    with st.sidebar:
        st.header("Inputs")
        drf_files = st.file_uploader(
            "Upload Brisnet race card files (.drf)",
            type=["drf", "txt", "csv"],
            accept_multiple_files=True,
        )
        history_file = st.file_uploader(
            "Optional: Upload historical results (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )
        selected_track = st.selectbox(
            "Filter focus track (optional)",
            options=["ALL"] + [f"{k} - {v}" for k, v in sorted(US_THOROUGHBRED_TRACKS.items())],
        )
        use_bias = st.checkbox("Use historical bias adjustments", value=True)

    if not drf_files:
        st.info("Upload at least one .drf file to begin analysis.")
        st.stop()

    frames = [parse_drf(f.getvalue(), f.name) for f in drf_files]
    card = pd.concat(frames, ignore_index=True)
    card = card.dropna(subset=["horse"])

    history = pd.DataFrame()
    if history_file is not None:
        history = parse_history(history_file.getvalue(), history_file.name)

    track_bias = compute_track_bias(history)
    scored = score_horses(card, track_bias=track_bias, use_bias=use_bias)

    if selected_track != "ALL":
        selected_code = selected_track.split(" - ")[0]
        scored = scored[scored["track"] == selected_code]

    st.metric("Races analyzed", int(scored[["track", "race"]].drop_duplicates().shape[0]))
    st.metric("Horses scored", int(scored.shape[0]))

    if history_file is not None and not history.empty:
        st.subheader("Detected Track Bias Snapshot")
        st.dataframe(track_bias.sort_values("track"), use_container_width=True, hide_index=True)

    build_track_tabs(scored)


if __name__ == "__main__":
    main()
