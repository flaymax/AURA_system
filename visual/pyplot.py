import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


def build_weekly_risk_dashboard(
    data: pd.DataFrame,
    *,
    target: str,
    score_main: str,
    date_col: str,
    score_alt1: Optional[str] = None,
    score_alt2: Optional[str] = None,
    entity_id: Optional[str] = None,
    week_freq: str = "W-MON",
    show_ci: bool = False,
    ci_alpha: float = 0.95,
    fig_title: Optional[str] = None,
    size: tuple[int, int] = (950, 520),
) -> go.Figure:
    """
    Weekly dashboard: average scores vs target,
    optional confidence band and weekly volume.
    """

    z_map = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_val = z_map.get(round(ci_alpha, 2), 1.96)

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    metrics = [target, score_main]
    for c in (score_alt1, score_alt2):
        if c:
            metrics.append(c)

    weekly = (
        df
        .groupby(pd.Grouper(key=date_col, freq=week_freq))[metrics]
        .mean()
        .sort_index()
    )

    # ---- confidence interval --------------------------------------------
    if show_ci:
        counts = (
            df
            .groupby(pd.Grouper(key=date_col, freq=week_freq))[target]
            .count()
            .reindex(weekly.index)
            .fillna(0)
        )

        p_hat = weekly[target]
        stderr = np.sqrt(p_hat * (1 - p_hat) / counts.clip(lower=1))

        weekly["_ci_lo"] = (p_hat - z_val * stderr).clip(0, 1)
        weekly["_ci_hi"] = (p_hat + z_val * stderr).clip(0, 1)

    # ---- weekly volume ---------------------------------------------------
    volume = None
    if entity_id:
        volume = (
            df
            .groupby(pd.Grouper(key=date_col, freq=week_freq))[entity_id]
            .nunique()
            .reindex(weekly.index)
        )

    # ================== plotting =========================================
    fig = go.Figure()

    def line(y, label, style="solid"):
        fig.add_trace(
            go.Scatter(
                x=weekly.index,
                y=weekly[y],
                name=label,
                mode="lines+markers",
                line=dict(width=2, dash=style),
                marker=dict(size=6),
            )
        )

    line(score_main, f"avg {score_main}")
    if score_alt1:
        line(score_alt1, f"avg {score_alt1}", style="dash")
    if score_alt2:
        line(score_alt2, f"avg {score_alt2}", style="dot")

    line(target, f"avg {target}")

    # ---- CI ribbon -------------------------------------------------------
    if show_ci:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([weekly.index, weekly.index[::-1]]),
                y=np.concatenate([
                    weekly["_ci_hi"].values,
                    weekly["_ci_lo"].values[::-1],
                ]),
                fill="toself",
                fillcolor="rgba(52, 73, 94, 0.18)",
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{int(ci_alpha*100)}% CI ({target})",
            )
        )

    # ---- volume bar ------------------------------------------------------
    if volume is not None:
        fig.add_trace(
            go.Bar(
                x=volume.index,
                y=volume.values,
                name="weekly volume",
                yaxis="y2",
                opacity=0.55,
            )
        )

        fig.update_layout(
            yaxis2=dict(
                title="Count",
                overlaying="y",
                side="right",
                rangemode="tozero",
                showgrid=False,
            )
        )

    # ---- layout ----------------------------------------------------------
    fig.update_layout(
        title=fig_title or "Weekly risk & performance overview",
        xaxis_title="Week",
        yaxis_title="Rate / Probability",
        width=size[0],
        height=size[1],
        template="simple_white",
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
        ),
        margin=dict(l=60, r=60, t=70, b=40),
    )

    return fig
