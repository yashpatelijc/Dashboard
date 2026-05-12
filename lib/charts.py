"""Plotly chart factory — dark theme template + curve chart helpers."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from lib.theme import (
    ACCENT, ACCENT_DIM, AMBER, BG_BASE, BG_SURFACE, BLUE, BORDER_DEFAULT,
    BORDER_SUBTLE, CYAN, GOLD, GREEN, GREEN_DIM, PURPLE, RED, RED_DIM,
    TEXT_BODY, TEXT_DIM, TEXT_HEADING, TEXT_MUTED,
)

# alias for backward compat
BG_PRIMARY = BG_BASE
BG_CARD = BG_SURFACE
BORDER = BORDER_DEFAULT

_TEMPLATE_NAME = "stirs_dark"


def _register_template() -> None:
    # Always re-register so palette updates pick up
    pio.templates[_TEMPLATE_NAME] = go.layout.Template(
        layout=dict(
            paper_bgcolor=BG_BASE,
            plot_bgcolor=BG_BASE,
            font=dict(family="Inter, system-ui, -apple-system, sans-serif",
                      color=TEXT_BODY, size=12),
            xaxis=dict(
                gridcolor=BORDER_SUBTLE,
                gridwidth=0.5,
                zerolinecolor=BORDER_DEFAULT,
                linecolor=BORDER_DEFAULT,
                tickcolor=BORDER_DEFAULT,
                color=TEXT_MUTED,
                tickfont=dict(family="JetBrains Mono, monospace", size=10),
                title=dict(font=dict(color=TEXT_DIM, size=11)),
            ),
            yaxis=dict(
                gridcolor=BORDER_SUBTLE,
                gridwidth=0.5,
                zerolinecolor=BORDER_DEFAULT,
                linecolor=BORDER_DEFAULT,
                tickcolor=BORDER_DEFAULT,
                color=TEXT_MUTED,
                tickfont=dict(family="JetBrains Mono, monospace", size=10),
                title=dict(font=dict(color=TEXT_DIM, size=11)),
            ),
            colorway=[ACCENT, BLUE, GREEN, RED, AMBER, PURPLE, CYAN, "#ec4899"],
            hoverlabel=dict(
                bgcolor=BG_SURFACE,
                font_color=TEXT_HEADING,
                bordercolor=ACCENT,
                font_family="JetBrains Mono, monospace",
                font_size=11,
            ),
            margin=dict(l=55, r=20, t=30, b=70),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER_SUBTLE,
                borderwidth=0,
                font=dict(color=TEXT_BODY, size=11, family="Inter, sans-serif"),
            ),
            title=dict(font=dict(size=13, color=ACCENT, family="Inter, sans-serif")),
        ),
    )
    pio.templates.default = _TEMPLATE_NAME


_register_template()


def make_curve_chart(
    contracts: list[str],
    current_y: list[float],
    current_label: str,
    compare_y: Optional[list[float]] = None,
    compare_label: Optional[str] = None,
    high_y: Optional[list[float]] = None,
    low_y: Optional[list[float]] = None,
    title: str = "",
    y_title: str = "Price",
    height: int = 460,
    value_decimals: int = 4,
    section_shading: Optional[tuple] = None,
    horizontal_lines: Optional[list] = None,
    horizontal_bands: Optional[list] = None,
    highlight_contract: Optional[str] = None,
) -> go.Figure:
    """Static curve chart — current line plus optional comparison line.

    If `high_y` / `low_y` are provided, render a translucent intraday-range band
    behind the close line. (Used in the default 'no compare, no animate' mode.)

    `connectgaps=True` keeps lines continuous through missing contracts.
    """
    fig = go.Figure()

    # ---- H-L range as a CONNECTED FILL BAND, but ONLY on contracts with valid data.
    # Filtering out NaN positions before plotting prevents the interpolation
    # artifacts (weird flat horizontal segments) we saw with connectgaps=True.
    # The band still appears connected across valid contracts because Plotly's
    # categorical x-axis preserves order — gaps simply skip.
    if high_y is not None and low_y is not None:
        valid_mask = [
            (h is not None and l is not None
             and not pd.isna(h) and not pd.isna(l))
            for h, l in zip(high_y, low_y)
        ]
        valid_x = [c for c, m in zip(contracts, valid_mask) if m]
        valid_high = [h for h, m in zip(high_y, valid_mask) if m]
        valid_low = [l for l, m in zip(low_y, valid_mask) if m]
        if valid_x:
            # Lower edge — invisible
            fig.add_trace(go.Scatter(
                x=valid_x, y=valid_low, mode="lines",
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            # Upper edge — fills toward the previous trace (the lower edge)
            fig.add_trace(go.Scatter(
                x=valid_x, y=valid_high, mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(232, 183, 93, 0.20)",
                name="Daily H–L range",
                hovertemplate=("<b>%{x}</b><br>High: %{y:" + f".{value_decimals}f"
                               + "}<extra></extra>"),
            ))

    # ---- Compare line ----
    if compare_y is not None:
        fig.add_trace(go.Scatter(
            x=contracts,
            y=compare_y,
            mode="lines+markers",
            name=compare_label or "Compare",
            line=dict(color=BLUE, width=2, dash="dot"),
            marker=dict(size=6, color=BLUE, line=dict(color=BG_PRIMARY, width=1)),
            connectgaps=True,
            hovertemplate=("<b>%{x}</b><br>" + (compare_label or "Compare") +
                           ": %{y:" + f".{value_decimals}f" + "}<extra></extra>"),
        ))

    # ---- Current (close) line — drawn on top, clean markers (no error-bar wicks) ----
    fig.add_trace(go.Scatter(
        x=contracts,
        y=current_y,
        mode="lines+markers",
        name=current_label,
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=8, color=GOLD, line=dict(color=BG_PRIMARY, width=1)),
        connectgaps=True,
        hovertemplate=("<b>%{x}</b><br>" + current_label +
                       ": %{y:" + f".{value_decimals}f" + "}<extra></extra>"),
    ))

    # ---- Optional overlays added BEFORE layout to preserve z-order ----
    if section_shading:
        front_end, mid_end = section_shading
        add_section_shading(fig, contracts, front_end, mid_end)
    if horizontal_bands:
        for spec in horizontal_bands:
            add_horizontal_band(fig, spec.get("lower"), spec.get("upper"),
                                spec.get("color", "rgba(232,183,93,0.08)"),
                                spec.get("label"))
    if horizontal_lines:
        for spec in horizontal_lines:
            add_horizontal_line(fig, spec.get("y"), spec.get("color", BLUE),
                                spec.get("label"), spec.get("dash", "dash"))
    if highlight_contract is not None and highlight_contract in contracts:
        try:
            ci = contracts.index(highlight_contract)
            hy = current_y[ci]
            if hy is not None and not pd.isna(hy):
                add_highlight_marker(fig, highlight_contract, hy)
        except Exception:
            pass

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left") if title else None,
        xaxis=dict(type="category", title=None, tickangle=-45,
                   showgrid=True, gridwidth=0.5),
        yaxis=dict(title=y_title, tickformat=f".{value_decimals}f"),
        height=height,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=50, r=20, t=30, b=80),
    )
    return fig


def make_animated_curve_chart(
    contracts: list[str],
    frames_data: dict[str, list[float]],
    title: str = "",
    y_title: str = "Price",
    height: int = 420,
    frame_duration_ms: int = 120,
) -> go.Figure:
    """Animated curve chart — frame per date with play/slider controls.

    frames_data: ordered dict {date_label: [y values per contract]}
    """
    if not frames_data:
        return go.Figure()

    date_labels = list(frames_data.keys())
    initial_label = date_labels[-1]   # start at most recent
    initial_y = frames_data[initial_label]

    # Compute y range across all frames for stable axis
    all_y = [v for vs in frames_data.values() for v in vs if v is not None and not pd.isna(v)]
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 1)
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.5

    base_trace = go.Scatter(
        x=contracts,
        y=initial_y,
        mode="lines+markers",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=8, color=GOLD, line=dict(color=BG_PRIMARY, width=1)),
        connectgaps=True,
        name="Curve",
        hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
    )

    plotly_frames = []
    for label in date_labels:
        plotly_frames.append(go.Frame(
            name=label,
            data=[go.Scatter(
                x=contracts,
                y=frames_data[label],
                mode="lines+markers",
                line=dict(color=GOLD, width=2.5),
                marker=dict(size=8, color=GOLD, line=dict(color=BG_PRIMARY, width=1)),
                connectgaps=True,
                hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
            )],
            layout=dict(annotations=[dict(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper", x=0.99, y=0.97,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(color=GOLD, size=14, family="JetBrains Mono, monospace"),
                bgcolor=BG_PRIMARY, bordercolor=GOLD, borderwidth=1, borderpad=4,
            )]),
        ))

    fig = go.Figure(data=[base_trace], frames=plotly_frames)

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        xaxis=dict(type="category", title=None, tickangle=-45),
        yaxis=dict(title=y_title, range=[y_min - y_pad, y_max + y_pad]),
        height=height,
        hovermode="x unified",
        showlegend=False,
        annotations=[dict(
            text=f"<b>{initial_label}</b>",
            xref="paper", yref="paper", x=0.99, y=0.97,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(color=GOLD, size=14, family="JetBrains Mono, monospace"),
            bgcolor=BG_PRIMARY, bordercolor=GOLD, borderwidth=1, borderpad=4,
        )],
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.0, y=-0.20, xanchor="left", yanchor="top",
            pad=dict(t=4, r=4),
            bgcolor=BG_CARD,
            bordercolor=BORDER,
            font=dict(color=TEXT_PRIMARY, size=12),
            buttons=[
                dict(label="▶ Play",  method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True),
                                      transition=dict(duration=60),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=len(date_labels) - 1,
            x=0.10, y=-0.20, xanchor="left", yanchor="top", len=0.85,
            pad=dict(t=4, r=4),
            currentvalue=dict(visible=False),
            bgcolor=BG_CARD,
            bordercolor=BORDER,
            tickcolor=TEXT_SECONDARY,
            font=dict(color=TEXT_SECONDARY, size=10),
            steps=[dict(method="animate",
                        args=[[label],
                              dict(frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0), mode="immediate")],
                        label=label) for label in date_labels],
        )],
    )
    return fig


# =============================================================================
# Section shading helper (used by Mode 1 & others)
# =============================================================================

def add_section_shading(fig: go.Figure, contracts: list, front_end: int, mid_end: int) -> None:
    """Add subtle background shading + boundary lines for front/mid/back regions.

    Uses Plotly shapes (vertical rectangles spanning the y-range).
    """
    n = len(contracts)
    if n == 0:
        return
    fe = min(front_end, n)
    me = min(mid_end, n)
    if me <= fe:
        me = min(fe + 1, n)

    # Section bands as vrects with clearly-distinct colors at moderate alpha
    band_specs = [
        (-0.5, fe - 0.5, "rgba(251, 191, 36, 0.14)", "rgba(251, 191, 36, 0.5)", "FRONT", fe),    # amber
        (fe - 0.5, me - 0.5, "rgba(34, 211, 238, 0.11)", "rgba(34, 211, 238, 0.45)", "MID", me - fe),  # cyan
        (me - 0.5, n - 0.5, "rgba(167, 139, 250, 0.11)", "rgba(167, 139, 250, 0.45)", "BACK", n - me),  # purple
    ]
    for x0, x1, fill, label_color, label, count in band_specs:
        if x1 <= x0:
            continue
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=fill,
            line_width=0,
            layer="below",
            annotation_text=f"<b>{label}</b> · {count}c",
            annotation_position="top left",
            annotation_font=dict(size=10, color=label_color, family="JetBrains Mono, monospace"),
        )

    # Boundary lines — slightly more visible
    if fe < n:
        fig.add_vline(x=fe - 0.5, line_dash="dot",
                      line_color="rgba(255,255,255,0.18)", line_width=1)
    if me < n:
        fig.add_vline(x=me - 0.5, line_dash="dot",
                      line_color="rgba(255,255,255,0.18)", line_width=1)


def add_horizontal_band(fig: go.Figure, lower: float, upper: float, color: str,
                        label: Optional[str] = None) -> None:
    """Add a horizontal shaded band between `lower` and `upper`."""
    if lower is None or upper is None:
        return
    fig.add_hrect(
        y0=min(lower, upper), y1=max(lower, upper),
        fillcolor=color, line_width=0, layer="below",
        annotation_text=label or "",
        annotation_position="right",
        annotation_font=dict(size=9, color=TEXT_DIM, family="JetBrains Mono, monospace"),
    )


def add_horizontal_line(fig: go.Figure, y: float, color: str, label: Optional[str] = None,
                        dash: str = "dash") -> None:
    """Add a horizontal line at y."""
    if y is None:
        return
    fig.add_hline(
        y=y, line_color=color, line_dash=dash, line_width=1, opacity=0.7,
        annotation_text=label or "", annotation_position="right",
        annotation_font=dict(size=9, color=color, family="JetBrains Mono, monospace"),
    )


def add_highlight_marker(fig: go.Figure, contract: str, y: float, label: str = "") -> None:
    """Add a single highlighted marker (used by jump-to-contract search)."""
    if contract is None or y is None:
        return
    fig.add_trace(go.Scatter(
        x=[contract], y=[y], mode="markers",
        marker=dict(size=18, color=ACCENT, symbol="circle-open",
                    line=dict(color=ACCENT, width=3)),
        showlegend=False,
        hoverinfo="skip",
        name=label,
    ))


# =============================================================================
# Mode 3 — Multi-date overlay
# =============================================================================

_MULTI_DATE_PALETTE = [
    "#a78bfa",   # purple — oldest
    "#60a5fa",   # blue
    "#22d3ee",   # cyan
    "#4ade80",   # green
    "#fbbf24",   # amber
    "#e8b75d",   # gold — most recent
]


def make_multi_date_curve_chart(
    contracts: list,
    date_labels: list,
    series_data: list,
    y_title: str = "Price",
    height: int = 460,
    section_shading: Optional[tuple] = None,
    horizontal_lines: Optional[list] = None,
    horizontal_bands: Optional[list] = None,
) -> go.Figure:
    """Multi-date overlay: each series is a separate line, color from cool→warm by recency."""
    fig = go.Figure()

    n = len(date_labels)
    palette = _MULTI_DATE_PALETTE
    # interpolate from palette if more dates than colors
    if n > len(palette):
        # simple cycle
        colors = [palette[i % len(palette)] for i in range(n)]
    else:
        # sample evenly
        step = max(1, len(palette) // n)
        colors = [palette[min(len(palette) - 1, i * step)] for i in range(n)]

    for i, (label, ys) in enumerate(zip(date_labels, series_data)):
        is_latest = (i == n - 1)
        fig.add_trace(go.Scatter(
            x=contracts, y=ys, mode="lines+markers",
            name=label,
            line=dict(color=ACCENT if is_latest else colors[i],
                      width=2.5 if is_latest else 1.6,
                      dash="solid" if is_latest else "dot"),
            marker=dict(size=7 if is_latest else 4,
                        color=ACCENT if is_latest else colors[i],
                        line=dict(color=BG_BASE, width=1)),
            connectgaps=True,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>",
        ))

    if section_shading:
        front_end, mid_end = section_shading
        add_section_shading(fig, contracts, front_end, mid_end)

    if horizontal_lines:
        for spec in horizontal_lines:
            add_horizontal_line(fig, spec.get("y"), spec.get("color", BLUE),
                                spec.get("label"), spec.get("dash", "dash"))
    if horizontal_bands:
        for spec in horizontal_bands:
            add_horizontal_band(fig, spec.get("lower"), spec.get("upper"),
                                spec.get("color", "rgba(232,183,93,0.08)"),
                                spec.get("label"))

    fig.update_layout(
        xaxis=dict(type="category", title=None, tickangle=-45),
        yaxis=dict(title=y_title, tickformat=".4f"),
        height=height, hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=55, r=20, t=30, b=80),
    )
    return fig


# =============================================================================
# Mode 4 — Ribbon (percentile envelope)
# =============================================================================

def make_ribbon_chart(
    contracts: list,
    today_y: list,
    today_label: str,
    bands: dict,                          # {p05, p25, p50, p75, p95, mean}
    y_title: str = "Price",
    height: int = 480,
    section_shading: Optional[tuple] = None,
    horizontal_lines: Optional[list] = None,
    horizontal_bands: Optional[list] = None,
    show_extras: Optional[dict] = None,   # {mean: bool, recent_avg: list, min_max: bool}
) -> go.Figure:
    """Percentile-envelope curve: today's line over historical p05/25/50/75/95 bands."""
    fig = go.Figure()

    p05 = [bands.get("p05", {}).get(c) for c in contracts] if bands else [None] * len(contracts)
    p25 = [bands.get("p25", {}).get(c) for c in contracts] if bands else [None] * len(contracts)
    p50 = [bands.get("p50", {}).get(c) for c in contracts] if bands else [None] * len(contracts)
    p75 = [bands.get("p75", {}).get(c) for c in contracts] if bands else [None] * len(contracts)
    p95 = [bands.get("p95", {}).get(c) for c in contracts] if bands else [None] * len(contracts)

    # 5-95 outer band — clearly visible but lighter than IQR
    fig.add_trace(go.Scatter(x=contracts, y=p05, mode="lines",
                             line=dict(width=0), connectgaps=True,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=contracts, y=p95, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(232, 183, 93, 0.13)",
                             connectgaps=True, name="5–95 %ile",
                             hovertemplate="<b>%{x}</b><br>p95: %{y:.4f}<extra></extra>"))
    # 25-75 IQR band — denser
    fig.add_trace(go.Scatter(x=contracts, y=p25, mode="lines",
                             line=dict(width=0), connectgaps=True,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=contracts, y=p75, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(232, 183, 93, 0.28)",
                             connectgaps=True, name="25–75 %ile (IQR)",
                             hovertemplate="<b>%{x}</b><br>p75: %{y:.4f}<extra></extra>"))
    # Median line
    fig.add_trace(go.Scatter(x=contracts, y=p50, mode="lines",
                             line=dict(color=TEXT_MUTED, width=1.2, dash="dot"),
                             connectgaps=True, name="Median",
                             hovertemplate="<b>%{x}</b><br>p50: %{y:.4f}<extra></extra>"))

    if show_extras and show_extras.get("mean"):
        mean_y = [bands.get("mean", {}).get(c) for c in contracts]
        fig.add_trace(go.Scatter(x=contracts, y=mean_y, mode="lines",
                                 line=dict(color=BLUE, width=1.2, dash="dash"),
                                 connectgaps=True, name="Mean",
                                 hovertemplate="<b>%{x}</b><br>mean: %{y:.4f}<extra></extra>"))
    if show_extras and show_extras.get("recent_avg") is not None:
        ra = show_extras["recent_avg"]
        fig.add_trace(go.Scatter(x=contracts, y=ra, mode="lines+markers",
                                 line=dict(color=GREEN, width=1.5),
                                 marker=dict(size=4, color=GREEN),
                                 connectgaps=True, name="Recent avg",
                                 hovertemplate="<b>%{x}</b><br>recent: %{y:.4f}<extra></extra>"))

    # Today's line on top
    fig.add_trace(go.Scatter(x=contracts, y=today_y, mode="lines+markers",
                             name=today_label,
                             line=dict(color=ACCENT, width=2.5),
                             marker=dict(size=8, color=ACCENT,
                                         line=dict(color=BG_BASE, width=1)),
                             connectgaps=True,
                             hovertemplate=f"<b>%{{x}}</b><br>{today_label}: %{{y:.4f}}<extra></extra>"))

    if section_shading:
        front_end, mid_end = section_shading
        add_section_shading(fig, contracts, front_end, mid_end)

    if horizontal_lines:
        for spec in horizontal_lines:
            add_horizontal_line(fig, spec.get("y"), spec.get("color", BLUE),
                                spec.get("label"), spec.get("dash", "dash"))
    if horizontal_bands:
        for spec in horizontal_bands:
            add_horizontal_band(fig, spec.get("lower"), spec.get("upper"),
                                spec.get("color"), spec.get("label"))

    fig.update_layout(
        xaxis=dict(type="category", title=None, tickangle=-45),
        yaxis=dict(title=y_title, tickformat=".4f"),
        height=height, hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=55, r=20, t=30, b=80),
    )
    return fig


# =============================================================================
# Mode 5 — Δ bar chart (per-contract change)
# =============================================================================

def make_change_bar_chart(
    contracts: list,
    changes: list,
    y_title: str = "Change (bp)",
    height: int = 220,
) -> go.Figure:
    """Per-contract change as a bar chart, color-coded by sign."""
    colors = [GREEN if (c is not None and c > 0)
              else (RED if (c is not None and c < 0) else TEXT_DIM)
              for c in changes]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contracts, y=changes, marker_color=colors,
        marker_line_width=0,
        text=[f"{c:+.2f}" if c is not None and not pd.isna(c) else "" for c in changes],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono, monospace", color=TEXT_MUTED),
        hovertemplate="<b>%{x}</b><br>Δ: %{y:.3f}<extra></extra>",
        name="Change",
    ))
    fig.update_layout(
        xaxis=dict(type="category", tickangle=-45, title=None, showgrid=False),
        yaxis=dict(title=y_title, zeroline=True, zerolinecolor=TEXT_DIM, zerolinewidth=1),
        height=height, showlegend=False,
        margin=dict(l=55, r=20, t=15, b=70),
        bargap=0.25,
    )
    return fig


# =============================================================================
# Mode 6 — Pack chart
# =============================================================================

def make_pack_chart(
    contracts: list,
    current_y: list,
    pack_groups: list,           # [(pack_name, [symbols])]
    pack_means: dict,            # {pack_name: mean_value}
    pack_colors: list,
    y_title: str = "Price",
    height: int = 480,
    section_shading: Optional[tuple] = None,
) -> go.Figure:
    """Standard curve plus pack-mean markers overlaid (larger, distinct color per pack)."""
    fig = go.Figure()

    # base curve
    fig.add_trace(go.Scatter(
        x=contracts, y=current_y, mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=7, color=ACCENT, line=dict(color=BG_BASE, width=1)),
        connectgaps=True, name="Curve",
        hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
    ))

    # Pack-mean markers — placed at the middle contract of each pack
    for i, (pack_name, syms) in enumerate(pack_groups):
        if not syms:
            continue
        mean_val = pack_means.get(pack_name)
        if mean_val is None or pd.isna(mean_val):
            continue
        mid_sym = syms[len(syms) // 2]
        pack_color = pack_colors[i] if i < len(pack_colors) else BLUE
        fig.add_trace(go.Scatter(
            x=[mid_sym], y=[mean_val], mode="markers",
            marker=dict(size=18, color=pack_color, symbol="diamond",
                        line=dict(color=BG_BASE, width=2)),
            name=f"{pack_name} mean",
            hovertemplate=f"<b>{pack_name}</b><br>{', '.join(syms)}<br>Mean: %{{y:.4f}}<extra></extra>",
        ))

    if section_shading:
        front_end, mid_end = section_shading
        add_section_shading(fig, contracts, front_end, mid_end)

    fig.update_layout(
        xaxis=dict(type="category", title=None, tickangle=-45),
        yaxis=dict(title=y_title, tickformat=".4f"),
        height=height, hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=55, r=20, t=30, b=80),
    )
    return fig


# =============================================================================
# Mode 8 — Volume / OI Δ chart
# =============================================================================

# =============================================================================
# Heatmap (date × contract)
# =============================================================================

def make_heatmap_chart(
    matrix: pd.DataFrame,
    title: str = "",
    height: int = 600,
    colorscale: str = "RdBu",
    zmid: float = None,
    show_values: bool = False,
    value_format: str = ".3f",
) -> go.Figure:
    """Generic heatmap. Expects a 2D DataFrame: rows = y-axis, cols = x-axis.

    For curve-evolution: rows = dates (most-recent at top), cols = contracts.
    For calendar-spread matrix: rows = front contract, cols = back contract.
    """
    if matrix.empty:
        return go.Figure()

    # Convert dates to strings for the y-axis if needed
    y_labels = matrix.index
    if isinstance(matrix.index, pd.DatetimeIndex):
        y_labels = matrix.index.strftime("%Y-%m-%d")

    fig = go.Figure()
    z = matrix.values
    fig.add_trace(go.Heatmap(
        z=z,
        x=list(matrix.columns),
        y=list(y_labels),
        colorscale=colorscale,
        zmid=zmid,
        colorbar=dict(
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
            outlinewidth=0,
        ),
        text=z if show_values else None,
        texttemplate=f"%{{text:{value_format}}}" if show_values else None,
        textfont=dict(size=9, family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b> · <b>%{x}</b><br>" + f"%{{z:{value_format}}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left") if title else None,
        xaxis=dict(type="category", title=None, tickangle=-45, side="bottom"),
        yaxis=dict(title=None, autorange="reversed",
                   tickfont=dict(family="JetBrains Mono, monospace", size=9)),
        height=height,
        margin=dict(l=80, r=20, t=30, b=80),
    )
    return fig


# =============================================================================
# Calendar-spread / fly matrix (2D)
# =============================================================================

def make_calendar_matrix_chart(
    rows_label: list,
    cols_label: list,
    z: list,
    title: str = "",
    height: int = 500,
    colorscale: str = "RdBu",
    zmid: float = 0,
    value_format: str = ".4f",
    show_values: bool = True,
) -> go.Figure:
    """Calendar-spread matrix heatmap: rows = front-leg, cols = back-leg, z = spread value."""
    fig = go.Figure(go.Heatmap(
        z=z, x=cols_label, y=rows_label,
        colorscale=colorscale, zmid=zmid,
        colorbar=dict(
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_MUTED),
            outlinewidth=0,
        ),
        text=z if show_values else None,
        texttemplate=f"%{{text:{value_format}}}" if show_values else None,
        textfont=dict(size=9, family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y} → %{x}</b><br>" + f"%{{z:{value_format}}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left") if title else None,
        xaxis=dict(type="category", title="Back leg", tickangle=-45),
        yaxis=dict(type="category", title="Front leg", autorange="reversed"),
        height=height,
        margin=dict(l=80, r=20, t=30, b=80),
    )
    return fig


# =============================================================================
# Z-score curve view (Ribbon variant)
# =============================================================================

def make_zscore_curve_chart(
    contracts: list,
    zscores: list,
    title: str = "",
    height: int = 460,
    section_shading: Optional[tuple] = None,
) -> go.Figure:
    """Bar chart of z-scores per contract with ±1σ, ±2σ horizontal threshold lines.

    Bar color: green for negative z (cheap), red for positive z (rich), tinted.
    Shows threshold lines at ±1σ and ±2σ.
    """
    fig = go.Figure()
    bar_colors = []
    for z in zscores:
        if z is None or pd.isna(z):
            bar_colors.append(TEXT_DIM)
        elif abs(z) >= 2:
            bar_colors.append(RED if z > 0 else "#22c55e")
        elif abs(z) >= 1:
            bar_colors.append(AMBER)
        else:
            bar_colors.append(BLUE)

    fig.add_trace(go.Bar(
        x=contracts, y=zscores, marker_color=bar_colors, marker_line_width=0,
        text=[f"{z:+.2f}σ" if z is not None and not pd.isna(z) else ""
              for z in zscores],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono, monospace", color=TEXT_MUTED),
        hovertemplate="<b>%{x}</b><br>z: %{y:.3f}σ<extra></extra>",
    ))

    # Threshold lines
    for level, alpha, dash in [(2, 0.6, "dash"), (1, 0.35, "dot"),
                                (-1, 0.35, "dot"), (-2, 0.6, "dash")]:
        fig.add_hline(y=level,
                      line_color=f"rgba(255,255,255,{alpha})",
                      line_dash=dash, line_width=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.55)", line_width=1.2)

    if section_shading:
        front_end, mid_end = section_shading
        add_section_shading(fig, contracts, front_end, mid_end)

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left") if title else None,
        xaxis=dict(type="category", title=None, tickangle=-45),
        yaxis=dict(title="Z-score (σ)", zeroline=False, showgrid=True,
                   gridwidth=0.5, gridcolor=BORDER_SUBTLE),
        height=height, showlegend=False,
        margin=dict(l=55, r=20, t=30, b=80),
    )
    return fig


# =============================================================================
# Carry-colored standard curve (markers tinted by carry value)
# =============================================================================

def make_carry_colored_curve(
    contracts: list,
    current_y: list,
    carry_per_day: list,
    y_title: str = "Price",
    height: int = 460,
    section_shading: Optional[tuple] = None,
    horizontal_lines: Optional[list] = None,
    horizontal_bands: Optional[list] = None,
) -> go.Figure:
    """Curve with markers colored by carry/day (cool=low/negative, warm=high/positive).

    Robust to missing carry values: None / NaN are passed to Plotly as NaN (renders
    as transparent on the colorscale). If ALL carry values are missing, falls back
    to a uniform-colored curve so the chart still renders.
    """
    import math
    fig = go.Figure()

    # Underlying line — neutral
    fig.add_trace(go.Scatter(
        x=contracts, y=current_y, mode="lines",
        line=dict(color="rgba(232, 183, 93, 0.4)", width=1.5),
        connectgaps=True, showlegend=False, hoverinfo="skip",
    ))

    # Convert None → NaN so Plotly's color validator accepts the array.
    carry_clean = [float("nan") if (c is None or pd.isna(c)) else float(c)
                   for c in carry_per_day]
    valid_carry = [c for c in carry_clean if not math.isnan(c)]

    if not valid_carry:
        # No carry data — render a plain accent-colored marker layer (no colorbar)
        fig.add_trace(go.Scatter(
            x=contracts, y=current_y, mode="markers",
            marker=dict(size=8, color=ACCENT, line=dict(color=BG_BASE, width=1)),
            hovertemplate="<b>%{x}</b><br>%{y:.4f}<br>"
                          "<i>(carry unavailable)</i><extra></extra>",
            showlegend=False,
        ))
    else:
        # Symmetric bound around 0 for a balanced diverging colorscale
        c_max = max(abs(min(valid_carry)), abs(max(valid_carry)))
        c_min = -c_max if c_max > 0 else min(valid_carry)
        if c_max == 0:
            c_min, c_max = -1.0, 1.0

        fig.add_trace(go.Scatter(
            x=contracts, y=current_y, mode="markers",
            marker=dict(
                size=11, line=dict(color=BG_BASE, width=1.2),
                color=carry_clean,
                colorscale="RdYlGn",
                cmin=c_min, cmax=c_max,
                colorbar=dict(
                    title=dict(text="Carry<br>(bp/day)",
                               font=dict(size=10, color=TEXT_MUTED)),
                    tickfont=dict(family="JetBrains Mono, monospace",
                                  size=10, color=TEXT_MUTED),
                    outlinewidth=0, thickness=12, len=0.5, y=0.5,
                ),
                showscale=True,
            ),
            customdata=carry_clean,
            hovertemplate="<b>%{x}</b><br>Price: %{y:.4f}<br>"
                          "Carry: %{customdata:.3f} bp/day<extra></extra>",
            showlegend=False,
        ))

    if section_shading:
        front_end, mid_end = section_shading
        add_section_shading(fig, contracts, front_end, mid_end)
    if horizontal_lines:
        for spec in horizontal_lines:
            add_horizontal_line(fig, spec.get("y"), spec.get("color", BLUE),
                                spec.get("label"), spec.get("dash", "dash"))
    if horizontal_bands:
        for spec in horizontal_bands:
            add_horizontal_band(fig, spec.get("lower"), spec.get("upper"),
                                spec.get("color"), spec.get("label"))

    fig.update_layout(
        xaxis=dict(type="category", title=None, tickangle=-45),
        yaxis=dict(title=y_title, tickformat=".4f"),
        height=height, hovermode="closest",
        margin=dict(l=55, r=80, t=30, b=80),
    )
    return fig


def make_volume_delta_chart(
    contracts: list,
    delta_values: list,
    label: str = "Volume Δ",
    height: int = 220,
) -> go.Figure:
    """Volume or OI delta bar chart; colored green/red by sign."""
    colors = [GREEN if (v is not None and v > 0)
              else (RED if (v is not None and v < 0) else TEXT_DIM)
              for v in delta_values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contracts, y=delta_values, marker_color=colors, marker_line_width=0,
        text=[f"{v:+.0f}" if v is not None and not pd.isna(v) else "" for v in delta_values],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono, monospace", color=TEXT_MUTED),
        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.0f}}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(type="category", tickangle=-45, title=None, showgrid=False),
        yaxis=dict(title=label, zeroline=True, zerolinecolor=TEXT_DIM, zerolinewidth=1),
        height=height, showlegend=False,
        margin=dict(l=55, r=20, t=15, b=70),
        bargap=0.25,
    )
    return fig


# =============================================================================
# Proximity / Z-score chart factories (Analysis subtabs)
# =============================================================================

def make_proximity_ribbon_chart(
    rows: list[dict],
    side: str,
    title: str = "",
    height: int = 360,
) -> go.Figure:
    """Horizontal-bar ribbon of dist-to-extreme in ATR units.

    Each row dict needs at least ``symbol``, ``dist_atr``, ``flag``; ``dist_bp``,
    ``streak``, ``fresh_break``, ``failed_break`` are also displayed if present.
    Bars colored by flag intensity (AT=red, NEAR=amber, APPROACHING=blue, FAR=dim).
    Threshold lines at 0.25 / 0.5 / 1.0 ATR.
    """
    fig = go.Figure()
    if not rows:
        fig.update_layout(height=height,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            xref="paper", yref="paper",
                                            showarrow=False,
                                            font=dict(color=TEXT_DIM, size=12))],
                          margin=dict(l=10, r=10, t=10, b=10))
        return fig
    syms = [r["symbol"] for r in rows]
    dists = [r.get("dist_atr") if r.get("dist_atr") is not None else 0.0 for r in rows]
    colors = []
    for r in rows:
        f = r.get("flag", "—")
        if f == "AT":
            colors.append(RED)
        elif f == "NEAR":
            colors.append(AMBER)
        elif f == "APPROACHING":
            colors.append(BLUE)
        else:
            colors.append(TEXT_DIM)

    text_labels = []
    for r in rows:
        d_atr = r.get("dist_atr")
        d_bp = r.get("dist_bp")
        flag = r.get("flag", "")
        streak = r.get("streak") or 0
        if d_atr is None:
            text_labels.append("")
        else:
            extras = []
            if streak >= 1:
                extras.append(f"⛓{streak}d")
            if r.get("fresh_break"):
                extras.append("FRESH")
            if r.get("failed_break"):
                extras.append("FAILED")
            extras_str = (" · " + " · ".join(extras)) if extras else ""
            text_labels.append(
                f"{d_atr:.2f}σ · {d_bp:+.2f}bp · {flag}{extras_str}"
                if d_bp is not None else f"{d_atr:.2f}σ · {flag}{extras_str}"
            )

    syms_disp = list(reversed(syms))
    dists_disp = list(reversed(dists))
    colors_disp = list(reversed(colors))
    text_disp = list(reversed(text_labels))

    fig.add_trace(go.Bar(
        x=dists_disp, y=syms_disp,
        orientation="h",
        marker=dict(color=colors_disp, line=dict(width=0)),
        text=text_disp,
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_BODY),
        hovertemplate="<b>%{y}</b><br>dist: %{x:.3f} ATR<extra></extra>",
        showlegend=False,
    ))

    for x_thresh, name, alpha in [(0.25, "AT", 0.55), (0.5, "NEAR", 0.45), (1.0, "APPROACH", 0.30)]:
        fig.add_vline(x=x_thresh, line_color=f"rgba(255,255,255,{alpha})",
                      line_dash="dot", line_width=1,
                      annotation_text=name, annotation_position="top right",
                      annotation_font=dict(size=9, color=TEXT_DIM,
                                           family="JetBrains Mono, monospace"))

    side_label = "high" if side.lower() == "high" else "low"
    fig.update_layout(
        title=dict(text=title or f"Closest to {side_label}", x=0, xanchor="left",
                   font=dict(size=12, color=ACCENT)),
        xaxis=dict(title=f"dist to {side_label} (ATR)", zeroline=False,
                   showgrid=True, gridcolor=BORDER_SUBTLE,
                   tickfont=dict(family="JetBrains Mono, monospace", size=9)),
        yaxis=dict(title=None, automargin=True,
                   tickfont=dict(family="JetBrains Mono, monospace", size=10),
                   showgrid=False),
        height=height,
        margin=dict(l=10, r=160, t=30, b=40),
        bargap=0.18,
        showlegend=False,
    )
    return fig


def make_density_heatmap_chart(
    matrix: pd.DataFrame,
    title: str = "",
    height: int = 320,
    max_pct: float = 1.0,
    show_values: bool = True,
) -> go.Figure:
    """Density heatmap — rows × cols × % AT/NEAR (0..1). Reds intensifying with %."""
    if matrix is None or matrix.empty:
        fig = go.Figure()
        fig.update_layout(height=height,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            xref="paper", yref="paper",
                                            showarrow=False,
                                            font=dict(color=TEXT_DIM, size=12))])
        return fig
    z = matrix.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(matrix.columns), y=list(matrix.index),
        colorscale=[[0.0, BG_BASE], [0.3, "#3a2422"], [0.65, "#a83f3f"], [1.0, RED]],
        zmin=0.0, zmax=max_pct,
        colorbar=dict(
            title=dict(text="% AT/NEAR", font=dict(size=10, color=TEXT_MUTED)),
            tickfont=dict(family="JetBrains Mono, monospace", size=9, color=TEXT_MUTED),
            outlinewidth=0, thickness=10, len=0.7, y=0.5,
            tickformat=".0%",
        ),
        text=[[f"{v*100:.0f}%" if (not pd.isna(v) and v > 0) else ""
               for v in row] for row in z] if show_values else None,
        texttemplate="%{text}" if show_values else None,
        textfont=dict(size=10, family="JetBrains Mono, monospace", color=TEXT_HEADING),
        hovertemplate="<b>%{y} · %{x}</b><br>%{z:.1%} AT/NEAR<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12, color=ACCENT)),
        xaxis=dict(type="category", title=None, tickangle=0, side="bottom"),
        yaxis=dict(type="category", title=None, autorange="reversed",
                   tickfont=dict(family="JetBrains Mono, monospace", size=10)),
        height=height,
        margin=dict(l=80, r=20, t=30, b=40),
    )
    return fig


def make_confluence_matrix_chart(
    matrix: pd.DataFrame,
    pattern_col: pd.Series = None,
    title: str = "",
    height: int = 480,
    z_mode: bool = False,
) -> go.Figure:
    """Confluence matrix: rows = contracts, cols = lookbacks, cells = position-in-range
    (0..1) or z-score, colored on a diverging palette around 0.5 (PIR) or 0 (z).

    ``pattern_col`` (optional, indexed same as matrix.index) is rendered as a
    labeled column at the right via annotations.
    """
    if matrix is None or matrix.empty:
        fig = go.Figure()
        fig.update_layout(height=height,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                            xref="paper", yref="paper",
                                            showarrow=False,
                                            font=dict(color=TEXT_DIM, size=12))])
        return fig
    z = matrix.values.astype(float)
    if z_mode:
        z_max = float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 2.0
        z_max = max(2.0, z_max)
        zmin, zmax, zmid = -z_max, z_max, 0.0
        cs = "RdBu_r"
        fmt = ".2f"
    else:
        zmin, zmax, zmid = 0.0, 1.0, 0.5
        cs = "RdBu_r"
        fmt = ".2f"

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in matrix.columns], y=list(matrix.index),
        colorscale=cs, zmin=zmin, zmax=zmax, zmid=zmid,
        colorbar=dict(
            title=dict(text=("z-score" if z_mode else "PIR"),
                       font=dict(size=10, color=TEXT_MUTED)),
            tickfont=dict(family="JetBrains Mono, monospace", size=9, color=TEXT_MUTED),
            outlinewidth=0, thickness=10, len=0.7, y=0.5,
        ),
        text=[[f"{v:{fmt}}" if not pd.isna(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=9, family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y} · %{x}</b><br>%{z:" + fmt + "}<extra></extra>",
    ))

    if pattern_col is not None and len(pattern_col) > 0:
        # x position outside the heatmap plot area
        last_col_idx = len(matrix.columns) - 1
        col_map = {
            "PERSISTENT": RED, "ACCELERATING": RED, "DECELERATING": AMBER,
            "FRESH": BLUE, "DRIFTED": AMBER, "REVERTING": GREEN,
            "DIVERGENT": PURPLE, "STABLE": TEXT_DIM, "MIXED": TEXT_MUTED,
        }
        for i, label in enumerate(pattern_col.values):
            if label is None or pd.isna(label):
                continue
            color = col_map.get(str(label), TEXT_BODY)
            fig.add_annotation(
                x=last_col_idx + 0.7, y=matrix.index[i],
                xref="x", yref="y",
                text=f"<b>{label}</b>",
                showarrow=False, xanchor="left",
                font=dict(size=10, family="JetBrains Mono, monospace", color=color),
            )

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12, color=ACCENT)),
        xaxis=dict(type="category", title="lookback (bars)", side="bottom",
                   range=[-0.5, len(matrix.columns) - 0.5 + 2.5]),
        yaxis=dict(type="category", title=None, autorange="reversed",
                   tickfont=dict(family="JetBrains Mono, monospace", size=10)),
        height=height,
        margin=dict(l=80, r=20, t=30, b=40),
    )
    return fig


def make_distribution_histogram(
    values: list,
    title: str = "",
    height: int = 220,
    bin_size: float = 0.25,
    x_title: str = "z-score",
    threshold_lines: Optional[list] = None,
) -> go.Figure:
    """Distribution histogram — used for Z(30d) across universe, or PIR distribution."""
    fig = go.Figure()
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        fig.update_layout(height=height, annotations=[dict(
            text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=TEXT_DIM, size=12))])
        return fig
    fig.add_trace(go.Histogram(
        x=vals, xbins=dict(size=bin_size),
        marker=dict(color=ACCENT, line=dict(color=BG_BASE, width=0.5)),
        opacity=0.85,
        hovertemplate=f"<b>%{{x}}</b><br>n: %{{y}}<extra></extra>",
    ))
    if threshold_lines:
        for spec in threshold_lines:
            fig.add_vline(x=spec.get("x"), line_color=spec.get("color", TEXT_MUTED),
                          line_dash=spec.get("dash", "dash"), line_width=1,
                          annotation_text=spec.get("label", ""),
                          annotation_position="top",
                          annotation_font=dict(size=9, color=spec.get("color", TEXT_MUTED),
                                               family="JetBrains Mono, monospace"))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left",
                   font=dict(size=12, color=ACCENT)) if title else None,
        xaxis=dict(title=x_title, showgrid=True, gridcolor=BORDER_SUBTLE,
                   tickfont=dict(family="JetBrains Mono, monospace", size=9)),
        yaxis=dict(title="count", showgrid=True, gridcolor=BORDER_SUBTLE,
                   tickfont=dict(family="JetBrains Mono, monospace", size=9)),
        height=height,
        margin=dict(l=50, r=20, t=30, b=40),
        bargap=0.05,
    )
    return fig


def make_mean_bands_chart(
    history: pd.DataFrame,
    mean: float, std: float,
    title: str = "",
    height: int = 320,
    band_unit_label: str = "",
    half_life: Optional[float] = None,
) -> go.Figure:
    """Per-contract close + mean ± 1σ ± 2σ horizontal bands (used in drill-down)."""
    fig = go.Figure()
    if history is None or history.empty or mean is None or std is None:
        fig.update_layout(height=height, annotations=[dict(
            text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=TEXT_DIM, size=12))])
        return fig
    fig.add_hrect(y0=mean - 2 * std, y1=mean + 2 * std,
                  fillcolor="rgba(232,183,93,0.06)", line_width=0, layer="below",
                  annotation_text="±2σ", annotation_position="right",
                  annotation_font=dict(size=9, color=TEXT_DIM,
                                       family="JetBrains Mono, monospace"))
    fig.add_hrect(y0=mean - std, y1=mean + std,
                  fillcolor="rgba(232,183,93,0.12)", line_width=0, layer="below",
                  annotation_text="±1σ", annotation_position="right",
                  annotation_font=dict(size=9, color=TEXT_DIM,
                                       family="JetBrains Mono, monospace"))
    fig.add_hline(y=mean, line_color=ACCENT, line_dash="dash", line_width=1,
                  annotation_text=f"μ {mean:.4f}", annotation_position="right",
                  annotation_font=dict(size=10, color=ACCENT,
                                       family="JetBrains Mono, monospace"))
    fig.add_trace(go.Scatter(
        x=history["date"] if "date" in history.columns else history.index,
        y=history["close"] if "close" in history.columns else history.iloc[:, 0],
        mode="lines", line=dict(color=BLUE, width=1.6),
        name="close",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.4f}<extra></extra>",
    ))
    if half_life is not None and np.isfinite(half_life):
        fig.add_annotation(
            text=f"OU half-life: <b>{half_life:.1f}d</b>",
            xref="paper", yref="paper", x=0.99, y=0.97,
            xanchor="right", yanchor="top", showarrow=False,
            font=dict(color=ACCENT, size=11, family="JetBrains Mono, monospace"),
            bgcolor=BG_SURFACE, bordercolor=BORDER_DEFAULT, borderwidth=1, borderpad=4,
        )
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12, color=ACCENT)),
        xaxis=dict(title=None, showgrid=True, gridcolor=BORDER_SUBTLE),
        yaxis=dict(title=band_unit_label or "Price", showgrid=True, gridcolor=BORDER_SUBTLE,
                   tickformat=".4f"),
        height=height, hovermode="x unified", showlegend=False,
        margin=dict(l=55, r=70, t=30, b=40),
    )
    return fig


def make_proximity_drill_chart(
    history: pd.DataFrame,
    lookbacks: list[int],
    title: str = "",
    height: int = 360,
) -> go.Figure:
    """Per-contract close + rolling H-L bands at multiple lookbacks layered (up to 3)."""
    fig = go.Figure()
    if history is None or history.empty:
        fig.update_layout(height=height, annotations=[dict(
            text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=TEXT_DIM, size=12))])
        return fig
    df = history.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    palette = ["rgba(232,183,93,0.13)", "rgba(96,165,250,0.10)", "rgba(167,139,250,0.08)"]
    layered = sorted(set([n for n in lookbacks if n >= 5]))[:3]
    for i, n in enumerate(layered):
        roll_high = df["close"].rolling(n, min_periods=max(2, n // 2)).max()
        roll_low = df["close"].rolling(n, min_periods=max(2, n // 2)).min()
        fig.add_trace(go.Scatter(
            x=df.index, y=roll_low, mode="lines",
            line=dict(width=0), connectgaps=True,
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=roll_high, mode="lines",
            line=dict(width=0), connectgaps=True,
            fill="tonexty", fillcolor=palette[i % len(palette)],
            name=f"{n}d H–L",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" + f"{n}d hi: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=4, color=ACCENT, line=dict(color=BG_BASE, width=0.5)),
        name="close",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>close: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12, color=ACCENT)),
        xaxis=dict(title=None),
        yaxis=dict(title="Price", tickformat=".4f"),
        height=height, hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=55, r=20, t=30, b=40),
    )
    return fig


def make_score_bar_chart(
    rows: list[dict],
    score_field: str = "score",
    title: str = "",
    height: int = 320,
    color: str = ACCENT,
) -> go.Figure:
    """Top-K composite-score horizontal bar chart."""
    fig = go.Figure()
    if not rows:
        fig.update_layout(height=height, annotations=[dict(
            text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=TEXT_DIM, size=12))])
        return fig
    syms = [r["symbol"] for r in rows]
    scores = [r.get(score_field) or 0.0 for r in rows]
    syms_disp = list(reversed(syms))
    scores_disp = list(reversed(scores))
    fig.add_trace(go.Bar(
        x=scores_disp, y=syms_disp, orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        text=[f"{s:.2f}" for s in scores_disp],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_BODY),
        hovertemplate="<b>%{y}</b><br>score: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12, color=ACCENT)),
        xaxis=dict(title="score (0..1)", showgrid=True, gridcolor=BORDER_SUBTLE,
                   tickfont=dict(family="JetBrains Mono, monospace", size=9),
                   range=[0, 1.05]),
        yaxis=dict(title=None, automargin=True, showgrid=False,
                   tickfont=dict(family="JetBrains Mono, monospace", size=10)),
        height=height,
        margin=dict(l=10, r=80, t=30, b=40),
        bargap=0.18,
    )
    return fig
