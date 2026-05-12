"""Global CSS injection — premium dark dashboard styling."""
from __future__ import annotations

import streamlit as st

from lib.theme import (
    ACCENT, ACCENT_BORDER, ACCENT_BRIGHT, ACCENT_DIM, ACCENT_GLOW,
    BG_BASE, BG_ELEVATED, BG_HOVER, BG_INPUT, BG_SIDEBAR, BG_SURFACE,
    BLUE, BORDER_DEFAULT, BORDER_STRONG, BORDER_SUBTLE,
    FS_BASE, FS_LG, FS_MD, FS_SM, FS_XS,
    GREEN, RADIUS_LG, RADIUS_MD, RADIUS_PILL, RADIUS_SM, RED,
    TEXT_BODY, TEXT_DIM, TEXT_DISABLED, TEXT_HEADING, TEXT_MUTED,
)

_CSS = f"""
<style>
@import url('https://rsms.me/inter/inter.css');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg-base: {BG_BASE};
    --bg-surface: {BG_SURFACE};
    --bg-elevated: {BG_ELEVATED};
    --bg-input: {BG_INPUT};
    --bg-hover: {BG_HOVER};
    --bg-sidebar: {BG_SIDEBAR};
    --border-subtle: {BORDER_SUBTLE};
    --border-default: {BORDER_DEFAULT};
    --border-strong: {BORDER_STRONG};
    --text-heading: {TEXT_HEADING};
    --text-body: {TEXT_BODY};
    --text-muted: {TEXT_MUTED};
    --text-dim: {TEXT_DIM};
    --text-disabled: {TEXT_DISABLED};
    --accent: {ACCENT};
    --accent-bright: {ACCENT_BRIGHT};
    --accent-dim: {ACCENT_DIM};
    --accent-glow: {ACCENT_GLOW};
    --green: {GREEN};
    --red: {RED};
    --amber: #fbbf24;
    --blue: {BLUE};
    --fs-xs: {FS_XS};
    --fs-sm: {FS_SM};
    --fs-base: {FS_BASE};
    --fs-md: {FS_MD};
    --fs-lg: {FS_LG};
    --r-sm: {RADIUS_SM};
    --r-md: {RADIUS_MD};
    --r-lg: {RADIUS_LG};
}}

/* ============ GLOBAL RESETS ============ */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-feature-settings: 'cv11', 'ss01', 'ss03';
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
body {{ background-color: var(--bg-base); }}

/* ============ FULL-WIDTH LAYOUT ============ */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    max-width: 100% !important;
}}
[data-testid="stMain"] > .block-container,
[data-testid="stAppViewContainer"] > .main > .block-container {{
    max-width: 100% !important;
}}

/* hide Streamlit branding noise */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stStatusWidget"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stHeader"] {{
    background-color: transparent;
    height: 0;
}}
[data-testid="stDecoration"] {{ display: none; }}

/* ============ TYPOGRAPHY ============ */
h1 {{
    font-family: 'Inter', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text-heading) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 0.25rem 0 !important;
    padding: 0 !important;
    border: none !important;
    line-height: 1.2 !important;
}}
h2 {{
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--text-heading) !important;
    letter-spacing: -0.01em !important;
    margin: 1.25rem 0 0.5rem 0 !important;
}}
h3 {{
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--text-heading) !important;
    margin: 1rem 0 0.5rem 0 !important;
}}
h4, h5 {{
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text-body) !important;
    margin: 0.75rem 0 0.25rem 0 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
p, .stMarkdown {{ color: var(--text-body); font-size: var(--fs-base); }}
[data-testid="stCaptionContainer"], .stCaption {{
    color: var(--text-muted) !important;
    font-size: var(--fs-sm) !important;
    line-height: 1.5 !important;
}}

/* monospace for numbers everywhere */
code, kbd, samp {{
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important;
    background-color: var(--bg-elevated) !important;
    color: var(--accent-bright) !important;
    padding: 1px 6px !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.85em !important;
    border: 1px solid var(--border-subtle);
}}

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {{
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-subtle);
}}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding-top: 1.5rem;
}}
[data-testid="stSidebarNav"] {{
    background-color: transparent !important;
    padding-top: 1rem;
}}
[data-testid="stSidebarNav"] ul {{
    padding: 0 0.5rem;
}}
[data-testid="stSidebarNav"] li {{
    margin: 2px 0;
}}
[data-testid="stSidebarNav"] li a {{
    border-radius: var(--r-md) !important;
    padding: 0.5rem 0.75rem !important;
    transition: all 0.12s ease !important;
    border: 1px solid transparent !important;
}}
[data-testid="stSidebarNav"] li a:hover {{
    background-color: var(--bg-hover) !important;
}}
[data-testid="stSidebarNav"] li a span {{
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    color: var(--text-body) !important;
}}
/* active page (Streamlit marks aria-current=page) */
[data-testid="stSidebarNav"] li a[aria-current="page"] {{
    background-color: var(--accent-glow) !important;
    border-color: var(--accent-border) !important;
}}
[data-testid="stSidebarNav"] li a[aria-current="page"] span {{
    color: var(--accent) !important;
}}
[data-testid="stSidebarNav"] hr {{
    border-color: var(--border-subtle);
    margin: 0.75rem 0.5rem !important;
}}

/* sidebar header (custom) */
.sidebar-brand {{
    padding: 0 1rem 1rem 1rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.5rem;
}}
.sidebar-brand .brand-name {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-heading);
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.sidebar-brand .brand-sub {{
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background-color: transparent;
    padding: 0;
    border-radius: 0;
    border: none;
    border-bottom: 1px solid var(--border-default);
    margin-bottom: 1rem;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: var(--text-muted) !important;
    padding: 0.6rem 1rem !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    font-size: var(--fs-md) !important;
    transition: all 0.15s ease !important;
    margin: 0 !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: var(--text-body) !important;
    background-color: var(--bg-elevated) !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: transparent !important;
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 0.5rem !important;
}}

/* ============ METRICS ============ */
[data-testid="stMetric"] {{
    background-color: var(--bg-surface);
    padding: 0.75rem 1rem;
    border-radius: var(--r-md);
    border: 1px solid var(--border-subtle);
    transition: border-color 0.15s ease;
}}
[data-testid="stMetric"]:hover {{
    border-color: var(--border-default);
}}
[data-testid="stMetricLabel"] {{
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: var(--fs-xs) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    color: var(--text-heading) !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important;
    letter-spacing: -0.01em;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: var(--fs-xs) !important;
}}

/* ============ DATAFRAMES ============ */
/* glide-data-grid (Streamlit's dataframe engine) reads colors from CSS vars
   below — set them so cell text/headers/icons are readable against our dark
   theme. Without these the default light-mode greys appear on dark backgrounds
   and become invisible. */
[data-testid="stDataFrame"] {{
    background-color: var(--bg-surface);
    border-radius: var(--r-md);
    border: 1px solid var(--border-subtle);
    overflow: hidden;
    --gdg-text-dark: {TEXT_HEADING};
    --gdg-text-medium: {TEXT_BODY};
    --gdg-text-light: {TEXT_MUTED};
    --gdg-bg-cell: {BG_SURFACE};
    --gdg-bg-cell-medium: {BG_SURFACE};
    --gdg-bg-header: {BG_ELEVATED};
    --gdg-bg-header-has-focus: {BG_ELEVATED};
    --gdg-bg-header-hovered: {BG_HOVER};
    --gdg-text-header: {TEXT_HEADING};
    --gdg-text-header-selected: {ACCENT};
    --gdg-bg-search: {BG_ELEVATED};
    --gdg-border-color: {BORDER_SUBTLE};
    --gdg-horizontal-border-color: {BORDER_SUBTLE};
    --gdg-bg-icon-header: {TEXT_MUTED};
    --gdg-fg-icon-header: {TEXT_MUTED};
    --gdg-accent-color: {ACCENT};
    --gdg-accent-fg: {BG_BASE};
    --gdg-accent-light: {ACCENT_GLOW};
    --gdg-cell-horizontal-padding: 8px;
    --gdg-cell-vertical-padding: 3px;
    --gdg-link-color: {ACCENT_BRIGHT};
    --gdg-drilldown-border: {ACCENT};
    --gdg-font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
}}
[data-testid="stDataFrame"] .dvn-scroller {{
    background-color: var(--bg-surface);
}}
/* Streamlit's "static" table (st.table) and any HTML table — force readable text */
table, .stTable table {{
    color: var(--text-body) !important;
    background-color: var(--bg-surface) !important;
    border-collapse: collapse !important;
}}
table thead th, .stTable thead th {{
    color: var(--text-heading) !important;
    background-color: var(--bg-elevated) !important;
    border-bottom: 1px solid var(--border-default) !important;
    padding: 6px 10px !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: var(--fs-sm) !important;
}}
table tbody td, .stTable tbody td {{
    color: var(--text-body) !important;
    background-color: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 5px 10px !important;
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important;
    font-size: var(--fs-sm) !important;
}}
table tbody tr:hover td {{
    background-color: var(--bg-hover) !important;
}}

/* ============ MULTISELECT / TAG PILLS (the "M60" blue-on-blue bug) ============ */
/* Streamlit multiselect renders each selected item as a tag. The default
   Streamlit dark theme leaves the tag text in accent color on accent
   background — unreadable. Override to make text bright on the accent bg. */
[data-baseweb="tag"] {{
    background-color: var(--accent-glow) !important;
    border: 1px solid var(--accent) !important;
    color: var(--text-heading) !important;
    font-weight: 600 !important;
    border-radius: var(--r-sm) !important;
}}
[data-baseweb="tag"] span,
[data-baseweb="tag"] [class*="Text"] {{
    color: var(--text-heading) !important;
    font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
    font-size: var(--fs-sm) !important;
}}
[data-baseweb="tag"] svg,
[data-baseweb="tag"] [role="button"] svg {{
    fill: var(--text-heading) !important;
    color: var(--text-heading) !important;
}}
[data-baseweb="tag"]:hover {{
    background-color: var(--accent) !important;
    color: var(--bg-base) !important;
}}
[data-baseweb="tag"]:hover span,
[data-baseweb="tag"]:hover [class*="Text"] {{
    color: var(--bg-base) !important;
}}
[data-baseweb="tag"]:hover svg {{
    fill: var(--bg-base) !important;
}}

/* Dropdown menus (multiselect / selectbox option lists) — readable */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {{
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {{
    color: var(--text-body) !important;
    background-color: transparent !important;
    padding: 0.4rem 0.75rem !important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover {{
    background-color: var(--bg-hover) !important;
    color: var(--text-heading) !important;
}}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"] {{
    background-color: var(--accent-glow) !important;
    color: var(--accent) !important;
}}

/* Select/multiselect input value text (shown when no items selected) */
[data-baseweb="select"] [data-baseweb="tag-container"] input,
[data-baseweb="select"] input {{
    color: var(--text-body) !important;
}}
[data-baseweb="select"] [class*="placeholder"],
[data-baseweb="select"] [data-baseweb="placeholder"] {{
    color: var(--text-dim) !important;
}}

/* ============ EXPANDER ============ */
[data-testid="stExpander"] {{
    background-color: var(--bg-surface);
    border-radius: var(--r-md);
    border: 1px solid var(--border-subtle);
    margin: 0.5rem 0;
}}
[data-testid="stExpander"] summary {{
    padding: 0.6rem 1rem !important;
    font-weight: 500 !important;
    color: var(--text-body) !important;
    font-size: var(--fs-md) !important;
}}
[data-testid="stExpander"] summary:hover {{
    color: var(--accent) !important;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    padding: 0.5rem 1rem 1rem 1rem !important;
}}

/* ============ BUTTONS ============ */
.stButton > button {{
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-body) !important;
    font-weight: 500 !important;
    font-size: var(--fs-sm) !important;
    border-radius: var(--r-md) !important;
    padding: 0.4rem 0.85rem !important;
    transition: all 0.12s ease !important;
    line-height: 1.2 !important;
}}
.stButton > button:hover {{
    background-color: var(--bg-elevated) !important;
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
}}
.stButton > button:focus:not(:active) {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}}
.stButton > button[kind="primary"] {{
    background-color: var(--accent) !important;
    color: var(--bg-base) !important;
    border-color: var(--accent) !important;
    font-weight: 600 !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: var(--accent-bright) !important;
    color: var(--bg-base) !important;
}}

/* ============ INPUTS ============ */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stTextArea > div > div,
.stNumberInput > div > div,
.stDateInput > div > div {{
    background-color: var(--bg-input) !important;
    border-color: var(--border-default) !important;
    border-radius: var(--r-md) !important;
}}
[data-baseweb="select"] > div:hover,
[data-baseweb="input"] > div:hover {{
    border-color: var(--border-strong) !important;
}}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}}
[data-baseweb="select"] [aria-selected="true"] {{
    background-color: var(--accent-glow) !important;
    color: var(--accent) !important;
}}
input, textarea, select {{
    color: var(--text-body) !important;
    font-size: var(--fs-md) !important;
    font-family: 'Inter', sans-serif !important;
}}

/* widget labels */
[data-testid="stWidgetLabel"] {{
    color: var(--text-muted) !important;
    font-size: var(--fs-xs) !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px !important;
}}

/* ============ TOGGLE ============ */
[data-baseweb="checkbox"] [role="switch"] {{
    background-color: var(--bg-elevated) !important;
}}
[data-baseweb="checkbox"] [role="switch"][aria-checked="true"] {{
    background-color: var(--accent) !important;
    box-shadow: 0 0 8px var(--accent-glow);
}}
[data-baseweb="checkbox"] {{
    font-size: var(--fs-md) !important;
}}

/* ============ SEGMENTED CONTROL ============ */
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="button"] {{
    background-color: var(--bg-surface) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border-subtle) !important;
    font-weight: 500 !important;
    font-size: var(--fs-sm) !important;
    transition: all 0.12s ease !important;
}}
[data-testid="stSegmentedControl"] button:hover {{
    background-color: var(--bg-elevated) !important;
    color: var(--text-body) !important;
}}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] [role="button"][aria-pressed="true"] {{
    background-color: var(--accent-glow) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}}

/* ============ PLOTLY ============ */
.js-plotly-plot {{
    border-radius: var(--r-lg);
    overflow: hidden;
    background-color: var(--bg-base);
}}
.js-plotly-plot .modebar {{
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-md) !important;
}}
.js-plotly-plot .modebar-btn path {{
    fill: var(--text-muted) !important;
}}
.js-plotly-plot .modebar-btn:hover path {{
    fill: var(--accent) !important;
}}

/* ============ ALERTS ============ */
[data-testid="stAlert"] {{
    border-radius: var(--r-md) !important;
    border: 1px solid var(--border-default) !important;
    font-size: var(--fs-md);
}}

/* ============ DIVIDER ============ */
hr {{
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 0.75rem 0 1rem 0 !important;
}}

/* ============ CUSTOM COMPONENTS (used by lib/components.py) ============ */
.page-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1rem;
}}
.page-title {{
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-heading);
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
}}
.page-subtitle {{
    font-size: var(--fs-md);
    color: var(--text-muted);
    margin-top: 0.25rem;
    line-height: 1.4;
}}
.page-meta {{
    text-align: right;
    font-size: var(--fs-xs);
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
}}

.status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    align-items: center;
    padding: 0.6rem 0.85rem;
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    font-size: var(--fs-sm);
    line-height: 1.5;
}}
.status-strip .item {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0 0.85rem;
    border-right: 1px solid var(--border-subtle);
}}
.status-strip .item:last-child {{ border-right: none; }}
.status-strip .item:first-child {{ padding-left: 0; }}
.status-strip .label {{
    color: var(--text-dim);
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}}
.status-strip .value {{
    color: var(--text-body);
    font-family: 'JetBrains Mono', monospace;
    font-size: var(--fs-sm);
    font-weight: 500;
}}
.status-strip .value.accent {{ color: var(--accent); }}
.status-strip .value.green  {{ color: var(--green); }}
.status-strip .value.red    {{ color: var(--red); }}

.live-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px {GREEN}99;
    animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.6; transform: scale(0.85); }}
}}

.kpi-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 2px 10px;
    background-color: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: {RADIUS_PILL};
    font-size: var(--fs-xs);
    color: var(--text-body);
    font-family: 'JetBrains Mono', monospace;
    margin-right: 4px;
}}
.kpi-chip .label {{ color: var(--text-dim); font-weight: 500; }}
.kpi-chip .val   {{ color: var(--text-body); font-weight: 600; }}
.kpi-chip.accent .val {{ color: var(--accent); }}
.kpi-chip.green  .val {{ color: var(--green); }}
.kpi-chip.red    .val {{ color: var(--red); }}

.section-card {{
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-lg);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s ease;
}}
.section-card:hover {{ border-color: var(--border-default); }}
.section-card .head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}}
.section-card .title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.005em;
    text-transform: uppercase;
}}
.section-card .meta {{
    font-size: var(--fs-xs);
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
}}

.summary-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0;
    padding: 0.5rem 0.85rem;
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    font-size: var(--fs-sm);
    margin-top: 0.5rem;
}}
.summary-row .stat {{
    padding: 0 1rem;
    border-right: 1px solid var(--border-subtle);
    line-height: 1.5;
}}
.summary-row .stat:last-child {{ border-right: none; }}
.summary-row .stat:first-child {{ padding-left: 0; }}
.summary-row .stat .lbl {{
    color: var(--text-dim);
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.4rem;
}}
.summary-row .stat .val {{
    color: var(--text-body);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}}
.summary-row .stat .val.accent {{ color: var(--accent); }}
.summary-row .stat .val.green  {{ color: var(--green); }}
.summary-row .stat .val.red    {{ color: var(--red); }}
.summary-row .stat .sub {{
    color: var(--text-dim);
    font-size: var(--fs-xs);
    font-family: 'JetBrains Mono', monospace;
    margin-left: 0.4rem;
}}

/* scrollbars */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--bg-base); }}
::-webkit-scrollbar-thumb {{
    background: var(--border-default);
    border-radius: var(--r-md);
    border: 2px solid var(--bg-base);
}}
::-webkit-scrollbar-thumb:hover {{ background: var(--border-strong); }}
</style>
"""


def inject_global_css() -> None:
    """Inject global CSS — call once per page at the top."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Compact branding header at top of sidebar."""
    import streamlit as st
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="brand-name">
                <span style="color: var(--accent);">◆</span>
                STIRS / Bonds
            </div>
            <div class="brand-sub">Multi-Economy Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
