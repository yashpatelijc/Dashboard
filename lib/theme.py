"""Theme tokens — premium dark financial dashboard palette.

Palette philosophy:
- Deep blue-tinted black background (not harsh pure-black)
- Warm amber accent (more saturated, more readable than gold)
- Soft semantic colors (mint, coral, sky) — never pure RGB
- High data-ink ratio: borders subtle, text high-contrast
"""

# -----------------------------------------------------------------------------
# Backgrounds (depth layers)
# -----------------------------------------------------------------------------
BG_BASE = "#0a0e14"          # page background
BG_SURFACE = "#11161f"       # cards, panels
BG_ELEVATED = "#1a212e"      # raised surfaces (hover)
BG_INPUT = "#0d121b"         # input fields
BG_HOVER = "#1f2733"         # interactive hover state
BG_SIDEBAR = "#0d1218"       # slightly darker than base for clear nav separation

# Aliases for backward compat
BG_PRIMARY = BG_BASE
BG_SECONDARY = BG_SIDEBAR
BG_CARD = BG_SURFACE

# -----------------------------------------------------------------------------
# Borders
# -----------------------------------------------------------------------------
BORDER_SUBTLE = "#1c2330"
BORDER_DEFAULT = "#252e3d"
BORDER_STRONG = "#3a4555"
BORDER = BORDER_DEFAULT  # alias

# -----------------------------------------------------------------------------
# Text
# -----------------------------------------------------------------------------
TEXT_HEADING = "#f0f4fa"     # page titles
TEXT_BODY = "#d4dae3"        # primary body text
TEXT_MUTED = "#8a96a8"       # captions / labels
TEXT_DIM = "#5e6975"         # tertiary
TEXT_DISABLED = "#3d454f"

# Aliases
TEXT_PRIMARY = TEXT_BODY
TEXT_SECONDARY = TEXT_MUTED

# -----------------------------------------------------------------------------
# Brand accent (warm amber)
# -----------------------------------------------------------------------------
ACCENT = "#e8b75d"
ACCENT_DIM = "#a8843e"
ACCENT_BRIGHT = "#f5c878"
ACCENT_GLOW = "rgba(232, 183, 93, 0.15)"
ACCENT_BORDER = "rgba(232, 183, 93, 0.35)"

# Aliases
GOLD = ACCENT
GOLD_DIM = ACCENT_DIM

# -----------------------------------------------------------------------------
# Semantic colors (modern soft palette)
# -----------------------------------------------------------------------------
GREEN = "#4ade80"            # positive / live
GREEN_DIM = "#2d8c4d"
RED = "#f87171"              # negative / alert
RED_DIM = "#9c4646"
AMBER = "#fbbf24"            # warning
AMBER_DIM = "#a87f17"
BLUE = "#60a5fa"             # info / compare
BLUE_DIM = "#3a6ba1"
PURPLE = "#a78bfa"           # highlight / special
PURPLE_DIM = "#6f5ba8"
CYAN = "#22d3ee"             # accent secondary
CYAN_DIM = "#1689a0"

# -----------------------------------------------------------------------------
# Type scale (rem)
# -----------------------------------------------------------------------------
FS_XS = "0.6875rem"          # 11px
FS_SM = "0.75rem"            # 12px
FS_BASE = "0.8125rem"        # 13px (default — dense)
FS_MD = "0.875rem"           # 14px
FS_LG = "1rem"               # 16px
FS_XL = "1.125rem"           # 18px
FS_2XL = "1.375rem"          # 22px
FS_3XL = "1.75rem"           # 28px

# -----------------------------------------------------------------------------
# Effects
# -----------------------------------------------------------------------------
SHADOW_SUBTLE = "0 1px 3px rgba(0, 0, 0, 0.3)"
SHADOW_RAISED = "0 4px 12px rgba(0, 0, 0, 0.4)"
SHADOW_GLOW = f"0 0 0 1px {ACCENT_BORDER}, 0 4px 16px rgba(232, 183, 93, 0.12)"

# Border-radius
RADIUS_SM = "4px"
RADIUS_MD = "6px"
RADIUS_LG = "10px"
RADIUS_PILL = "999px"
