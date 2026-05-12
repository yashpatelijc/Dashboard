"""Australia Fundamentals — categorized BBG ticker inventory.

Covers indicators relevant to RBA policy and 90-day Bank Bill (YBA) pricing.
"""
from __future__ import annotations

from typing import TypedDict


class TickerEntry(TypedDict):
    ticker: str
    bucket: str
    category: str
    subcategory: str
    name: str
    analysis: str


def E(ticker, bucket, category, subcategory, name, analysis) -> TickerEntry:
    return TickerEntry(ticker=ticker, bucket=bucket, category=category,
                        subcategory=subcategory, name=name, analysis=analysis)


CATEGORIES = [
    "Policy Rates & Monetary Ops",
    "Cash Yield Curve",
    "Inflation",
    "Employment & Labor",
    "Growth & Output",
    "Consumer",
    "Housing",
    "Business & Factory",
    "Trade & External",
    "Commodities",
    "Energy",
    "FX",
    "Equity Indices",
]


AUSTRALIA_TICKERS: list[TickerEntry] = [
    # ---- POLICY RATES & MONETARY OPS ----
    E("RBACTRD Index", "rates_drivers", "Policy Rates & Monetary Ops", "RBA Policy Rate",
      "Reserve Bank of Australia Cash Rate Target",
      "Headline RBA policy rate. Anchor for YBA curve."),
    E("RBAERRA Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "RBA Cash Rate (Effective)",
      "Effective overnight cash rate — actual settlement reference."),
    E("BBSW3M Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "3M Bank Bill Swap Rate (BBSW)",
      "Reference rate for 90-day Bank Bill futures. Roughly = RBA cash + funding spread."),

    # ---- CASH YIELD CURVE ----
    E("GACGB2 Index", "rates_drivers", "Cash Yield Curve", "ACGB — Short",
      "Australia 2Y ACGB Yield",
      "Cleanest RBA-expectation benchmark."),
    E("GACGB5 Index", "rates_drivers", "Cash Yield Curve", "ACGB — Belly",
      "Australia 5Y ACGB Yield",
      "Belly anchor. PC2 slope cross-check."),
    E("GACGB10 Index", "rates_drivers", "Cash Yield Curve", "ACGB — Long",
      "Australia 10Y ACGB Yield",
      "Long-end benchmark. Trades closely with US 10Y."),

    # ---- INFLATION ----
    E("AUCPIYOY Index", "eco", "Inflation", "Headline CPI",
      "Australia CPI YoY (quarterly)",
      "RBA's primary target. Quarterly — huge market mover at release."),
    E("AUCPIQOQ Index", "eco", "Inflation", "Headline CPI",
      "Australia CPI QoQ",
      "Quarterly change — RBA's main quarterly inflation read."),
    E("AUCPITRM Index", "eco", "Inflation", "Core CPI",
      "Australia Trimmed Mean CPI YoY",
      "RBA's preferred core measure. Used in inflation target assessment."),
    E("AUCPMWMM Index", "eco", "Inflation", "Core CPI",
      "Australia Weighted Median CPI YoY",
      "Alt core measure. RBA cross-checks Trimmed Mean vs Weighted Median."),
    E("AUCPIMOM Index", "eco", "Inflation", "Monthly CPI",
      "Australia Monthly CPI Indicator YoY",
      "Higher-frequency CPI (introduced 2023). Front-runs quarterly print."),

    # ---- EMPLOYMENT & LABOR ----
    E("AULFRTE Index", "eco", "Employment & Labor", "Unemployment",
      "Australia Unemployment Rate",
      "RBA labour-market tightness gauge."),
    E("AUEMC Index", "eco", "Employment & Labor", "Employment Change",
      "Australia Employment Change (MoM)",
      "Monthly NFP-equivalent. Big intraday mover."),
    E("AULPCPYR Index", "eco", "Employment & Labor", "Wages",
      "Australia Wage Price Index YoY",
      "Quarterly wages data. Direct wage-spiral indicator for RBA."),
    E("AUPRTC Index", "eco", "Employment & Labor", "Participation",
      "Australia Labour Force Participation",
      "Slack indicator. High participation + low unemployment = RBA hawkish."),

    # ---- GROWTH & OUTPUT ----
    E("AUNAGDPC Index", "eco", "Growth & Output", "GDP",
      "Australia GDP YoY",
      "Quarterly. RBA growth signal."),
    E("AUNAGDPQ Index", "eco", "Growth & Output", "GDP",
      "Australia GDP QoQ",
      "QoQ growth — RBA's key real-time read."),
    E("ANZAIPMA Index", "eco", "Growth & Output", "PMI Manufacturing",
      "Australia Manufacturing PMI (AiG)",
      "Cycle indicator."),
    E("ANZAIPSA Index", "eco", "Growth & Output", "PMI Services",
      "Australia Services PMI (AiG)",
      "Service-sector cycle indicator."),

    # ---- CONSUMER ----
    E("WMCCCONS Index", "eco", "Consumer", "Consumer Confidence",
      "Westpac-MI Consumer Confidence",
      "Monthly sentiment. RBA watches the components."),
    E("AURSTOTL Index", "eco", "Consumer", "Retail Sales",
      "Australia Retail Sales MoM",
      "Spending coincident indicator."),

    # ---- HOUSING ----
    E("AUCBHPMC Index", "eco", "Housing", "House Prices",
      "Australia CoreLogic Home Value Index MoM",
      "Housing wealth channel — RBA tracks for financial stability."),
    E("AUBPVTC Index", "eco", "Housing", "Building Approvals",
      "Australia Building Approvals MoM",
      "Forward-looking construction indicator."),

    # ---- COMMODITIES ----
    E("IRNOR62 Index", "rates_drivers", "Commodities", "Iron Ore",
      "Iron Ore 62% Fe China Port (CFR)",
      "Australia's #1 export. Direct AUD and growth driver."),
    E("XAU Curncy", "rates_drivers", "Commodities", "Gold",
      "Gold Spot",
      "Australia is a top-5 gold producer — AUD-positive."),

    # ---- ENERGY ----
    E("WTI Index", "rates_drivers", "Energy", "Oil",
      "WTI Crude Oil Front-Month",
      "Energy inflation channel."),
    E("LNG Index", "rates_drivers", "Energy", "Natural Gas — LNG",
      "Asia LNG Spot",
      "Australia is a top-2 LNG exporter. Direct trade-balance impact."),

    # ---- TRADE & EXTERNAL ----
    E("AUTBTBNT Index", "eco", "Trade & External", "Trade Balance",
      "Australia Trade Balance",
      "Commodity-driven. Surplus = AUD up, RBA easier."),

    # ---- FX ----
    E("AUDUSD Curncy", "rates_drivers", "FX", "Major Cross",
      "AUD/USD Spot",
      "Commodity-cycle proxy. RBA watches trade-weighted AUD."),
    E("AUDNZD Curncy", "rates_drivers", "FX", "Major Cross",
      "AUD/NZD Spot",
      "Antipodean relative — RBA vs RBNZ divergence."),

    # ---- EQUITY INDICES ----
    E("AS51 Index", "rates_drivers", "Equity Indices", "Large Cap",
      "ASX 200 Index",
      "Australian risk barometer. Heavy resources weighting."),
]
