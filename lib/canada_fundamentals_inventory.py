"""Canada Fundamentals — categorized BBG ticker inventory.

Covers indicators relevant to BoC policy and CORRA pricing.
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
    "Energy",
    "FX",
    "Equity Indices",
]


CANADA_TICKERS: list[TickerEntry] = [
    # ---- POLICY RATES & MONETARY OPS ----
    E("CABROVER Index", "rates_drivers", "Policy Rates & Monetary Ops", "BoC Policy Rate",
      "Bank of Canada Overnight Rate Target",
      "Headline BoC policy rate. Anchor for CORRA / GoC curve."),
    E("CAONREPO Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "Canadian Overnight Repo Rate Average (CORRA)",
      "THE underlying for CRA (CORRA 3M futures). Daily, compounded over 3M reference."),
    E("CAGSWP3 Curncy", "rates_drivers", "Policy Rates & Monetary Ops", "OIS",
      "3Y CAD OIS",
      "Pure BoC path. Cleanest forward."),

    # ---- CASH YIELD CURVE ----
    E("GCAN2YR Index", "rates_drivers", "Cash Yield Curve", "GoC — Short",
      "Canada 2Y GoC Yield",
      "Cleanest BoC-expectation gauge."),
    E("GCAN5YR Index", "rates_drivers", "Cash Yield Curve", "GoC — Belly",
      "Canada 5Y GoC Yield",
      "Belly anchor. PC2 slope cross-check."),
    E("GCAN10YR Index", "rates_drivers", "Cash Yield Curve", "GoC — Long",
      "Canada 10Y GoC Yield",
      "Long-end benchmark. Trades closely with US 10Y."),

    # ---- INFLATION ----
    E("CACPIYOY Index", "eco", "Inflation", "Headline CPI",
      "Canada CPI YoY",
      "BoC primary target. Each 0.1pp surprise → ~3-5 bp on CORRA front."),
    E("CACPIMOM Index", "eco", "Inflation", "Headline CPI",
      "Canada CPI MoM",
      "High-frequency. BoC focuses on monthly momentum."),
    E("CACPIMCO Index", "eco", "Inflation", "Core CPI",
      "Canada Core CPI YoY (CPI-Median)",
      "BoC's preferred core measure (median trimmed)."),
    E("CACPIMTR Index", "eco", "Inflation", "Core CPI",
      "Canada Core CPI YoY (CPI-Trim)",
      "Alt core (trimmed mean). BoC uses 3-measure dashboard."),
    E("CACPIMCM Index", "eco", "Inflation", "Core CPI",
      "Canada Core CPI YoY (CPI-Common)",
      "Third core measure. BoC's average-of-three approach."),

    # ---- EMPLOYMENT & LABOR ----
    E("CANLNETJ Index", "eco", "Employment & Labor", "Employment Change",
      "Canada Net Employment Change",
      "Monthly NFP-equivalent. Big intraday mover for CAD/CORRA."),
    E("CANLOOR Index", "eco", "Employment & Labor", "Unemployment",
      "Canada Unemployment Rate",
      "Trend gauge. BoC labour-market tightness barometer."),
    E("CALFEMHE Index", "eco", "Employment & Labor", "Wages",
      "Canada Average Hourly Wages YoY (permanent employees)",
      "BoC's wage-growth indicator. Direct hawk/dove driver."),

    # ---- GROWTH & OUTPUT ----
    E("CGE9YOY Index", "eco", "Growth & Output", "GDP",
      "Canada GDP YoY",
      "Monthly GDP — leading the quarterly print."),
    E("CGE9MOM Index", "eco", "Growth & Output", "GDP",
      "Canada GDP MoM",
      "High-frequency. BoC's key real-time read."),
    E("CPMICAMA Index", "eco", "Growth & Output", "PMI Manufacturing",
      "Canada Manufacturing PMI (S&P)",
      "Cycle indicator."),
    E("CIVPIINS Index", "eco", "Growth & Output", "Industrial Production",
      "Canada Industrial Product Price Index MoM",
      "Producer prices — pass-through to CPI."),

    # ---- CONSUMER ----
    E("CACONFRP Index", "eco", "Consumer", "Consumer Confidence",
      "Canada Consumer Confidence (Conference Board)",
      "Sentiment indicator."),
    E("CARSCONS Index", "eco", "Consumer", "Retail Sales",
      "Canada Retail Sales MoM",
      "Coincident with consumer spending."),

    # ---- HOUSING ----
    E("CAHPNETT Index", "eco", "Housing", "House Prices",
      "Canada Teranet/National Bank HPI YoY",
      "Real-estate stress — BoC monitors for financial stability."),
    E("CAHSTLP Index", "eco", "Housing", "Housing Starts",
      "Canada Housing Starts",
      "Forward construction indicator."),

    # ---- ENERGY ----
    E("CL1 Comdty", "rates_drivers", "Energy", "Oil — WTI",
      "WTI Crude Oil Front-Month",
      "Canada is a top oil exporter. Direct CAD and BoC inflation channel."),
    E("WCS Index", "rates_drivers", "Energy", "Oil — WCS",
      "Western Canadian Select",
      "Heavy oil reference. WCS-WTI discount = pipeline-capacity stress."),

    # ---- TRADE & EXTERNAL ----
    E("CATBTOTB Index", "eco", "Trade & External", "Trade Balance",
      "Canada Trade Balance",
      "Commodity-driven. Surplus = CAD up."),

    # ---- FX ----
    E("USDCAD Curncy", "rates_drivers", "FX", "Major Cross",
      "USD/CAD Spot",
      "Commodity-cycle proxy. BoC watches trade-weighted CAD."),
    E("CADJPY Curncy", "rates_drivers", "FX", "Major Cross",
      "CAD/JPY Spot",
      "Risk-on/off proxy. CAD-funded carry trade."),

    # ---- EQUITY INDICES ----
    E("SPTSX Index", "rates_drivers", "Equity Indices", "Large Cap",
      "S&P/TSX Composite Index",
      "Canadian risk barometer. Heavy energy + financials weighting."),
]
