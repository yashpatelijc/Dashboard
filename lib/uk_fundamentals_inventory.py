"""UK Fundamentals — categorized BBG ticker inventory.

Covers indicators relevant to BoE policy and SONIA pricing.
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
    "Fiscal / Gilt Operations",
    "Energy",
    "Volatility",
    "FX",
    "Equity Indices",
]


UK_TICKERS: list[TickerEntry] = [
    # ---- POLICY RATES & MONETARY OPS ----
    E("UKBRBASE Index", "rates_drivers", "Policy Rates & Monetary Ops", "BoE Policy Rate",
      "Bank of England Bank Rate",
      "Headline BoE policy rate. Anchor for SONIA / Gilt curve."),
    E("SONIO Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "Sterling Overnight Index Average (SONIA)",
      "THE underlying for SON (SONIA 3M futures). Daily, compounded over 3M reference."),
    E("SONIA Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "SONIA Compounded Index",
      "Index used for 3M compounding calculation in SON futures settlement."),

    # ---- CASH YIELD CURVE (Gilt / OIS) ----
    E("GTGBP2Y Govt", "rates_drivers", "Cash Yield Curve", "Gilt — Short",
      "UK 2Y Gilt Yield",
      "Cleanest BoE-expectation gauge."),
    E("GTGBP5Y Govt", "rates_drivers", "Cash Yield Curve", "Gilt — Belly",
      "UK 5Y Gilt Yield",
      "Belly anchor. PC2 slope cross-check."),
    E("GTGBP10Y Govt", "rates_drivers", "Cash Yield Curve", "Gilt — Long",
      "UK 10Y Gilt Yield",
      "Long-end. LDI and pension demand make it noisy at times."),
    E("BPSWS3 Curncy", "rates_drivers", "Cash Yield Curve", "GBP OIS",
      "3Y GBP OIS",
      "Pure BoE path. No credit, minimal term premium."),
    E("BPSWS5 Curncy", "rates_drivers", "Cash Yield Curve", "GBP OIS",
      "5Y GBP OIS",
      "5y benchmark for term-premium decomposition."),

    # ---- INFLATION ----
    E("UKRPCJYR Index", "eco", "Inflation", "Headline CPI",
      "UK CPI YoY",
      "BoE primary target. Each 0.1pp surprise → ~3-5 bp on front SONIA."),
    E("CPIYCH Index", "eco", "Inflation", "Core CPI",
      "UK Core CPI YoY (ex-energy/food)",
      "Stickier measure — BoE focuses heavily on this for cuts/hikes timing."),
    E("UKRPCJMM Index", "eco", "Inflation", "MoM CPI",
      "UK CPI MoM",
      "High-frequency. Watch for sustained 0.2%+ prints."),
    E("UKRP Index", "eco", "Inflation", "RPI",
      "UK Retail Price Index YoY",
      "Pension/index-linked benchmark. Legacy measure but still affects breakevens."),

    # ---- EMPLOYMENT & LABOR ----
    E("UKUEILOR Index", "eco", "Employment & Labor", "Unemployment",
      "UK ILO Unemployment Rate",
      "Trend gauge. BoE labour market tightness barometer."),
    E("UKAVE3MN Index", "eco", "Employment & Labor", "Wages",
      "UK Average Weekly Earnings 3M YoY",
      "BoE's key wage indicator. Direct hawk/dove driver."),
    E("UKAWEX3R Index", "eco", "Employment & Labor", "Wages",
      "UK Average Weekly Earnings ex-bonuses 3M YoY",
      "Cleaner core wage. Less noise from one-off bonuses."),

    # ---- GROWTH & OUTPUT ----
    E("UKGRYBYY Index", "eco", "Growth & Output", "GDP",
      "UK GDP YoY",
      "Quarterly headline."),
    E("UKGRYBQQ Index", "eco", "Growth & Output", "GDP",
      "UK GDP QoQ",
      "QoQ change — high-frequency response."),
    E("UKMOIPMC Index", "eco", "Growth & Output", "PMI Manufacturing",
      "UK Manufacturing PMI (S&P)",
      "Cycle gauge."),
    E("UKMOIPSC Index", "eco", "Growth & Output", "PMI Services",
      "UK Services PMI (S&P)",
      "Largest sector. Most-watched single number."),
    E("UKMOIPCC Index", "eco", "Growth & Output", "PMI Composite",
      "UK Composite PMI (S&P)",
      "Best single cycle read."),

    # ---- CONSUMER ----
    E("UKCCI Index", "eco", "Consumer", "Consumer Confidence",
      "UK GfK Consumer Confidence",
      "Coincident with retail spending."),
    E("UKRSAYOY Index", "eco", "Consumer", "Retail Sales",
      "UK Retail Sales YoY (volume)",
      "Excluding fuel — cleaner read."),

    # ---- HOUSING ----
    E("UKHPHPYC Index", "eco", "Housing", "House Prices",
      "UK Halifax House Price Index YoY",
      "Real-estate stress channel. BoE hiking → mortgage rate transmission."),
    E("UKNHPYY Index", "eco", "Housing", "House Prices",
      "UK Nationwide House Price Index YoY",
      "Alternative HPI. Cross-check with Halifax."),
    E("UKAMM Index", "eco", "Housing", "Mortgage Approvals",
      "UK Mortgage Approvals",
      "Forward indicator for housing turnover."),

    # ---- ENERGY ----
    E("NBPNXT Index", "rates_drivers", "Energy", "Natural Gas — NBP",
      "UK NBP Natural Gas Front-Month",
      "Inflation transmission via utility bills."),
    E("BRENT Index", "rates_drivers", "Energy", "Oil",
      "Brent Crude Oil Front-Month",
      "UK uses Brent as oil reference for inflation."),

    # ---- FX ----
    E("GBPUSD Curncy", "rates_drivers", "FX", "Major Cross",
      "GBP/USD Spot",
      "Trade-weighted cable. Imported inflation channel."),
    E("EURGBP Curncy", "rates_drivers", "FX", "Major Cross",
      "EUR/GBP Spot",
      "BoE-ECB policy divergence proxy."),

    # ---- EQUITY INDICES ----
    E("UKX Index", "rates_drivers", "Equity Indices", "Large Cap",
      "FTSE 100 Index",
      "UK risk-on/off proxy. Heavy commodity weighting."),
    E("MCX Index", "rates_drivers", "Equity Indices", "Mid Cap",
      "FTSE 250 Index",
      "Domestic-economy gauge (more UK-focused than FTSE 100)."),
]
