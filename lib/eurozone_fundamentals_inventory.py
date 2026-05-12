"""Eurozone Fundamentals — categorized BBG ticker inventory with analytical use cases.

Covers indicators relevant to ECB policy and Euribor / €STR / SARON pricing.
Same TickerEntry schema as `us_fundamentals_inventory.py` for UI compatibility.
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
    "Sovereign Spreads",
    "Energy",
    "Volatility",
    "Credit",
    "FX",
    "Equity Indices",
]


EUROZONE_TICKERS: list[TickerEntry] = [
    # ---- POLICY RATES & MONETARY OPS ----
    E("EUMRMRD Index", "rates_drivers", "Policy Rates & Monetary Ops", "ECB Policy Rate",
      "ECB Main Refinancing Operations Rate (MRO)",
      "Headline ECB policy rate. Anchor for Euribor / €STR curve."),
    E("EUDR1T Index", "rates_drivers", "Policy Rates & Monetary Ops", "ECB Policy Rate",
      "ECB Deposit Facility Rate (DFR)",
      "Effective floor — banks park reserves here. DFR is the actual short-rate anchor since 2014 floor system."),
    E("EUDR3M Index", "rates_drivers", "Policy Rates & Monetary Ops", "ECB Policy Rate",
      "ECB Marginal Lending Facility Rate",
      "Top of the ECB corridor. Used in corridor-width and stress assessment."),
    E("ESTRON Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "Euro Short-Term Rate (€STR)",
      "THE underlying for FER (€STR-based). Settles to compounded €STR over 3M reference period."),
    E("EUR003M Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market",
      "3-Month Euribor",
      "THE underlying for ER (Euribor 3M futures). Daily fixing."),
    E("SSARON Index", "rates_drivers", "Policy Rates & Monetary Ops", "Money Market — CHF",
      "Swiss Average Rate Overnight (SARON)",
      "THE underlying for FSR (SARON 3M futures). CHF market reference."),
    E("SZLTPRT Index", "rates_drivers", "Policy Rates & Monetary Ops", "SNB Policy",
      "SNB Policy Rate (3M LIBOR target → SARON guidance)",
      "Anchor for FSR. SNB targets SARON close to policy rate."),

    # ---- CASH YIELD CURVE (Bund / OIS) ----
    E("GTDEM2Y Govt", "rates_drivers", "Cash Yield Curve", "Bund — Short End",
      "Germany 2Y Bund Yield",
      "Cleanest benchmark for ECB expectations. Trades closely vs front Euribor."),
    E("GTDEM5Y Govt", "rates_drivers", "Cash Yield Curve", "Bund — Belly",
      "Germany 5Y Bund Yield",
      "Belly anchor — co-moves with 1y1y / 2y1y forwards."),
    E("GTDEM10Y Govt", "rates_drivers", "Cash Yield Curve", "Bund — Long",
      "Germany 10Y Bund Yield",
      "Long-end benchmark. PCA PC1 cross-check."),
    E("EUSWE3 Curncy", "rates_drivers", "Cash Yield Curve", "EUR OIS",
      "3Y EUR OIS",
      "Cleanest forward path of ECB rate (no credit, no term premium noise)."),
    E("EUSWE5 Curncy", "rates_drivers", "Cash Yield Curve", "EUR OIS",
      "5Y EUR OIS",
      "5y benchmark for term-premium decomposition."),

    # ---- INFLATION ----
    E("ECCPEMUY Index", "eco", "Inflation", "Headline HICP",
      "Eurozone HICP YoY (flash)",
      "ECB's primary inflation target. Each 0.1pp surprise → ~3-5 bp on front Euribor."),
    E("CPEXEMUY Index", "eco", "Inflation", "Core HICP",
      "Eurozone HICP ex-energy/food/alcohol YoY (core)",
      "ECB's preferred core measure. Lower noise, slower-moving — bigger market reaction."),
    E("CPEXEMUM Index", "eco", "Inflation", "Core HICP",
      "Eurozone HICP ex-energy/food/alcohol MoM",
      "High-frequency core inflation indicator."),
    E("ECCPGEY Index", "eco", "Inflation", "Country — Germany",
      "Germany CPI YoY (HICP)",
      "Leading indicator for Eurozone aggregate — comes out 1 day earlier."),
    E("FRCPIYOY Index", "eco", "Inflation", "Country — France",
      "France CPI YoY (HICP)",
      "Second-largest member. Italy/Spain add up after."),

    # ---- EMPLOYMENT & LABOR ----
    E("UMRTEMU Index", "eco", "Employment & Labor", "Unemployment",
      "Eurozone Unemployment Rate",
      "Trend gauge for wage pressure. Lags inflation by ~6m."),
    E("EUWGRYY Index", "eco", "Employment & Labor", "Wages",
      "Eurozone Negotiated Wages YoY",
      "Forward-looking inflation input. ECB watches this carefully — direct trade thesis driver."),

    # ---- GROWTH & OUTPUT ----
    E("EUGNEMUY Index", "eco", "Growth & Output", "GDP",
      "Eurozone GDP YoY",
      "Headline growth. Quarterly. Marginal market impact in normal times."),
    E("EUITEMUY Index", "eco", "Growth & Output", "Industrial Production",
      "Eurozone Industrial Production YoY",
      "Cycle indicator. Predictive for the next quarter's GDP."),
    E("MPMIEZMA Index", "eco", "Growth & Output", "PMI Manufacturing",
      "Eurozone Manufacturing PMI (S&P)",
      "Flash → final. Big intraday mover."),
    E("MPMIEZSA Index", "eco", "Growth & Output", "PMI Services",
      "Eurozone Services PMI (S&P)",
      "Bigger weight to GDP than manufacturing — watch services."),
    E("MPMIEZCA Index", "eco", "Growth & Output", "PMI Composite",
      "Eurozone Composite PMI (S&P)",
      "Best single read on the cycle."),

    # ---- CONSUMER ----
    E("EUCCEMU Index", "eco", "Consumer", "Consumer Confidence",
      "Eurozone Consumer Confidence (DG-ECFIN)",
      "Soft data — pre-empts retail sales by 1m."),
    E("RSSAEZY Index", "eco", "Consumer", "Retail Sales",
      "Eurozone Retail Sales YoY",
      "Coincident with consumer side of GDP."),

    # ---- BUSINESS & FACTORY ----
    E("GRZEWI Index", "eco", "Business & Factory", "Business Surveys",
      "Germany ZEW Economic Sentiment",
      "First survey to print each month. Trend bellwether."),
    E("GRIFPBUS Index", "eco", "Business & Factory", "Business Surveys",
      "Germany Ifo Business Climate",
      "Detailed business survey. Two sub-indices: current + expectations."),

    # ---- SOVEREIGN SPREADS ----
    E("GTBPGB2Y Govt", "rates_drivers", "Sovereign Spreads", "BTP-Bund 2Y",
      "Italy 2Y BTP Yield",
      "Italy-Bund spread = fragmentation gauge. ECB TPI threshold consideration."),
    E("GTBPGB10Y Govt", "rates_drivers", "Sovereign Spreads", "BTP-Bund 10Y",
      "Italy 10Y BTP Yield",
      "Wider BTP-Bund => ECB potentially tighter for less peripheral support."),

    # ---- ENERGY ----
    E("TTFGNXT Index", "rates_drivers", "Energy", "Natural Gas — TTF",
      "Dutch TTF Front-Month Natural Gas",
      "Critical for Eurozone inflation (gas → utility CPI). Geopolitical shock channel."),

    # ---- FX ----
    E("EURUSD Curncy", "rates_drivers", "FX", "Major Cross",
      "EUR/USD Spot",
      "Trade-weighted EUR move shifts imported inflation. ECB watches the TWI."),
    E("EURGBP Curncy", "rates_drivers", "FX", "Major Cross",
      "EUR/GBP Spot",
      "Cross-CB policy divergence proxy."),
    E("EURCHF Curncy", "rates_drivers", "FX", "Major Cross",
      "EUR/CHF Spot",
      "SARON / SNB context. SNB FX-intervention gauge."),

    # ---- EQUITY INDICES ----
    E("SX5E Index", "rates_drivers", "Equity Indices", "Eurozone Blue Chip",
      "Euro Stoxx 50",
      "Risk-off / risk-on proxy. Coincident with credit spread moves."),
    E("DAX Index", "rates_drivers", "Equity Indices", "Germany",
      "DAX 40 Index",
      "Largest single-country equity reference."),

    # ---- VOLATILITY ----
    E("V2X Index", "rates_drivers", "Volatility", "Implied Vol — Equity",
      "VStoxx (Euro Stoxx 50 implied vol)",
      "EUR-area risk barometer. Spikes coincident with rate-vol spikes."),
]
