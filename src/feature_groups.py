"""
Single source of truth for feature definitions, subgroups, and model/feature type lists.

All submission scripts import from here to avoid duplication.
"""

FULL_FEATURE_STRING = (
    "endbartime|sumret|sumabsret|sumret3|sumret4|sumpret2|sumbipow|sumautocov|sumvolume|numobs|"
    "sumret2_ewstock|sumret3_ewstock|sumret4_ewstock|sumabsret_ewstock|sumbipow_ewstock|sumpret2_ewstock|"
    "turnover_ewstock|buyturnover_ewstock|sellturnover_ewstock|effspread_ewstock|spread_ewstock|"
    "sumret2_vwstock|sumret3_vwstock|sumret4_vwstock|sumabsret_vwstock|sumbipow_vwstock|sumpret2_vwstock|"
    "turnover_vwstock|buyturnover_vwstock|sellturnover_vwstock|effspread_vwstock|spread_vwstock|"
    "turnover_spy|buyturnover_spy|sellturnover_spy|"
    "stocktwits_attention|stocktwits_sentiment|stocktwits_sentcount|"
    "vix|vvix|vix3m|"
    "voldemand_spx_open_and_close|voldemand_spx_open_only|voldemand_all_open_and_close|voldemand_all_open_only|"
)

ALL_FEATURES = [f for f in FULL_FEATURE_STRING.split("|") if f]

ALL_MODELS = ["ridge", "xgboost", "lightgbm", "random_forest"]
FEATURE_TYPES = ["raw", "har", "pca", "ae"]

SUBGROUPS = {
    "baseline": [],
    "moments": [f for f in ALL_FEATURES if f.startswith("sum") and "stock" not in f and "volume" not in f],
    "liquidity": [f for f in ALL_FEATURES if any(x in f for x in ["volume", "turnover", "spread", "numobs"])],
    "market_ew": [f for f in ALL_FEATURES if ("ewstock" in f) and not any(x in f for x in ["turnover", "spread"])],
    "market_vw": [f for f in ALL_FEATURES if ("vwstock" in f) and not any(x in f for x in ["turnover", "spread"])],
    "sentiment": [f for f in ALL_FEATURES if "stocktwits" in f],
    "implied_vol": [f for f in ALL_FEATURES if "vix" in f],
    "vol_demand": [f for f in ALL_FEATURES if "voldemand" in f],
    "all_features": ALL_FEATURES,
}
