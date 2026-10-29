"""Data loading, cleaning, splitting and encoding for the London Airbnb midterm project.

Every cleaning step records how many rows/cells it touched in a log, so the report's
"before and after" numbers come straight from the code. Steps are split into two kinds:

* deterministic, row-wise fixes (parsing, invalid values, duplicates), applied before the
  exploration/confirmation split because they do not learn anything from the data;
* data-dependent steps (outlier fences, rare-level merging, top amenities), which are fitted on
  the exploration split only and then applied unchanged to the confirmation split.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_DATE = "2026-09-17"
DATA_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/"
    f"{SNAPSHOT_DATE}/data/listings.csv.gz"
)
EXPECTED_SHA256 = "8017870734a9bffa1b424937372e5f02f8c4a114e5060534d7423be5fd718b4c"

SPLIT_SEED = 558          # course code; fixed before any exploration
EXPLORATION_SHARE = 0.30  # share of hosts whose listings go to the exploration split

REVIEW_SCORE_COLS = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value",
]
BOOL_COLS = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability"]
DATE_COLS = ["last_scraped", "calendar_last_scraped", "first_review", "last_review"]
# Columns that repeat `price` (checked: equal for 59,761 of 59,905 priced listings) or are
# Airbnb-internal metadata with no analytic content.
REDUNDANT_COLS = [
    "price_quote_checkin_date", "price_quote_checkout_date", "price_quote_total_price",
    "price_quote_price_per_night", "price_quote_raw", "scrape_id", "calendar_last_scraped",
]

# Marketing superlatives used to score how much a description "oversells". Fixed before any
# comparison with review scores; matched as whole words, case-insensitively.
HYPE_WORDS = [
    "stunning", "luxury", "luxurious", "amazing", "gorgeous", "spectacular", "breathtaking",
    "incredible", "exceptional", "exquisite", "fabulous", "magnificent", "immaculate",
    "perfect", "perfectly", "unique", "exclusive", "opulent", "lavish", "premium", "iconic",
    "unbeatable", "unrivalled", "unrivaled", "superb", "fantastic", "beautiful",
    "beautifully", "dream", "dreamy", "elegant", "sumptuous", "world-class", "five-star",
]
_HYPE_RE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in HYPE_WORDS) + r")(?!\w)", re.I)
_TAG_RE = re.compile(r"<[^>]+>")


# ----------------------------------------------------------------------------- download / load
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_raw(path: Path) -> Path:
    """Download the snapshot if it is missing, then verify its checksum."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    actual = sha256_of(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"{path} has SHA-256 {actual}, expected {EXPECTED_SHA256}. Inside Airbnb may have "
            "replaced the file; results will not match the report."
        )
    return path


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ----------------------------------------------------------------------------- parsers
def parse_price(series: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5. The '$' is an export artefact: the quotes say the currency is GBP."""
    cleaned = series.astype("string").str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def parse_bathrooms_text(text) -> tuple[float, bool, bool]:
    """'1.5 shared baths' -> (1.5, True, False); 'Half-bath' -> (0.5, False, False)."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or pd.isna(text):
        return (np.nan, False, False)
    t = str(text).strip().lower()
    shared, private = "shared" in t, "private" in t
    if t.startswith("no bathroom"):
        return (0.0, shared, private)
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", t)
    if match:
        return (float(match.group(1)), shared, private)
    if "half" in t:
        return (0.5, shared, private)
    return (np.nan, shared, private)


def parse_tf(series: pd.Series) -> pd.Series:
    return series.map({"t": True, "f": False}).astype("boolean")


def strip_html(text) -> str:
    if text is None or pd.isna(text):
        return ""
    return html.unescape(_TAG_RE.sub(" ", str(text))).strip()


def count_hype(text: str) -> int:
    return len(_HYPE_RE.findall(text or ""))


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


# ----------------------------------------------------------------------------- audit
def raw_audit(df: pd.DataFrame) -> dict:
    """Numbers about the raw file used in Sections 1 and 3 of the report."""
    price = parse_price(df["price"])
    miss = df.isna().mean()
    dup_host_key = df.duplicated(["host_id", "name", "latitude", "longitude"])
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "cells": int(df.shape[0] * df.shape[1]),
        "empty_columns": sorted(miss[miss == 1].index.tolist()),
        "n_empty_columns": int((miss == 1).sum()),
        "price_missing": int(price.isna().sum()),
        "price_missing_pct": float(price.isna().mean() * 100),
        "price_missing_from_previous_scrape": int((price.isna() & (df["source"] == "previous scrape")).sum()),
        "price_missing_zero_availability_pct": float((df.loc[price.isna(), "availability_365"] == 0).mean() * 100),
        "price_present_zero_availability_pct": float((df.loc[price.notna(), "availability_365"] == 0).mean() * 100),
        "price_min": float(price.min()),
        "price_median": float(price.median()),
        "price_max": float(price.max()),
        "rating_missing": int(df["review_scores_rating"].isna().sum()),
        "rating_missing_pct": float(df["review_scores_rating"].isna().mean() * 100),
        "rating_below_1": int((df["review_scores_rating"] < 1).sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "dup_same_host_name_coords": int(dup_host_key.sum()),
        "min_nights_over_365": int((df["minimum_nights"] > 365).sum()),
        "min_nights_max": float(df["minimum_nights"].max()),
        "description_with_html": int(df["description"].fillna("").str.contains("<", regex=False).sum()),
        "bathrooms_text_levels": int(df["bathrooms_text"].nunique()),
        "distinct_amenities": int(len({a for s in df["amenities"].dropna() for a in json.loads(s)})),
        "max_london_listings_per_host": int(df["calculated_host_listings_count"].max()),
        "max_airbnb_wide_listings_per_host": int(df["host_listings_count"].max()),
        "hosts": int(df["host_id"].nunique()),
    }


# ----------------------------------------------------------------------------- deterministic cleaning
def clean_deterministic(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Row-wise fixes that learn nothing from the data. Returns the clean frame and a step log."""
    df = raw.copy()
    log: list[dict] = []

    def record(step, action, touched, reason):
        log.append({"step": step, "action": action, "touched": int(touched), "reason": reason})

    # 1. Columns with no information at all.
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    record("15 columns are 100% empty", "dropped", len(empty),
           "Airbnb stopped exposing these fields (e.g. host_since, license, instant_bookable); "
           "nothing to impute from")

    # 2. Redundant / constant metadata.
    redundant = [c for c in REDUNDANT_COLS if c in df.columns]
    df = df.drop(columns=redundant)
    record("price_quote_* repeat price; scrape metadata constant", "dropped", len(redundant),
           "price_quote_price_per_night equals price for 99.8% of priced listings")

    # 3. Price is text with a misleading '$'.
    df["price_gbp"] = parse_price(df.pop("price"))
    record("price stored as text '$267.50'", "parsed to a number in GBP", df["price_gbp"].notna().sum(),
           "format fix; Airbnb's own quotes give the currency as GBP")

    # 4. Bathrooms only exist as free text.
    parsed = df["bathrooms_text"].map(parse_bathrooms_text)
    df["baths"] = parsed.map(lambda t: t[0]).astype("float64")
    df["bath_shared"] = parsed.map(lambda t: t[1]).astype("boolean")
    df["bath_private"] = parsed.map(lambda t: t[2]).astype("boolean")
    df = df.drop(columns=["bathrooms_text"])
    record("bathrooms only as text ('1.5 shared baths')", "parsed to count + shared/private flags",
           df["baths"].notna().sum(), "47 text levels carry two facts: how many and whether shared")

    # 5. Booleans, dates, source flag.
    for c in BOOL_COLS:
        df[c] = parse_tf(df[c])
    for c in [c for c in DATE_COLS if c in df.columns]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["from_previous_scrape"] = df.pop("source").eq("previous scrape")

    # 6. Invalid review scores (Airbnb stars run 1-5, so an average below 1 is impossible).
    bad = (df[REVIEW_SCORE_COLS] < 1).any(axis=1)
    df.loc[bad, REVIEW_SCORE_COLS] = np.nan
    df["score_invalid_flag"] = bad
    record("review score averages below 1", "set to missing and flagged", bad.sum(),
           "a 1-5 star average cannot be below 1; these are recording errors, not bad listings")

    # 7. Minimum-night placeholders.
    over = df["minimum_nights"] > 365
    df["min_nights_capped_flag"] = over
    df.loc[over, "minimum_nights"] = 365.0
    record("minimum_nights up to 1,125", "capped at 365 and flagged", over.sum(),
           "1,125 is Airbnb's form maximum, a placeholder for 'long lets only', not a real rule")

    # 8. Text features from the description.
    desc = df["description"].map(strip_html)
    record("HTML tags in descriptions", "stripped", raw["description"].fillna("").str.contains("<", regex=False).sum(),
           "tags such as <br /> are markup, not words")
    df["description"] = desc
    df["description_words"] = desc.map(count_words)
    df["hype_count"] = desc.map(count_hype)
    df["amenity_list"] = df.pop("amenities").map(json.loads)
    df["amenity_count"] = df["amenity_list"].map(len)

    # 9. Duplicate units: the same place listed twice.
    #    (a) same host, name and coordinates; (b) same name, description and coordinates under
    #    two host accounts. Keep the copy with the most reviews (the most reliable rating).
    df = df.sort_values(["number_of_reviews", "id"], ascending=[False, True])
    key_a = df.duplicated(["host_id", "name", "latitude", "longitude"])
    key_b = df.duplicated(["name", "description", "latitude", "longitude"])
    dup = key_a | key_b
    large_copies = int((dup & (df["calculated_host_listings_count"] >= 10)).sum())
    record("same unit listed twice", "dropped extra copies, kept the one with most reviews", dup.sum(),
           f"{int(key_a.sum())} same-host copies and {int((key_b & ~key_a).sum())} cross-account "
           f"copies; {large_copies} of all copies belong to hosts with 10+ listings, the group P1 "
           "studies, so leaving them would count those hosts twice")
    df = df.loc[~dup].sort_values("id").reset_index(drop=True)

    # Host portfolio size, recounted after duplicates are removed. Before removal this equals
    # Inside Airbnb's calculated_host_listings_count exactly (checked for all 51,348 hosts).
    df["host_london_listings"] = df.groupby("host_id")["id"].transform("size")
    n = df["host_london_listings"]
    df["host_class"] = np.select([n == 1, n <= 9], ["single (1)", "small (2-9)"], default="large (10+)")
    df["bedroom_band"] = pd.cut(df["bedrooms"], [-0.5, 1.5, 2.5, np.inf], labels=["0-1", "2", "3+"]).astype("string")
    df["price_missing_flag"] = df["price_gbp"].isna()
    df["rating_missing_flag"] = df["review_scores_rating"].isna()
    return df, log


# ----------------------------------------------------------------------------- split
def split_by_host(df: pd.DataFrame, seed: int = SPLIT_SEED, share: float = EXPLORATION_SHARE):
    """Assign whole hosts to exploration or confirmation, so no host straddles the split."""
    hosts = np.sort(df["host_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(hosts)
    n_explore = int(round(share * len(hosts)))
    explore_hosts = set(shuffled[:n_explore].tolist())
    is_explore = df["host_id"].isin(explore_hosts)
    return df.loc[is_explore].copy(), df.loc[~is_explore].copy()


# ----------------------------------------------------------------------------- data-dependent steps
def fit_encoding(explore: pd.DataFrame, top_amenities: int = 100, min_level_rows: int = 30) -> dict:
    """Learn outlier fences, kept levels and amenity vocabulary from the exploration split only."""
    log_price = np.log(explore["price_gbp"].dropna())
    q1, q3 = np.percentile(log_price, [25, 75])
    iqr = q3 - q1
    prop_counts = explore["property_type"].value_counts()
    amen_counts = pd.Series([a for lst in explore["amenity_list"] for a in lst]).value_counts()
    return {
        "log_price_q1": float(q1), "log_price_q3": float(q3),
        "price_low": float(np.exp(q1 - 1.5 * iqr)), "price_high": float(np.exp(q3 + 1.5 * iqr)),
        "property_levels": sorted(prop_counts[prop_counts >= min_level_rows].index.tolist()),
        "amenities": amen_counts.head(top_amenities).index.tolist(),
        "boroughs": sorted(explore["neighbourhood_cleansed"].unique().tolist()),
        "room_types": sorted(explore["room_type"].unique().tolist()),
    }


def apply_outlier_caps(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    low, high = params["price_low"], params["price_high"]
    below, above = out["price_gbp"] < low, out["price_gbp"] > high
    out["price_capped_flag"] = below | above
    out["price_gbp_capped"] = out["price_gbp"].clip(lower=low, upper=high)
    return out, {"below_fence": int(below.sum()), "above_fence": int(above.sum())}


NUMERIC_FEATURES = [
    "accommodates", "bedrooms", "beds", "baths", "price_gbp_capped", "minimum_nights",
    "maximum_nights", "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
    "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90", "availability_365", "availability_eoy",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "number_of_reviews_ly",
    "estimated_occupancy_l365d", "estimated_revenue_l365d", "reviews_per_month",
    *REVIEW_SCORE_COLS,
    "host_listings_count", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms", "hosts_time_as_user_years",
    "hosts_time_as_user_months", "hosts_time_as_host_years", "hosts_time_as_host_months",
    "latitude", "longitude", "description_words", "hype_count", "amenity_count",
    "days_since_first_review", "days_since_last_review",
]
FLAG_FEATURES = [
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability",
    "bath_shared", "bath_private", "from_previous_scrape", "score_invalid_flag",
    "min_nights_capped_flag", "price_missing_flag", "rating_missing_flag", "price_capped_flag",
]


def encode(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Model-ready numeric table: numbers, 0/1 flags, one-hot categories, multi-hot amenities.

    Structural missing values (no price because the listing is not bookable; no rating because
    it has no reviews) are left as NaN next to their flags: the confirmatory tests never use
    imputed values, and the modelling stage will impute inside its own training folds.
    """
    snap = pd.Timestamp(SNAPSHOT_DATE)
    work = df.copy()
    work["days_since_first_review"] = (snap - work["first_review"]).dt.days
    work["days_since_last_review"] = (snap - work["last_review"]).dt.days
    parts = [work[NUMERIC_FEATURES].astype("float64"),
             work[FLAG_FEATURES].astype("float64")]
    borough = pd.Categorical(work["neighbourhood_cleansed"], categories=params["boroughs"])
    parts.append(pd.get_dummies(borough, prefix="borough", dtype="float64").set_axis(work.index))
    room = pd.Categorical(work["room_type"], categories=params["room_types"])
    parts.append(pd.get_dummies(room, prefix="room", dtype="float64").set_axis(work.index))
    prop = work["property_type"].where(work["property_type"].isin(params["property_levels"]), "Other")
    prop = pd.Categorical(prop, categories=params["property_levels"] + ["Other"])
    parts.append(pd.get_dummies(prop, prefix="property", dtype="float64").set_axis(work.index))
    exploded = work["amenity_list"].explode()
    exploded = exploded[exploded.isin(set(params["amenities"]))]
    amen = (pd.crosstab(exploded.index, exploded)
            .reindex(index=work.index, columns=params["amenities"], fill_value=0)
            .clip(upper=1).astype("float64"))
    amen.columns = [f"amenity_{i:03d}" for i in range(len(params["amenities"]))]
    parts.append(amen)
    return pd.concat(parts, axis=1)
