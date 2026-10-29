"""Data dictionary: type, unit, range and missingness for every raw attribute (plus derived ones).

Types follow the course's three-way scheme (nominal / discrete / continuous); `kind` adds the
storage form (identifier, URL, free text, list, date, binary) so nothing is forced into a box.
Ranges and missingness are computed from the raw file, not typed by hand.
"""

from __future__ import annotations

import json

import pandas as pd

E = "empty in this snapshot"
# column: (type, kind, unit, meaning)
SPEC = {
    "id": ("nominal", "identifier", "", "Airbnb listing ID"),
    "listing_url": ("nominal", "URL", "", "listing page on airbnb.com"),
    "scrape_id": ("nominal", "identifier", "", "Inside Airbnb scrape run (constant)"),
    "last_scraped": ("discrete", "date", "day", "day the listing was scraped"),
    "source": ("nominal", "category", "", "'previous scrape' = carried over from an earlier scrape; blank = found by this city scrape"),
    "name": ("nominal", "free text", "", "listing title written by the host"),
    "description": ("nominal", "free text (HTML)", "", "listing description written by the host"),
    "neighborhood_overview": ("nominal", "free text", "", E),
    "picture_url": ("nominal", "URL", "", "cover photo"),
    "host_id": ("nominal", "identifier", "", "Airbnb host ID"),
    "host_url": ("nominal", "URL", "", "host page"),
    "host_profile_id": ("nominal", "identifier", "", "host profile ID"),
    "host_profile_url": ("nominal", "URL", "", "host profile page"),
    "host_name": ("nominal", "free text", "", "host first name (personal data, never used)"),
    "host_since": ("discrete", "date", "day", E),
    "hosts_time_as_user_years": ("discrete", "count", "years", "whole years since the host joined Airbnb"),
    "hosts_time_as_user_months": ("discrete", "count", "months", "months beyond the whole years (0-11)"),
    "hosts_time_as_host_years": ("discrete", "count", "years", "whole years the host has hosted"),
    "hosts_time_as_host_months": ("discrete", "count", "months", "months beyond the whole years (0-11)"),
    "host_location": ("nominal", "free text", "", "host's stated home location"),
    "host_about": ("nominal", "free text", "", "host biography"),
    "host_response_time": ("nominal", "category", "", E),
    "host_response_rate": ("continuous", "percentage", "%", E),
    "host_acceptance_rate": ("continuous", "percentage", "%", E),
    "host_is_superhost": ("nominal", "binary (t/f)", "", "host holds Airbnb's Superhost badge"),
    "host_thumbnail_url": ("nominal", "URL", "", E),
    "host_picture_url": ("nominal", "URL", "", "host photo"),
    "host_neighbourhood": ("nominal", "category", "", E),
    "host_listings_count": ("discrete", "count", "listings", "host's listings across all of Airbnb (all cities)"),
    "host_total_listings_count": ("discrete", "count", "listings", E),
    "host_verifications": ("nominal", "list", "", E),
    "host_has_profile_pic": ("nominal", "binary (t/f)", "", "host has a profile photo"),
    "host_identity_verified": ("nominal", "binary (t/f)", "", "host identity verified by Airbnb"),
    "neighbourhood": ("nominal", "category", "", E),
    "neighbourhood_cleansed": ("nominal", "category", "", "London borough, geocoded by Inside Airbnb"),
    "neighbourhood_group_cleansed": ("nominal", "category", "", E),
    "latitude": ("continuous", "coordinate", "degrees", "latitude (Airbnb blurs exact locations)"),
    "longitude": ("continuous", "coordinate", "degrees", "longitude"),
    "property_type": ("nominal", "category", "", "host-chosen property type"),
    "room_type": ("nominal", "category", "", "entire home/apt, private room, shared room or hotel room"),
    "accommodates": ("discrete", "count", "guests", "maximum number of guests"),
    "bathrooms": ("continuous", "count", "bathrooms", E),
    "bathrooms_text": ("nominal", "free text", "", "bathrooms as text, e.g. '1.5 shared baths' (parsed)"),
    "bedrooms": ("discrete", "count", "rooms", "number of bedrooms (0 = studio)"),
    "beds": ("discrete", "count", "beds", "number of beds"),
    "amenities": ("nominal", "list (JSON)", "", "amenities the host ticked"),
    "price": ("continuous", "money as text", "GBP per night", "nightly price; stored as '$267.50' although the currency is GBP"),
    "price_quote_checkin_date": ("discrete", "date", "day", "check-in date of the price quote Inside Airbnb requested"),
    "price_quote_checkout_date": ("discrete", "date", "day", "check-out date of that quote"),
    "price_quote_total_price": ("continuous", "money", "GBP", "total price of the quoted stay"),
    "price_quote_price_per_night": ("continuous", "money", "GBP per night", "quoted nightly price (repeats price)"),
    "price_quote_raw": ("nominal", "JSON text", "", "raw quote returned by Airbnb (currency GBP)"),
    "minimum_nights": ("discrete", "count", "nights", "current minimum stay"),
    "maximum_nights": ("discrete", "count", "nights", "current maximum stay"),
    "minimum_minimum_nights": ("discrete", "count", "nights", "smallest minimum stay in the next 365 days"),
    "maximum_minimum_nights": ("discrete", "count", "nights", "largest minimum stay in the next 365 days"),
    "minimum_maximum_nights": ("discrete", "count", "nights", "smallest maximum stay in the next 365 days"),
    "maximum_maximum_nights": ("discrete", "count", "nights", "largest maximum stay in the next 365 days"),
    "minimum_nights_avg_ntm": ("continuous", "average", "nights", "average minimum stay over the next 12 months"),
    "maximum_nights_avg_ntm": ("continuous", "average", "nights", "average maximum stay over the next 12 months"),
    "calendar_updated": ("nominal", "category", "", E),
    "has_availability": ("nominal", "binary (t/f)", "", "listing has any availability"),
    "availability_30": ("discrete", "count", "days", "days available in the next 30"),
    "availability_60": ("discrete", "count", "days", "days available in the next 60"),
    "availability_90": ("discrete", "count", "days", "days available in the next 90"),
    "availability_365": ("discrete", "count", "days", "days available in the next 365"),
    "calendar_last_scraped": ("discrete", "date", "day", "day the calendar was scraped"),
    "number_of_reviews": ("discrete", "count", "reviews", "total reviews"),
    "number_of_reviews_ltm": ("discrete", "count", "reviews", "reviews in the last 12 months"),
    "number_of_reviews_l30d": ("discrete", "count", "reviews", "reviews in the last 30 days"),
    "availability_eoy": ("discrete", "count", "days", "days available until the end of the year"),
    "number_of_reviews_ly": ("discrete", "count", "reviews", "reviews in the last year (Inside Airbnb definition)"),
    "estimated_occupancy_l365d": ("discrete", "count", "nights", "Inside Airbnb's estimate of nights booked in the last 365 days"),
    "estimated_revenue_l365d": ("continuous", "money", "GBP", "estimated occupancy times price"),
    "first_review": ("discrete", "date", "day", "date of the first review"),
    "last_review": ("discrete", "date", "day", "date of the latest review"),
    "review_scores_rating": ("continuous", "average", "stars (1-5)", "average overall guest rating (P1 response)"),
    "review_scores_accuracy": ("continuous", "average", "stars (1-5)", "average 'did the listing match its page?' score (P2 response)"),
    "review_scores_cleanliness": ("continuous", "average", "stars (1-5)", "average cleanliness score"),
    "review_scores_checkin": ("continuous", "average", "stars (1-5)", "average check-in score"),
    "review_scores_communication": ("continuous", "average", "stars (1-5)", "average communication score"),
    "review_scores_location": ("continuous", "average", "stars (1-5)", "average location score"),
    "review_scores_value": ("continuous", "average", "stars (1-5)", "average value-for-money score"),
    "license": ("nominal", "free text", "", E),
    "instant_bookable": ("nominal", "binary (t/f)", "", E),
    "calculated_host_listings_count": ("discrete", "count", "listings", "host's listings in this London snapshot"),
    "calculated_host_listings_count_entire_homes": ("discrete", "count", "listings", "of which entire homes"),
    "calculated_host_listings_count_private_rooms": ("discrete", "count", "listings", "of which private rooms"),
    "calculated_host_listings_count_shared_rooms": ("discrete", "count", "listings", "of which shared rooms"),
    "reviews_per_month": ("continuous", "rate", "reviews per month", "reviews divided by months since the first review"),
}

DERIVED = {
    "price_gbp": ("continuous", "money", "GBP per night", "price parsed to a number"),
    "price_gbp_capped": ("continuous", "money", "GBP per night", "price capped at the exploration-split log-price fences"),
    "baths": ("continuous", "count", "bathrooms", "bathroom count parsed from bathrooms_text (half = 0.5)"),
    "bath_shared": ("nominal", "binary", "", "bathroom is shared"),
    "bath_private": ("nominal", "binary", "", "bathroom is described as private"),
    "from_previous_scrape": ("nominal", "binary", "", "listing carried over from an earlier scrape"),
    "description_words": ("discrete", "count", "words", "words in the description after removing HTML"),
    "hype_count": ("discrete", "count", "words", "marketing superlatives in the description (fixed list)"),
    "amenity_count": ("discrete", "count", "amenities", "number of amenities listed"),
    "host_london_listings": ("discrete", "count", "listings", "host's London listings after duplicate removal"),
    "host_class": ("nominal", "category", "", "single (1), small (2-9) or large (10+) London portfolio"),
    "bedroom_band": ("nominal", "ordinal category", "", "0-1, 2 or 3+ bedrooms"),
}


def _range(series: pd.Series, kind: str) -> str:
    s = series.dropna()
    if s.empty:
        return "-"
    if kind.startswith("date"):
        d = pd.to_datetime(s, errors="coerce").dropna()
        return f"{d.min().date()} to {d.max().date()}"
    if kind in {"identifier", "URL"}:
        return f"{s.nunique():,} unique"
    if kind.startswith("list"):
        lengths = s.map(lambda v: len(json.loads(v)) if isinstance(v, str) else len(v))
        return f"{lengths.min()} to {lengths.max()} items per row"
    if pd.api.types.is_bool_dtype(s) or kind.startswith("binary"):
        return ", ".join(f"{k}: {v:,}" for k, v in s.value_counts().items())
    if pd.api.types.is_numeric_dtype(s):
        return f"{s.min():,.4g} to {s.max():,.4g}"
    if kind.startswith("free text") or kind.endswith("text"):
        lengths = s.astype(str).str.len()
        return f"{s.nunique():,} distinct; {lengths.min()} to {lengths.max()} characters"
    levels = s.nunique()
    if levels <= 6:
        return f"{levels} levels: " + ", ".join(str(v) for v in s.value_counts().index)
    return f"{levels:,} levels"


def build(raw: pd.DataFrame, clean: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for col in raw.columns:
        typ, kind, unit, meaning = SPEC[col]
        miss = raw[col].isna()
        rows.append({"attribute": col, "origin": "raw", "type": typ, "kind": kind, "unit": unit or "-",
                     "range": _range(raw[col], kind), "missing": int(miss.sum()),
                     "missing_pct": round(100 * miss.mean(), 1), "meaning": meaning})
    if clean is not None:
        for col, (typ, kind, unit, meaning) in DERIVED.items():
            if col not in clean.columns:
                continue
            miss = clean[col].isna()
            rows.append({"attribute": col, "origin": "derived", "type": typ, "kind": kind,
                         "unit": unit or "-", "range": _range(clean[col], kind),
                         "missing": int(miss.sum()), "missing_pct": round(100 * miss.mean(), 1),
                         "meaning": meaning})
    missing_spec = set(raw.columns) - set(SPEC)
    if missing_spec:
        raise KeyError(f"no dictionary entry for {sorted(missing_spec)}")
    return pd.DataFrame(rows)
