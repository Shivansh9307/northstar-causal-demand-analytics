#!/usr/bin/env python3
"""
PromoPulse: Causal Promotion, Demand and Inventory Optimisation
Synthetic dataset generator for Northstar Retail Group.

Dependencies:
    pip install pandas numpy Faker

Run:
    python generate_retail_dataset.py

Outputs:
    data/raw/
"""

from __future__ import annotations

import calendar
import logging
import math
import shutil
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from faker import Faker


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_config() -> Dict[str, Any]:
    """Read config/config.yaml. Values there are the single source of truth."""
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = _load_config()

SEED = int(CONFIG["seed"])
FULL_MODE = bool(CONFIG["full_mode"])

_scale = CONFIG["scale"]["full" if FULL_MODE else "dev"]
N_STORES = int(_scale["n_stores"])
N_SKUS = int(_scale["n_skus"])
START_DATE = str(_scale["start_date"])
END_DATE = str(_scale["end_date"])

OUTPUT_DIR = PROJECT_ROOT / CONFIG["paths"]["raw"]
GROUND_TRUTH_DIR = PROJECT_ROOT / CONFIG["paths"]["ground_truth"]
REPORTS_DIR = PROJECT_ROOT / CONFIG["paths"]["reports"]

_gen = CONFIG["generation"]
CHUNK_DAYS = int(_gen["chunk_days"])
PROMO_DENSITY_TARGET = float(_gen["promo_density_target"])
CAMPAIGN_SHARE = float(_gen["campaign_share"])
COHORT_OFFSETS_DAYS = [int(x) for x in _gen["cohort_offsets_days"]]
N_COHORTS = int(_gen["n_cohorts"])

RNG = np.random.default_rng(SEED)
FAKER = Faker("en_GB")
Faker.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("promopulse")


# =============================================================================
# REFERENCE DATA
# =============================================================================

REGIONAL_LOCATIONS = [
    ("London", "London", "England", 51.5074, -0.1278),
    ("Birmingham", "West Midlands", "England", 52.4862, -1.8904),
    ("Manchester", "North West", "England", 53.4808, -2.2426),
    ("Liverpool", "North West", "England", 53.4084, -2.9916),
    ("Leeds", "Yorkshire and the Humber", "England", 53.8008, -1.5491),
    ("Sheffield", "Yorkshire and the Humber", "England", 53.3811, -1.4701),
    ("Newcastle", "North East", "England", 54.9783, -1.6178),
    ("Bristol", "South West", "England", 51.4545, -2.5879),
    ("Nottingham", "East Midlands", "England", 52.9548, -1.1581),
    ("Leicester", "East Midlands", "England", 52.6369, -1.1398),
    ("Northampton", "East Midlands", "England", 52.2405, -0.9027),
    ("Reading", "South East", "England", 51.4543, -0.9781),
    ("Southampton", "South East", "England", 50.9097, -1.4044),
    ("Cardiff", "South Wales", "Wales", 51.4816, -3.1791),
    ("Swansea", "South Wales", "Wales", 51.6214, -3.9436),
    ("Edinburgh", "Scotland", "Scotland", 55.9533, -3.1883),
    ("Glasgow", "Scotland", "Scotland", 55.8642, -4.2518),
    ("Aberdeen", "Scotland", "Scotland", 57.1497, -2.0943),
    ("Dundee", "Scotland", "Scotland", 56.4620, -2.9707),
    ("Cambridge", "East of England", "England", 52.2053, 0.1218),
    ("Oxford", "South East", "England", 51.7520, -1.2577),
    ("Portsmouth", "South East", "England", 50.8198, -1.0880),
    ("Exeter", "South West", "England", 50.7184, -3.5339),
    ("Norwich", "East of England", "England", 52.6309, 1.2974),
]

FORMAT_PARAMETERS = {
    "City Convenience": {
        "area": (130, 380),
        "footfall": (650, 1450),
        "store_factor": (0.80, 1.10),
    },
    "High Street": {
        "area": (450, 1200),
        "footfall": (500, 1100),
        "store_factor": (0.90, 1.20),
    },
    "Suburban Supermarket": {
        "area": (1600, 4200),
        "footfall": (850, 1800),
        "store_factor": (1.10, 1.55),
    },
    "Retail Park Superstore": {
        "area": (5000, 11000),
        "footfall": (1300, 3000),
        "store_factor": (1.35, 2.00),
    },
}

CATEGORY_CONFIG = {
    "Fresh Produce": {
        "subcategories": ["Fruit", "Vegetables", "Salads", "Fresh Herbs"],
        "base_demand": (8, 36),
        "price": (0.70, 4.50),
        "shelf_life": (3, 10),
        "perishable_probability": 0.98,
        "seasonal": ["None", "Summer", "Winter"],
        "elasticity": "High",
        "volatility": "High",
    },
    "Bakery": {
        "subcategories": ["Bread", "Morning Goods", "Cakes", "Biscuits"],
        "base_demand": (7, 28),
        "price": (0.80, 4.20),
        "shelf_life": (2, 8),
        "perishable_probability": 0.78,
        "seasonal": ["None", "Christmas", "Easter"],
        "elasticity": "Medium",
        "volatility": "Medium",
    },
    "Dairy & Eggs": {
        "subcategories": ["Milk", "Cheese", "Yogurt", "Eggs", "Butter"],
        "base_demand": (8, 30),
        "price": (1.00, 5.50),
        "shelf_life": (5, 25),
        "perishable_probability": 0.88,
        "seasonal": ["None", "Christmas", "Easter"],
        "elasticity": "Low",
        "volatility": "Medium",
    },
    "Ambient Grocery": {
        "subcategories": ["Tea & Coffee", "Pasta & Rice", "Tins", "Sauces", "Cereals"],
        "base_demand": (3, 18),
        "price": (0.90, 7.50),
        "shelf_life": (120, 730),
        "perishable_probability": 0.02,
        "seasonal": ["None", "Christmas", "Winter"],
        "elasticity": "Medium",
        "volatility": "Low",
    },
    "Frozen": {
        "subcategories": ["Frozen Meals", "Frozen Vegetables", "Ice Cream", "Frozen Pizza"],
        "base_demand": (3, 15),
        "price": (1.50, 6.50),
        "shelf_life": (120, 540),
        "perishable_probability": 0.22,
        "seasonal": ["None", "Summer", "Winter"],
        "elasticity": "Medium",
        "volatility": "Medium",
    },
    "Beverages": {
        "subcategories": ["Water", "Soft Drinks", "Juice", "Beer & Cider", "Wine"],
        "base_demand": (6, 35),
        "price": (0.60, 12.00),
        "shelf_life": (120, 730),
        "perishable_probability": 0.01,
        "seasonal": ["Summer", "Christmas", "None"],
        "elasticity": "High",
        "volatility": "High",
    },
    "Snacks & Confectionery": {
        "subcategories": ["Crisps", "Chocolate", "Sweets", "Biscuits", "Nuts"],
        "base_demand": (5, 25),
        "price": (0.70, 5.00),
        "shelf_life": (120, 540),
        "perishable_probability": 0.01,
        "seasonal": ["None", "Christmas", "Easter"],
        "elasticity": "High",
        "volatility": "Medium",
    },
    "Household & Cleaning": {
        "subcategories": ["Laundry", "Cleaning", "Paper Products", "Washing Up"],
        "base_demand": (2, 12),
        "price": (1.20, 9.00),
        "shelf_life": (365, 1460),
        "perishable_probability": 0.00,
        "seasonal": ["None", "Winter"],
        "elasticity": "Medium",
        "volatility": "Low",
    },
    "Health & Beauty": {
        "subcategories": ["Toiletries", "Skincare", "Oral Care", "Baby Care"],
        "base_demand": (1, 9),
        "price": (1.50, 14.00),
        "shelf_life": (365, 1095),
        "perishable_probability": 0.00,
        "seasonal": ["None", "Christmas"],
        "elasticity": "Low",
        "volatility": "Low",
    },
    "Seasonal": {
        "subcategories": ["Christmas", "Easter", "Summer Outdoor", "Halloween"],
        "base_demand": (1, 14),
        "price": (1.00, 15.00),
        "shelf_life": (90, 730),
        "perishable_probability": 0.02,
        "seasonal": ["Christmas", "Easter", "Summer", "Winter"],
        "elasticity": "High",
        "volatility": "High",
    },
}

CATEGORY_WEIGHTS = {
    "Fresh Produce": 0.13,
    "Bakery": 0.09,
    "Dairy & Eggs": 0.11,
    "Ambient Grocery": 0.16,
    "Frozen": 0.08,
    "Beverages": 0.12,
    "Snacks & Confectionery": 0.11,
    "Household & Cleaning": 0.08,
    "Health & Beauty": 0.06,
    "Seasonal": 0.06,
}

SPECIAL_PRODUCT_NAMES = [
    ("Fresh Produce", "British Strawberries 400g", "Fruit"),
    ("Dairy & Eggs", "Northstar Free Range Eggs 6 Pack", "Eggs"),
    ("Ambient Grocery", "Yorkshire Tea 80 Bags", "Tea & Coffee"),
    ("Beverages", "Sparkling Water 500ml", "Water"),
    ("Bakery", "Northstar White Farmhouse Loaf 800g", "Bread"),
    ("Fresh Produce", "British Maris Piper Potatoes 2.5kg", "Vegetables"),
    ("Dairy & Eggs", "Northstar Semi Skimmed Milk 2 Pints", "Milk"),
    ("Frozen", "Northstar Vanilla Ice Cream 1L", "Ice Cream"),
    ("Snacks & Confectionery", "Milk Chocolate Bar 100g", "Chocolate"),
    ("Seasonal", "Northstar Easter Milk Chocolate Egg 180g", "Easter"),
]

PROMO_TYPES = [
    "Percent Off",
    "Multi-buy",
    "Clubcard-style Price",
    "Bundle",
    "Display-only",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_output_directory() -> None:
    """Create a clean output directory for the regenerable fact tables."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ground truth and reports live outside data/raw and are not wiped, so a
    # failed run cannot silently destroy the validation artifact.
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def easter_sunday(year: int) -> date:
    """
    Return Gregorian Easter Sunday using the Meeus/Jones/Butcher algorithm.

    The single-letter names are the algorithm's own, kept verbatim so this can be
    checked line by line against the published form. `l` is flagged as ambiguous
    by convention; renaming it would make the correspondence harder to verify,
    which matters more here than the letter's shape.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741 — the algorithm's own name
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day_value = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day_value)


def england_wales_holidays(year: int) -> Dict[date, str]:
    """Create plausible England/Wales bank holiday dates."""
    holidays: Dict[date, str] = {
        date(year, 1, 1): "New Year's Day",
        date(year, 12, 25): "Christmas Day",
        date(year, 12, 26): "Boxing Day",
    }
    easter = easter_sunday(year)
    holidays[easter - timedelta(days=2)] = "Good Friday"
    holidays[easter + timedelta(days=1)] = "Easter Monday"

    may_day = date(year, 5, 1)
    holidays[may_day + timedelta(days=(7 - may_day.weekday()) % 7)] = "Early May Bank Holiday"

    last_may = date(year, 5, 31)
    holidays[last_may - timedelta(days=(last_may.weekday() - 0) % 7)] = "Spring Bank Holiday"

    last_aug = date(year, 8, 31)
    holidays[last_aug - timedelta(days=(last_aug.weekday() - 0) % 7)] = "Summer Bank Holiday"
    return holidays


def scotland_holidays(year: int) -> Dict[date, str]:
    """Create broadly plausible Scottish holidays."""
    holidays = england_wales_holidays(year)
    holidays[date(year, 1, 2)] = "2 January Holiday"
    aug_first = date(year, 8, 1)
    scottish_august = aug_first + timedelta(days=(7 - aug_first.weekday()) % 7)
    holidays[scottish_august] = "Scottish Summer Bank Holiday"
    return holidays


def weighted_choice(items: List[str], weights: List[float]) -> str:
    return str(RNG.choice(items, p=np.array(weights) / np.sum(weights)))


def unit_description(category: str, subcategory: str) -> Tuple[str, str]:
    """Return a realistic unit size and measure."""
    choices = {
        "Fresh Produce": [("400g", "g"), ("1kg", "g"), ("Each", "each"), ("250g", "g")],
        "Bakery": [("400g", "g"), ("800g", "g"), ("4 Pack", "pack"), ("6 Pack", "pack")],
        "Dairy & Eggs": [("500g", "g"), ("1L", "ml"), ("6 Pack", "pack"), ("200g", "g")],
        "Ambient Grocery": [("500g", "g"), ("1kg", "g"), ("80 Bags", "bags"), ("400g", "g")],
        "Frozen": [("500g", "g"), ("1kg", "g"), ("1L", "ml"), ("2 Pack", "pack")],
        "Beverages": [("500ml", "ml"), ("1L", "ml"), ("2L", "ml"), ("4 Pack", "pack")],
        "Snacks & Confectionery": [("100g", "g"), ("150g", "g"), ("6 Pack", "pack")],
        "Household & Cleaning": [("750ml", "ml"), ("1L", "ml"), ("12 Pack", "pack")],
        "Health & Beauty": [("250ml", "ml"), ("75ml", "ml"), ("Each", "each")],
        "Seasonal": [("180g", "g"), ("300g", "g"), ("Each", "each"), ("6 Pack", "pack")],
    }
    return choices[category][int(RNG.integers(0, len(choices[category])))]


def product_name(category: str, subcategory: str, brand_type: str, unit_size: str) -> str:
    """Create plausible UK grocery product names."""
    prefixes = {
        "Own Label": "Northstar",
        "Branded": RNG.choice(
            ["Britannia", "Harbour", "Yorkshire", "Oakfield", "Crown", "Greenfield"]
        ),
        "Premium": "Northstar Finest",
    }
    nouns = {
        "Fruit": [
            "British Strawberries", "Easy Peeler Oranges", "Pink Lady Apples",
            "Seedless Grapes",
        ],
        "Vegetables": [
            "British Carrots", "Maris Piper Potatoes", "Tenderstem Broccoli",
            "Baby Spinach",
        ],
        "Salads": ["Mixed Leaf Salad", "Classic Coleslaw", "Sweetcorn Salad"],
        "Fresh Herbs": ["Fresh Basil", "Fresh Coriander", "Fresh Parsley"],
        "Bread": ["White Farmhouse Loaf", "Wholemeal Bread", "Sourdough Loaf"],
        "Morning Goods": ["All Butter Croissants", "Chocolate Muffins", "Pain au Chocolat"],
        "Cakes": ["Victoria Sponge Cake", "Chocolate Brownies", "Lemon Drizzle Cake"],
        "Biscuits": ["Digestive Biscuits", "Chocolate Chip Cookies", "Custard Creams"],
        "Milk": ["Semi Skimmed Milk", "Whole Milk", "Lactose Free Milk"],
        "Cheese": ["Mature Cheddar", "Mozzarella", "Red Leicester"],
        "Yogurt": ["Greek Style Yogurt", "Strawberry Yogurt", "Natural Yogurt"],
        "Eggs": ["Free Range Eggs", "Large British Eggs"],
        "Butter": ["Salted Butter", "Unsalted Butter"],
        "Tea & Coffee": ["Breakfast Tea", "Gold Blend Coffee", "Decaf Tea"],
        "Pasta & Rice": ["Penne Pasta", "Basmati Rice", "Spaghetti"],
        "Tins": ["Chopped Tomatoes", "Baked Beans", "Sweetcorn"],
        "Sauces": ["Tomato Pasta Sauce", "Curry Sauce", "Mayonnaise"],
        "Cereals": ["Oat Cereal", "Honey Granola", "Corn Flakes"],
        "Frozen Meals": ["Chicken Tikka Meal", "Lasagne Meal", "Vegetable Curry"],
        "Frozen Vegetables": ["Garden Peas", "Mixed Vegetables", "Sweetcorn"],
        "Ice Cream": ["Vanilla Ice Cream", "Salted Caramel Ice Cream"],
        "Frozen Pizza": ["Margherita Pizza", "Pepperoni Pizza"],
        "Water": ["Sparkling Water", "Still Water"],
        "Soft Drinks": ["Cola", "Lemonade", "Orange Fizz"],
        "Juice": ["Orange Juice", "Apple Juice"],
        "Beer & Cider": ["Premium Lager", "Apple Cider"],
        "Wine": ["Merlot Red Wine", "Pinot Grigio"],
        "Crisps": ["Sea Salt Crisps", "Cheese & Onion Crisps"],
        "Chocolate": ["Milk Chocolate Bar", "Dark Chocolate Bar"],
        "Sweets": ["Fruit Pastilles", "Jelly Sweets"],
        "Nuts": ["Salted Peanuts", "Mixed Nuts"],
        "Laundry": ["Colour Laundry Liquid", "Laundry Capsules"],
        "Cleaning": ["Multi Surface Cleaner", "Bathroom Spray"],
        "Paper Products": ["Kitchen Roll", "Toilet Tissue"],
        "Washing Up": ["Washing Up Liquid", "Dishwasher Tablets"],
        "Toiletries": ["Shower Gel", "Deodorant", "Hand Soap"],
        "Skincare": ["Moisturising Cream", "Face Wash"],
        "Oral Care": ["Toothpaste", "Toothbrush"],
        "Baby Care": ["Baby Wipes", "Nappy Pants"],
        "Christmas": ["Christmas Chocolate Selection", "Festive Shortbread"],
        "Easter": ["Easter Milk Chocolate Egg", "Mini Chocolate Eggs"],
        "Summer Outdoor": ["Barbecue Charcoal", "Picnic Plates"],
        "Halloween": ["Halloween Sweet Tub", "Pumpkin Decoration"],
    }
    noun = str(RNG.choice(nouns.get(subcategory, ["Everyday Grocery Item"])))
    return f"{prefixes[brand_type]} {noun} {unit_size}"


def write_csv(df: pd.DataFrame, filename: str, mode: str = "w") -> None:
    """Write a dataframe with a header only when creating a new file."""
    path = OUTPUT_DIR / filename
    df.to_csv(path, mode=mode, index=False, header=(mode == "w"))


# =============================================================================
# DIMENSIONS
# =============================================================================

def generate_stores(n_stores: int) -> pd.DataFrame:
    formats = list(FORMAT_PARAMETERS.keys())
    format_weights = [0.30, 0.28, 0.27, 0.15]
    rows: List[Dict[str, Any]] = []

    for idx in range(n_stores):
        store_format = weighted_choice(formats, format_weights)
        city, region, country, lat, lon = REGIONAL_LOCATIONS[idx % len(REGIONAL_LOCATIONS)]
        params = FORMAT_PARAMETERS[store_format]
        competition = int(RNG.integers(2, 11))
        deprivation = int(RNG.integers(1, 11))
        income_index = round(float(np.clip(RNG.normal(100, 15), 65, 145)), 1)
        postcode_prefix = city[:2].upper().replace(" ", "")
        postcode_district = f"{postcode_prefix}{int(RNG.integers(1, 20))}"
        footfall = int(RNG.integers(*params["footfall"]) * (1 + (income_index - 100) / 700))
        opening_year = int(RNG.integers(2005, 2023))

        rows.append(
            {
                "store_id": f"STR{idx + 1:03d}",
                "store_name": f"Northstar {city} {store_format}",
                "city": city,
                "region": region,
                "country": country,
                "postcode_district": postcode_district,
                "store_format": store_format,
                "opening_date": date(
                    opening_year, int(RNG.integers(1, 13)), int(RNG.integers(1, 28))
                ).isoformat(),
                "floor_area_sqm": int(RNG.integers(*params["area"])),
                "local_deprivation_decile": deprivation,
                "competition_intensity_score": competition,
                "average_daily_footfall": footfall,
                "store_income_index": income_index,
                "latitude": round(lat + float(RNG.normal(0, 0.06)), 5),
                "longitude": round(lon + float(RNG.normal(0, 0.08)), 5),
                "is_active": True,
                "store_demand_factor": round(
                    float(RNG.uniform(*params["store_factor"]) * (1 - 0.012 * (competition - 5))),
                    4,
                ),
            }
        )
    stores = pd.DataFrame(rows)
    # Rollout cohort is a store attribute, so it belongs on the store dimension
    # rather than being re-derived wherever a campaign is built. Cohorts are
    # assigned by footfall rank, which is what makes campaign entry staggered.
    footfall_rank = stores["average_daily_footfall"].rank(method="first", ascending=False)
    stores["rollout_cohort"] = ((footfall_rank - 1) % N_COHORTS).astype(int)
    return stores


def generate_products(n_skus: int) -> pd.DataFrame:
    categories = list(CATEGORY_CONFIG.keys())
    category_weights = [CATEGORY_WEIGHTS[c] for c in categories]
    rows: List[Dict[str, Any]] = []

    for idx in range(n_skus):
        if idx < len(SPECIAL_PRODUCT_NAMES):
            category, name, subcategory = SPECIAL_PRODUCT_NAMES[idx]
        else:
            category = weighted_choice(categories, category_weights)
            cfg = CATEGORY_CONFIG[category]
            subcategory = str(RNG.choice(cfg["subcategories"]))
            name = ""

        cfg = CATEGORY_CONFIG[category]
        brand_type = weighted_choice(["Own Label", "Branded", "Premium"], [0.49, 0.36, 0.15])
        unit_size, measure = unit_description(category, subcategory)

        if not name:
            name = product_name(category, subcategory, brand_type, unit_size)

        base_price = float(RNG.uniform(*cfg["price"]))
        if brand_type == "Branded":
            base_price *= float(RNG.uniform(1.10, 1.35))
        elif brand_type == "Premium":
            base_price *= float(RNG.uniform(1.30, 1.70))

        margin = float(RNG.uniform(0.29, 0.42))
        if brand_type == "Own Label":
            margin += float(RNG.uniform(0.05, 0.12))
        elif brand_type == "Premium":
            margin += float(RNG.uniform(0.03, 0.09))
        margin = min(margin, 0.60)
        price = round(max(0.50, base_price), 2)
        cost = round(price * (1 - margin), 2)
        cost = min(cost, round(price - 0.01, 2))

        is_perishable = bool(RNG.random() < cfg["perishable_probability"])
        shelf_life = int(RNG.integers(*cfg["shelf_life"]))
        if not is_perishable:
            shelf_life = max(shelf_life, 90)

        promotion_sensitivity = cfg["elasticity"]
        if brand_type == "Own Label" and promotion_sensitivity != "High":
            promotion_sensitivity = weighted_choice(["Medium", "High"], [0.45, 0.55])

        price_elasticity = cfg["elasticity"]
        if brand_type == "Branded":
            price_elasticity = weighted_choice(["Low", "Medium"], [0.55, 0.45])

        rows.append(
            {
                "sku_id": f"SKU{idx + 1:04d}",
                "sku_name": name,
                "brand_type": brand_type,
                "category": category,
                "subcategory": subcategory,
                "supplier_id": f"SUP{int(RNG.integers(1, 46)):03d}",
                "shelf_life_days": shelf_life,
                "is_perishable": is_perishable,
                "unit_size": unit_size,
                "unit_of_measure": measure,
                "regular_unit_price_gbp": price,
                "unit_cost_gbp": cost,
                "baseline_gross_margin_pct": round((price - cost) / price * 100, 2),
                "price_elasticity_segment": price_elasticity,
                "promotion_sensitivity_segment": promotion_sensitivity,
                "demand_volatility_segment": cfg["volatility"],
                "seasonal_profile": str(RNG.choice(cfg["seasonal"])),
                "minimum_display_stock": int(RNG.integers(2, 10)),
                "reorder_lead_time_days": int(RNG.integers(1, 5 if is_perishable else 9)),
                "base_demand_units": round(float(RNG.uniform(*cfg["base_demand"])), 3),
                # Fractional change in latent demand per year. Negative values are
                # "weakening momentum" SKUs, which are preferentially selected for
                # promotion (see generate_promotions). This trend is applied for real
                # in generate_facts, so the resulting decline is visible in observed
                # sales history and is therefore recoverable by propensity models.
                "demand_trend_annual": round(
                    float(np.clip(RNG.normal(-0.02, 0.12), -0.35, 0.30)), 4
                ),
                # A minority of SKUs are never promoted, guaranteeing a genuine
                # never-treated control pool for difference-in-differences.
                "promotion_eligible": bool(RNG.random() < 0.88),
            }
        )
    return pd.DataFrame(rows)


def generate_calendar(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows: List[Dict[str, Any]] = []

    for ts in dates:
        current = ts.date()
        year = current.year
        england_wales = england_wales_holidays(year)
        scotland = scotland_holidays(year)
        all_holidays = {**england_wales, **scotland}
        holiday_name = all_holidays.get(current, "")
        easter = easter_sunday(year)

        month = current.month
        day_of_year = current.timetuple().tm_yday
        base_temp = 10 + 7.8 * math.sin((day_of_year - 82) * 2 * math.pi / 365.25)
        temperature = float(np.clip(RNG.normal(base_temp, 3.2), -8, 35))
        rainfall_mean = 1.2 if month in [6, 7, 8] else 3.0 if month in [9, 10, 11, 12] else 2.2
        rainfall = round(float(RNG.gamma(shape=1.5, scale=rainfall_mean)), 1)

        heatwave = bool(month in [5, 6, 7, 8] and temperature >= 25.0 and rainfall < 2.5)
        if heatwave:
            weather = "Heatwave"
        elif rainfall >= 9:
            weather = "Heavy Rain"
        elif rainfall >= 3:
            weather = "Rain"
        elif temperature <= 2:
            weather = "Cold"
        elif temperature >= 21:
            weather = "Warm"
        else:
            weather = "Cloudy" if RNG.random() < 0.55 else "Dry"

        school_holiday = (
            (month == 2 and 12 <= current.day <= 18)
            or (month in [4, 8])
            or (month == 10 and 21 <= current.day <= 31)
            or (month == 12 and current.day >= 20)
        )
        payday_window = current.day >= 25 or current.day <= 3
        christmas_period = (month == 12 and current.day >= 1) or (month == 1 and current.day <= 2)
        black_friday_period = month == 11 and 20 <= current.day <= 30
        easter_period = abs((current - easter).days) <= 10

        rows.append(
            {
                "date": current.isoformat(),
                "year": year,
                "quarter": f"Q{((month - 1) // 3) + 1}",
                "month": month,
                "month_name": calendar.month_name[month],
                "week_of_year": int(ts.isocalendar().week),
                "day_of_week": current.weekday(),
                "day_name": calendar.day_name[current.weekday()],
                "is_weekend": current.weekday() >= 5,
                "is_month_end": (current + timedelta(days=1)).month != month,
                "is_payday_window": payday_window,
                "is_school_holiday": school_holiday,
                "is_bank_holiday": bool(holiday_name),
                "bank_holiday_name": holiday_name,
                "is_easter_period": easter_period,
                "is_christmas_period": christmas_period,
                "is_black_friday_period": black_friday_period,
                "is_heatwave": heatwave,
                "weather_condition": weather,
                "temperature_celsius": round(temperature, 1),
                "rainfall_mm": rainfall,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# PROMOTIONS AND GROUND TRUTH
# =============================================================================

def generate_ground_truth(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    elasticity_map = {"Low": -0.55, "Medium": -1.05, "High": -1.65}
    sensitivity_map = {"Low": 0.08, "Medium": 0.18, "High": 0.31}
    volatility_map = {"Low": 0.07, "Medium": 0.14, "High": 0.24}

    for product in products.itertuples(index=False):
        elasticity = (
            elasticity_map[product.price_elasticity_segment] * float(RNG.uniform(0.82, 1.18))
        )
        promo_uplift = (
            sensitivity_map[product.promotion_sensitivity_segment]
            * float(RNG.uniform(0.80, 1.30))
        )
        rows.append(
            {
                "sku_id": product.sku_id,
                "category": product.category,
                "true_base_demand": product.base_demand_units,
                "true_price_elasticity": round(elasticity, 4),
                "true_promo_uplift_pct": round(promo_uplift * 100, 2),
                "true_display_uplift_pct": round(float(RNG.uniform(6, 18)), 2),
                "true_email_app_uplift_pct": round(float(RNG.uniform(3, 11)), 2),
                "true_seasonality_strength": round(
                    volatility_map[product.demand_volatility_segment], 4
                ),
                "true_weekend_effect_pct": round(float(RNG.uniform(3, 18)), 2),
                "true_stockout_lost_sales_factor": round(float(RNG.uniform(0.70, 1.00)), 3),
                "true_cannibalisation_factor": round(float(RNG.uniform(0.02, 0.10)), 3),
                "true_margin_impact_of_promotion": round(float(RNG.uniform(0.20, 0.75)), 3),
                "true_demand_trend_pct_per_year": round(product.demand_trend_annual * 100, 2),
                "data_generation_notes": (
                    "Latent demand includes category seasonality, store effect, price response, "
                    "promotion effect, support-channel uplift, long-run trend, autocorrelation "
                    "and random noise. IMPORTANT: true_promo_uplift_pct is a structural "
                    "coefficient applied per 10 percentage points of discount, and it compounds "
                    "with the separate price response. It is NOT the average treatment effect "
                    "and must not be compared directly against a DiD or PSM estimate. Use "
                    "true_realised_att_pct for that comparison."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_timing_weights(
    calendar_df: pd.DataFrame,
    products: pd.DataFrame,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Precompute a promo-start-date probability vector per (seasonal_profile, category).

    The original draft rebuilt these weights inside the per-event loop, which is
    O(events x days) with a dict lookup per cell. At a realistic promo density that
    is tens of millions of lookups. There are at most 5 x 10 distinct combinations,
    so they are cached once here instead.
    """
    base = np.ones(len(calendar_df), dtype=float)
    christmas = calendar_df["is_christmas_period"].to_numpy(dtype=float)
    easter = calendar_df["is_easter_period"].to_numpy(dtype=float)
    heatwave = calendar_df["is_heatwave"].to_numpy(dtype=float)
    base += 4.5 * christmas
    base += 2.6 * easter
    base += 1.7 * calendar_df["is_bank_holiday"].to_numpy(dtype=float)
    base += 0.7 * calendar_df["is_payday_window"].to_numpy(dtype=float)

    weights: Dict[Tuple[str, str], np.ndarray] = {}
    combos = products[["seasonal_profile", "category"]].drop_duplicates()
    for profile, category in combos.itertuples(index=False):
        weight = base.copy()
        if category in ("Beverages", "Frozen", "Seasonal"):
            weight += 3.0 * heatwave
        if profile == "Christmas":
            weight += 4.0 * christmas
        if profile == "Easter":
            weight += 3.0 * easter
        weights[(str(profile), str(category))] = weight / weight.sum()
    return weights


def generate_promotions(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deliberately non-random promotion assignment.

    Selection bias (never random):
      - high baseline margin, seasonal relevance and own-label status raise a SKU's
        selection probability;
      - SKUs with weakening demand momentum (negative demand_trend_annual) are
        preferentially promoted - the classic "prop up a declining line" behaviour.
        Unlike the original draft, that trend is actually applied to latent demand
        in generate_facts, so it leaves a visible pre-promotion decline that a
        propensity model can legitimately recover;
      - higher-footfall stores run more events;
      - timing clusters on Christmas, Easter, heatwaves, bank holidays and paydays.

    Identification structure:
      - CAMPAIGN_SHARE of events belong to multi-store campaigns sharing a
        campaign_id. Stores join by rollout cohort, staggered by
        COHORT_OFFSETS_DAYS. This is the staggered rollout that makes
        difference-in-differences identifiable, and it replaces the earlier
        12%-of-events date floor, which never produced a coordinated rollout.
      - Remaining events are store-local tactical promotions supplying realistic
        assignment noise.
      - SKUs flagged promotion_eligible=False are never promoted anywhere, which
        preserves a genuine never-treated control pool.
    """
    dates = pd.to_datetime(calendar_df["date"]).dt.date.tolist()
    n_days = len(dates)
    end_bound = pd.Timestamp(END_DATE).date()
    calendar_lookup = calendar_df.set_index("date").to_dict("index")
    timing_weights = build_timing_weights(calendar_df, products)

    store_rows = list(stores.itertuples(index=False))
    stores_by_cohort: Dict[int, List[Any]] = defaultdict(list)
    for store_row in store_rows:
        stores_by_cohort[int(store_row.rollout_cohort)].append(store_row)

    eligible = products[products["promotion_eligible"]].reset_index(drop=True)
    product_lookup = {row.sku_id: row for row in eligible.itertuples(index=False)}
    product_ids = eligible["sku_id"].tolist()

    margin = eligible["baseline_gross_margin_pct"].to_numpy(dtype=float) / 100
    seasonal_bonus = np.where(eligible["seasonal_profile"].to_numpy() != "None", 0.22, 0.0)
    own_label_bonus = np.where(eligible["brand_type"].to_numpy() == "Own Label", 0.10, 0.0)
    # A -20%/yr trend earns a large bonus; a growing line earns none.
    trend = eligible["demand_trend_annual"].to_numpy(dtype=float)
    weakening_bonus = np.clip(-trend, 0.0, None) * 1.6
    product_probs = margin + seasonal_bonus + own_label_bonus + weakening_bonus
    product_probs /= product_probs.sum()

    store_intensity = np.array(
        [
            0.45
            + 0.00018 * s.average_daily_footfall
            + 0.07 * (s.store_format in ("Suburban Supermarket", "Retail Park Superstore"))
            for s in store_rows
        ],
        dtype=float,
    )
    store_probs = store_intensity / store_intensity.sum()

    # Event budget sized to land near the configured promo density. Overlapping
    # events on the same store x SKU collapse in build_promo_lookup, so realised
    # density comes out a little under target and is reported by print_summary.
    mean_duration_days = 8.5
    total_events = int(
        PROMO_DENSITY_TARGET * len(stores) * len(products) * n_days / mean_duration_days
    )
    n_campaign_events = int(total_events * CAMPAIGN_SHARE)

    promo_rows: List[Dict[str, Any]] = []
    promo_id = 1

    def emit(store: Any, sku_id: str, start: date, campaign_id: str, wave_days: int) -> None:
        """Append one promotion event, shared by the campaign and tactical paths."""
        nonlocal promo_id
        if start > end_bound:
            return
        product = product_lookup[sku_id]
        duration = int(RNG.integers(3, 15))
        end = min(start + timedelta(days=duration - 1), end_bound)
        promo_type = weighted_choice(PROMO_TYPES, [0.42, 0.18, 0.18, 0.10, 0.12])
        discount = 0 if promo_type == "Display-only" else int(RNG.choice([5, 10, 15, 20, 25, 30]))
        display = bool(promo_type == "Display-only" or RNG.random() < 0.32)
        email = bool(RNG.random() < 0.24)
        leaflet = bool(RNG.random() < 0.18)

        expected_units = product.base_demand_units * store.store_demand_factor * duration
        promo_cost = max(
            5.0,
            expected_units
            * product.regular_unit_price_gbp
            * (discount / 100)
            * float(RNG.uniform(0.12, 0.32)),
        )

        record = calendar_lookup[start.isoformat()]
        if record["is_christmas_period"]:
            theme = "Festive Favourites"
        elif record["is_easter_period"]:
            theme = "Easter Treats"
        elif record["is_heatwave"]:
            theme = "Summer Refresh"
        elif record["is_bank_holiday"]:
            theme = "Bank Holiday Event"
        elif record["is_payday_window"]:
            theme = "Payday Offers"
        else:
            theme = "Everyday Value"

        promo_rows.append(
            {
                "promotion_id": f"PRM{promo_id:07d}",
                "campaign_id": campaign_id,
                "campaign_wave_days": wave_days,
                "store_id": store.store_id,
                "sku_id": sku_id,
                "promo_start_date": start.isoformat(),
                "promo_end_date": end.isoformat(),
                "promo_type": promo_type,
                "discount_pct": discount,
                "display_support_flag": display,
                "email_or_app_support_flag": email,
                "leaflet_support_flag": leaflet,
                "campaign_theme": theme,
                "promotion_cost_gbp": round(float(promo_cost), 2),
                "vendor_funded_pct": round(float(RNG.choice([0, 25, 40, 50, 60, 75])), 2),
                # National campaigns are planned; tactical store promos are not.
                "promotion_planned_flag": bool(campaign_id),
            }
        )
        promo_id += 1

    # ---- multi-store staggered campaigns (the DiD-identifying variation) ----
    emitted = 0
    campaign_number = 1
    max_per_cohort = max(2, len(store_rows) // N_COHORTS)
    while emitted < n_campaign_events:
        sku_id = str(RNG.choice(product_ids, p=product_probs))
        product = product_lookup[sku_id]
        weights = timing_weights[(str(product.seasonal_profile), str(product.category))]
        base_start = dates[int(RNG.choice(n_days, p=weights))]
        per_cohort = int(RNG.integers(2, max_per_cohort + 1))
        campaign_id = f"CMP{campaign_number:05d}"

        for cohort in range(N_COHORTS):
            members = stores_by_cohort[cohort]
            if not members:
                continue
            wave_days = COHORT_OFFSETS_DAYS[cohort % len(COHORT_OFFSETS_DAYS)]
            start = base_start + timedelta(days=wave_days)
            take = min(per_cohort, len(members))
            for idx in RNG.choice(len(members), size=take, replace=False):
                emit(members[int(idx)], sku_id, start, campaign_id, wave_days)
                emitted += 1
        campaign_number += 1

    # ---- store-local tactical promotions ----
    for _ in range(max(0, total_events - emitted)):
        store = store_rows[int(RNG.choice(len(store_rows), p=store_probs))]
        sku_id = str(RNG.choice(product_ids, p=product_probs))
        product = product_lookup[sku_id]
        weights = timing_weights[(str(product.seasonal_profile), str(product.category))]
        emit(store, sku_id, dates[int(RNG.choice(n_days, p=weights))], "", 0)

    promotions = pd.DataFrame(promo_rows).drop_duplicates(
        subset=["store_id", "sku_id", "promo_start_date"],
        keep="first",
    )
    return promotions.reset_index(drop=True)


def build_promo_lookup(
    promotions: pd.DataFrame,
) -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    lookup: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)

    for row in promotions.itertuples(index=False):
        for dt in pd.date_range(row.promo_start_date, row.promo_end_date, freq="D"):
            key = (row.store_id, row.sku_id)
            existing = lookup[dt.strftime("%Y-%m-%d")].get(key)
            if existing is None or row.discount_pct > existing["discount_pct"]:
                lookup[dt.strftime("%Y-%m-%d")][key] = {
                    "promo_type": row.promo_type,
                    "discount_pct": row.discount_pct,
                    "display_support_flag": row.display_support_flag,
                    "email_or_app_support_flag": row.email_or_app_support_flag,
                    "leaflet_support_flag": row.leaflet_support_flag,
                    "promotion_cost_gbp": row.promotion_cost_gbp,
                    "vendor_funded_pct": row.vendor_funded_pct,
                    "duration_days": max(
                        1,
                        (
                            pd.Timestamp(row.promo_end_date)
                            - pd.Timestamp(row.promo_start_date)
                        ).days + 1,
                    ),
                }
    return lookup


# =============================================================================
# FACT TABLE SIMULATION
# =============================================================================

def seasonality_multiplier(product: pd.Series, cal: Dict[str, Any]) -> float:
    category = product["category"]
    profile = product["seasonal_profile"]
    multiplier = 1.0

    if cal["is_weekend"]:
        multiplier *= 1.06

    if category == "Beverages" and cal["temperature_celsius"] >= 21:
        multiplier *= 1.18
    if category == "Beverages" and cal["is_heatwave"]:
        multiplier *= 1.40
    if category == "Frozen" and cal["temperature_celsius"] >= 20:
        multiplier *= 1.18
    if category == "Fresh Produce" and cal["temperature_celsius"] >= 20:
        multiplier *= 1.08
    if category == "Bakery" and cal["is_weekend"]:
        multiplier *= 1.10
    if category == "Household & Cleaning" and cal["rainfall_mm"] >= 8:
        multiplier *= 1.04

    if profile == "Summer" and cal["month"] in [6, 7, 8]:
        multiplier *= 1.24
    if profile == "Winter" and cal["month"] in [11, 12, 1, 2]:
        multiplier *= 1.18
    if profile == "Christmas" and cal["is_christmas_period"]:
        multiplier *= 2.40
    if profile == "Easter" and cal["is_easter_period"]:
        multiplier *= 2.15

    if cal["is_christmas_period"] and category in [
        "Seasonal",
        "Snacks & Confectionery",
        "Beverages",
        "Ambient Grocery",
    ]:
        multiplier *= 1.20
    if cal["is_easter_period"] and category in ["Bakery", "Snacks & Confectionery", "Seasonal"]:
        multiplier *= 1.22
    if cal["is_bank_holiday"]:
        multiplier *= 1.08
    if cal["is_payday_window"]:
        multiplier *= 1.04

    return multiplier


def generate_facts(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    promo_lookup: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Simulate stock, deliveries, latent demand and observed demand together.
    Facts are written incrementally to avoid holding the full table in memory.
    """
    LOGGER.info("Generating daily inventory and demand fact tables in chunks.")

    store_ids = stores["store_id"].tolist()
    sku_ids = products["sku_id"].tolist()
    n_stores = len(store_ids)
    n_skus = len(sku_ids)
    n_pairs = n_stores * n_skus

    product_index = products.set_index("sku_id").loc[sku_ids].reset_index()
    store_index = stores.set_index("store_id").loc[store_ids].reset_index()
    truth_index = ground_truth.set_index("sku_id").loc[sku_ids].reset_index()

    store_factor = store_index["store_demand_factor"].to_numpy(dtype=float)
    store_footfall_base = store_index["average_daily_footfall"].to_numpy(dtype=float)
    competition = store_index["competition_intensity_score"].to_numpy(dtype=float)

    base_demand = product_index["base_demand_units"].to_numpy(dtype=float)
    regular_price = product_index["regular_unit_price_gbp"].to_numpy(dtype=float)
    unit_cost = product_index["unit_cost_gbp"].to_numpy(dtype=float)
    # `shelf_life_days` is deliberately not pulled here: waste is driven by
    # `is_perishable` below, and the unused array only looked like an input.
    minimum_display = product_index["minimum_display_stock"].to_numpy(dtype=int)
    lead_time = product_index["reorder_lead_time_days"].to_numpy(dtype=int)
    is_perishable = product_index["is_perishable"].to_numpy(dtype=bool)
    categories = product_index["category"].to_numpy(dtype=str)
    volatility = product_index["demand_volatility_segment"].to_numpy(dtype=str)
    demand_trend = product_index["demand_trend_annual"].to_numpy(dtype=float)
    elasticity = truth_index["true_price_elasticity"].to_numpy(dtype=float)
    promo_uplift = truth_index["true_promo_uplift_pct"].to_numpy(dtype=float) / 100
    display_uplift = truth_index["true_display_uplift_pct"].to_numpy(dtype=float) / 100
    email_uplift = truth_index["true_email_app_uplift_pct"].to_numpy(dtype=float) / 100
    cannibal_factor = truth_index["true_cannibalisation_factor"].to_numpy(dtype=float)

    pair_store_idx = np.repeat(np.arange(n_stores), n_skus)
    pair_sku_idx = np.tile(np.arange(n_skus), n_stores)
    pair_store_ids = np.repeat(np.array(store_ids), n_skus)
    pair_sku_ids = np.tile(np.array(sku_ids), n_stores)

    pair_base = base_demand[pair_sku_idx] * store_factor[pair_store_idx]
    pair_regular_price = regular_price[pair_sku_idx]
    pair_cost = unit_cost[pair_sku_idx]
    pair_lead_time = lead_time[pair_sku_idx]
    pair_perishable = is_perishable[pair_sku_idx]
    pair_min_display = minimum_display[pair_sku_idx]
    pair_categories = categories[pair_sku_idx]
    pair_elasticity = elasticity[pair_sku_idx]
    pair_promo_uplift = promo_uplift[pair_sku_idx]
    pair_display_uplift = display_uplift[pair_sku_idx]
    pair_email_uplift = email_uplift[pair_sku_idx]
    pair_cannibal = cannibal_factor[pair_sku_idx]
    pair_volatility = volatility[pair_sku_idx]
    pair_trend = demand_trend[pair_sku_idx]

    # Precomputed once for the day loop: SKU rows for the seasonality function, and
    # a flat (store, category) bin index for vectorised cannibalisation counting.
    product_series = [product_index.iloc[i] for i in range(n_skus)]
    category_codes = pd.Categorical(categories).codes.astype(int)
    n_categories = int(category_codes.max()) + 1
    n_store_category_bins = n_stores * n_categories
    pair_store_category_idx = pair_store_idx * n_categories + category_codes[pair_sku_idx]

    # True treatment effect accumulators. For a promoted row the counterfactual
    # "no promotion" latent demand differs from the actual only through the three
    # promotion channels (price response, promo uplift, support uplift), which
    # enter lambda multiplicatively. Their product minus one is therefore the exact
    # simulated treatment effect on latent demand for that row - the number Phase 4
    # must recover. Averaging over treated rows gives a true ATT per SKU.
    att_effect_sum = np.zeros(n_skus, dtype=float)
    att_effect_count = np.zeros(n_skus, dtype=float)

    initial_cover = RNG.integers(4, 10, size=n_pairs)
    opening_stock = np.maximum(
        pair_min_display + 1,
        np.ceil(pair_base * initial_cover).astype(int),
    )
    lag_1 = np.maximum(0, np.round(pair_base * RNG.uniform(0.7, 1.3, n_pairs))).astype(int)
    lag_7 = lag_1.copy()
    rolling_7 = lag_1.astype(float)
    rolling_28 = lag_1.astype(float)
    demand_memory = lag_1.astype(float)

    sales_history: List[np.ndarray] = []
    pending_deliveries: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_pairs, dtype=int))
    inventory_path = OUTPUT_DIR / "fact_inventory_delivery.csv"
    daily_path = OUTPUT_DIR / "fact_daily_store_sku.csv"

    metrics: Dict[str, Any] = {
        "rows": 0,
        "stockouts": 0,
        "promo_rows": 0,
        "revenue": 0.0,
        "gross_profit": 0.0,
        "promotion_distribution": Counter(),
        "category_sales": Counter(),
        "dates": [],
    }

    inventory_columns: List[str] = []
    daily_columns: List[str] = []
    inventory_chunks: List[pd.DataFrame] = []
    daily_chunks: List[pd.DataFrame] = []

    for day_idx, cal_row in enumerate(calendar_df.itertuples(index=False)):
        cal = cal_row._asdict()
        current_date = cal["date"]
        metrics["dates"].append(current_date)

        day_factor = np.ones(n_pairs, dtype=float)
        # Seasonality depends only on the SKU, so evaluate it once per SKU rather
        # than once per store x SKU pair (n_skus instead of n_pairs iterations).
        sku_seasonality = np.array(
            [seasonality_multiplier(row, cal) for row in product_series],
            dtype=float,
        )
        day_factor *= sku_seasonality[pair_sku_idx]

        weekday_effect = 1.0 + np.where(cal["day_of_week"] >= 5, 0.07, 0.0)
        day_factor *= weekday_effect

        # Long-run momentum. SKUs with a negative trend genuinely decline over the
        # panel, which is what makes them attractive promotion candidates and what
        # a propensity model can pick up from observed sales history.
        day_factor *= np.exp(np.log1p(pair_trend) * (day_idx / 365.25))

        store_footfall = (
            store_footfall_base[pair_store_idx]
            * (1 + np.where(cal["is_weekend"], 0.09, 0))
            * (1 + np.where(cal["is_bank_holiday"], 0.11, 0))
            * (1 + np.where(cal["is_school_holiday"], 0.04, 0))
            * RNG.normal(1.0, 0.05, n_pairs)
        )
        store_footfall = np.maximum(50, store_footfall).astype(int)

        local_event_store = RNG.random(n_stores) < 0.022
        local_event_flag = local_event_store[pair_store_idx]
        day_factor *= np.where(local_event_flag, RNG.uniform(1.06, 1.22, n_pairs), 1.0)

        supplier_disruption = RNG.random(n_pairs) < 0.0025
        closure_store = RNG.random(n_stores) < 0.00065
        closure_flag = closure_store[pair_store_idx]
        extreme_weather_flag = bool(
            cal["weather_condition"] == "Heavy Rain" and RNG.random() < 0.08
        )
        unusual_spike = RNG.random(n_pairs) < 0.0008
        data_entry_error = RNG.random(n_pairs) < 0.00025
        anomaly_flag = (
            closure_flag | supplier_disruption | unusual_spike | data_entry_error
            | extreme_weather_flag
        )

        scheduled = ((day_idx + pair_sku_idx) % np.maximum(pair_lead_time + 1, 2) == 0)
        reorder_point = np.ceil(
            pair_base * (pair_lead_time + 2.3) + pair_min_display
        ).astype(int)
        replenishment_need = opening_stock < reorder_point
        delivery_scheduled = scheduled | replenishment_need
        delivery_delay = delivery_scheduled & (
            (RNG.random(n_pairs) < 0.035) | supplier_disruption
        )

        target_stock = np.ceil(
            pair_base * RNG.integers(6, 12, size=n_pairs) + pair_min_display
        ).astype(int)
        delivery_units = np.where(
            delivery_scheduled & ~delivery_delay,
            np.maximum(0, target_stock - opening_stock) + RNG.integers(0, 6, size=n_pairs),
            0,
        ).astype(int)
        delivery_units += pending_deliveries.pop(day_idx, np.zeros(n_pairs, dtype=int))

        delayed_amount = np.where(
            delivery_delay,
            np.maximum(1, np.ceil(pair_base * RNG.uniform(1, 3, n_pairs))).astype(int),
            0,
        )
        for delay_day in [1, 2, 3]:
            arriving = np.where(
                delivery_delay & (RNG.integers(1, 4, size=n_pairs) == delay_day),
                delayed_amount,
                0,
            )
            pending_deliveries[day_idx + delay_day] += arriving

        promo_day = promo_lookup.get(current_date, {})
        promo_key_values = [
            promo_day.get((sid, skuid))
            for sid, skuid in zip(pair_store_ids, pair_sku_ids)
        ]
        promo_flag = np.array([x is not None for x in promo_key_values], dtype=bool)
        discount = np.array([x["discount_pct"] if x else 0 for x in promo_key_values], dtype=float)
        def _promo_field(key: str, default, dtype):
            """Pull one promotion attribute across the pair axis, defaulting off-promo rows."""
            return np.array(
                [x[key] if x else default for x in promo_key_values], dtype=dtype
            )

        promo_type = _promo_field("promo_type", "None", object)
        display_flag = _promo_field("display_support_flag", False, bool)
        email_flag = _promo_field("email_or_app_support_flag", False, bool)
        leaflet_flag = _promo_field("leaflet_support_flag", False, bool)
        vendor_funded = _promo_field("vendor_funded_pct", 0, float)
        promo_event_cost = _promo_field("promotion_cost_gbp", 0, float)
        promo_duration = _promo_field("duration_days", 1, float)

        available_before_sales = opening_stock + delivery_units

        # If nothing is physically available, no displayed promotion is observed that day.
        promo_flag = promo_flag & (available_before_sales > 0)
        discount = np.where(promo_flag, discount, 0)
        promo_type = np.where(promo_flag, promo_type, "None")
        display_flag &= promo_flag
        email_flag &= promo_flag
        leaflet_flag &= promo_flag

        actual_price = np.round(pair_regular_price * (1 - discount / 100), 2)
        actual_price = np.maximum(0.05, actual_price)

        price_ratio = actual_price / pair_regular_price
        price_effect = np.power(price_ratio, pair_elasticity)
        promo_effect = 1 + (pair_promo_uplift * (discount / 10)) * promo_flag
        promo_effect *= np.where(promo_type == "Multi-buy", 1.08, 1.0)
        promo_effect *= np.where(promo_type == "Clubcard-style Price", 1.05, 1.0)
        promo_effect *= np.where(promo_type == "Bundle", 1.10, 1.0)
        promo_effect *= np.where(promo_type == "Display-only", 1.06, 1.0)
        support_effect = (
            1
            + display_flag * pair_display_uplift
            + email_flag * pair_email_uplift
            + leaflet_flag * 0.07
        )

        # Exact simulated treatment effect on latent demand for promoted rows.
        # Equals 1.0 for untreated rows (no discount, no uplift, no support), so
        # only treated rows contribute.
        treatment_multiplier = price_effect * promo_effect * support_effect
        if promo_flag.any():
            treated_sku_idx = pair_sku_idx[promo_flag]
            np.add.at(att_effect_sum, treated_sku_idx, (treatment_multiplier - 1.0)[promo_flag])
            np.add.at(att_effect_count, treated_sku_idx, 1.0)

        # Category/store cannibalisation: multiple promotions reduce similar
        # non-promoted SKU demand. Counted with a flat (store, category) bin index
        # so this stays vectorised at realistic promotion density.
        category_promo_count = np.bincount(
            pair_store_category_idx[promo_flag], minlength=n_store_category_bins
        )
        promos_in_bin = category_promo_count[pair_store_category_idx]
        cannibalisation = np.where(
            promo_flag,
            1.0,
            np.maximum(0.82, 1 - pair_cannibal * promos_in_bin),
        )

        # Pantry loading: selected ambient and beverage products dip after promoted periods.
        pantry_dip = np.where(
            ((pair_categories == "Ambient Grocery") | (pair_categories == "Beverages"))
            & ~promo_flag
            & (demand_memory > pair_base * 1.35),
            0.91,
            1.0,
        )

        autocorrelation = 0.82 + 0.18 * np.clip(demand_memory / np.maximum(pair_base, 1), 0.5, 1.8)
        lambda_demand = (
            pair_base
            * day_factor
            * price_effect
            * promo_effect
            * support_effect
            * cannibalisation
            * pantry_dip
            * autocorrelation
        )
        lambda_demand *= np.where(unusual_spike, RNG.uniform(2.0, 4.0, n_pairs), 1.0)
        lambda_demand *= np.where(closure_flag, 0.0, 1.0)
        lambda_demand = np.maximum(0.01, lambda_demand)

        dispersion = np.where(
            pair_volatility == "High",
            4.0,
            np.where(pair_volatility == "Medium", 8.0, 15.0),
        )
        gamma_rate = RNG.gamma(shape=dispersion, scale=lambda_demand / dispersion)
        potential_demand = RNG.poisson(gamma_rate).astype(int)

        damaged = np.where(
            delivery_units > 0,
            RNG.binomial(delivery_units, np.where(pair_perishable, 0.006, 0.002)),
            0,
        ).astype(int)

        excess_stock_ratio = available_before_sales / np.maximum(lambda_demand * 3, 1)
        waste_probability = np.where(
            pair_perishable,
            np.clip(0.006 + 0.012 * np.maximum(excess_stock_ratio - 1, 0), 0.005, 0.12),
            0.0002,
        )
        waste = RNG.binomial(
            np.maximum(0, available_before_sales - damaged),
            waste_probability,
        ).astype(int)

        sale_available = np.maximum(0, available_before_sales - damaged - waste)
        units_sold = np.minimum(potential_demand, sale_available).astype(int)
        closing_stock = np.maximum(0, sale_available - units_sold).astype(int)
        stockout = (potential_demand > units_sold) & (sale_available <= potential_demand)
        hours_out = np.where(
            stockout,
            np.round(RNG.uniform(1.0, 10.5, n_pairs), 1),
            0.0,
        )
        stock_cover = np.round(closing_stock / np.maximum(rolling_7, 1), 2)
        lost_sales = np.maximum(0, potential_demand - units_sold).astype(int)

        revenue = np.round(units_sold * actual_price, 2)
        cogs = np.round(units_sold * pair_cost, 2)
        retailer_promo_cost = np.round(
            promo_flag * (promo_event_cost / promo_duration) * (1 - vendor_funded / 100),
            2,
        )
        gross_profit = np.round(revenue - cogs - retailer_promo_cost, 2)
        gross_margin = np.where(revenue > 0, np.round(gross_profit / revenue * 100, 2), 0.0)

        transaction_count = np.maximum(
            0,
            np.ceil(units_sold / RNG.uniform(1.05, 2.8, n_pairs)),
        ).astype(int)
        average_basket = np.where(
            transaction_count > 0,
            np.round(revenue / transaction_count * RNG.uniform(1.7, 4.0, n_pairs), 2),
            0.0,
        )

        anomaly_label = np.where(
            closure_flag,
            "Store closure",
            np.where(
                supplier_disruption,
                "Supplier disruption",
                np.where(
                    unusual_spike,
                    "Unusual demand spike",
                    np.where(
                        data_entry_error,
                        "Data-entry anomaly",
                        np.where(extreme_weather_flag, "Extreme weather", ""),
                    ),
                ),
            ),
        )

        inventory_df = pd.DataFrame(
            {
                "date": current_date,
                "store_id": pair_store_ids,
                "sku_id": pair_sku_ids,
                "opening_stock_units": opening_stock,
                "delivery_units": delivery_units,
                "delivery_scheduled_flag": delivery_scheduled,
                "delivery_delay_flag": delivery_delay,
                "damaged_units": damaged,
                "expired_or_wasted_units": waste,
                "closing_stock_units": closing_stock,
                "stockout_flag": stockout,
                "hours_out_of_stock": hours_out,
                "stock_cover_days": stock_cover,
                "reorder_point_units": reorder_point,
                "expected_lead_time_days": pair_lead_time,
            }
        )

        daily_df = pd.DataFrame(
            {
                "date": current_date,
                "store_id": pair_store_ids,
                "sku_id": pair_sku_ids,
                "regular_unit_price_gbp": pair_regular_price,
                "actual_unit_price_gbp": actual_price,
                "discount_pct": discount,
                "promo_flag": promo_flag,
                "promo_type": promo_type,
                "display_support_flag": display_flag,
                "email_or_app_support_flag": email_flag,
                "leaflet_support_flag": leaflet_flag,
                "competitor_price_index": np.round(
                    np.clip(
                        1.0
                        + 0.015 * (competition[pair_store_idx] - 5)
                        + RNG.normal(0, 0.035, n_pairs),
                        0.82,
                        1.22,
                    ),
                    3,
                ),
                "store_footfall": store_footfall,
                "local_event_flag": local_event_flag,
                "holiday_flag": cal["is_bank_holiday"],
                "weather_condition": cal["weather_condition"],
                "temperature_celsius": cal["temperature_celsius"],
                "rainfall_mm": cal["rainfall_mm"],
                "opening_stock_units": opening_stock,
                "delivery_units": delivery_units,
                "closing_stock_units": closing_stock,
                "stockout_flag": stockout,
                "hours_out_of_stock": hours_out,
                "lag_1_units_sold": lag_1,
                "lag_7_units_sold": lag_7,
                "rolling_7_day_avg_units_sold": np.round(rolling_7, 2),
                "rolling_28_day_avg_units_sold": np.round(rolling_28, 2),
                "day_of_week": cal["day_of_week"],
                "month": cal["month"],
                "week_of_year": cal["week_of_year"],
                "is_weekend": cal["is_weekend"],
                "is_payday_window": cal["is_payday_window"],
                "is_school_holiday": cal["is_school_holiday"],
                "is_bank_holiday": cal["is_bank_holiday"],
                "units_sold": units_sold,
                "potential_demand_units": potential_demand,
                "sales_revenue_gbp": revenue,
                "cost_of_goods_sold_gbp": cogs,
                # Exposed so the gross-profit identity is externally checkable and
                # so Phase 6's promotion budget optimiser has the real cost base.
                "retailer_promo_cost_gbp": retailer_promo_cost,
                "gross_profit_gbp": gross_profit,
                "gross_margin_pct": gross_margin,
                "waste_units": waste,
                "lost_sales_estimate_units": lost_sales,
                "transaction_count": transaction_count,
                "average_basket_value_gbp": average_basket,
                "anomaly_flag": anomaly_flag,
                "anomaly_type": anomaly_label,
            }
        )

        inventory_chunks.append(inventory_df)
        daily_chunks.append(daily_df)
        inventory_columns = inventory_df.columns.tolist()
        daily_columns = daily_df.columns.tolist()

        metrics["rows"] += n_pairs
        metrics["stockouts"] += int(stockout.sum())
        metrics["promo_rows"] += int(promo_flag.sum())
        metrics["revenue"] += float(revenue.sum())
        metrics["gross_profit"] += float(gross_profit.sum())
        metrics["promotion_distribution"].update(promo_type[promo_flag].tolist())
        revenue_by_category = daily_df.groupby(pair_categories)["sales_revenue_gbp"].sum()
        for category, category_revenue in revenue_by_category.items():
            metrics["category_sales"][category] += float(category_revenue)

        if (day_idx + 1) % CHUNK_DAYS == 0 or day_idx == len(calendar_df) - 1:
            inv_chunk = pd.concat(inventory_chunks, ignore_index=True)
            daily_chunk = pd.concat(daily_chunks, ignore_index=True)
            write_csv(
                inv_chunk,
                "fact_inventory_delivery.csv",
                mode="a" if inventory_path.exists() else "w",
            )
            write_csv(
                daily_chunk,
                "fact_daily_store_sku.csv",
                mode="a" if daily_path.exists() else "w",
            )
            inventory_chunks.clear()
            daily_chunks.clear()
            LOGGER.info(
                "Processed %s / %s days (%s fact rows).",
                day_idx + 1,
                len(calendar_df),
                f"{metrics['rows']:,}",
            )

        sales_history.append(units_sold.copy())
        if len(sales_history) > 28:
            sales_history.pop(0)
        lag_1 = units_sold.copy()
        lag_7 = sales_history[-7].copy() if len(sales_history) >= 7 else units_sold.copy()
        rolling_7 = np.mean(sales_history[-7:], axis=0)
        rolling_28 = np.mean(sales_history, axis=0)
        demand_memory = 0.65 * demand_memory + 0.35 * potential_demand
        opening_stock = closing_stock.copy()

    metrics["inventory_columns"] = inventory_columns
    metrics["daily_columns"] = daily_columns
    # True ATT on latent demand, per SKU, averaged over that SKU's treated rows.
    # NaN where a SKU was never promoted (the never-treated control pool).
    with np.errstate(invalid="ignore", divide="ignore"):
        metrics["true_att_by_sku"] = pd.Series(
            np.where(att_effect_count > 0, att_effect_sum / att_effect_count, np.nan) * 100,
            index=sku_ids,
        )
    metrics["true_att_overall_pct"] = float(
        att_effect_sum.sum() / max(att_effect_count.sum(), 1) * 100
    )
    return metrics


# =============================================================================
# DATA DICTIONARY, README AND VALIDATION
# =============================================================================

def create_data_dictionary(table_columns: Dict[str, List[str]]) -> pd.DataFrame:
    descriptions = {
        "date": "Calendar date at the table grain.",
        "store_id": "Unique Northstar store identifier.",
        "sku_id": "Unique stock keeping unit identifier.",
        "units_sold": "Observed units sold after inventory constraints.",
        "potential_demand_units": (
            "Simulated latent demand before stock constraints; validation only."
        ),
        "promo_flag": "Whether an observed promotion was active on the date.",
        "stockout_flag": "Whether demand exceeded saleable available stock.",
        "gross_profit_gbp": "Revenue less cost of goods sold and retailer-funded promotion cost.",
        "anomaly_flag": "Synthetic anomalous event indicator for data-quality exploration.",
    }
    unsafe = {
        "potential_demand_units",
        "lost_sales_estimate_units",
        "anomaly_flag",
        "anomaly_type",
    }
    rows = []

    for table_name, columns in table_columns.items():
        if table_name.startswith("dim_"):
            grain = "One row per entity"
        elif table_name == "fact_promotions.csv":
            grain = "One row per promotion event"
        elif table_name == "ground_truth_simulation_parameters.csv":
            grain = "One row per SKU"
        else:
            grain = "One row per date × store_id × sku_id"

        for column in columns:
            safe = column not in unsafe and table_name != "ground_truth_simulation_parameters.csv"
            rows.append(
                {
                    "table_name": table_name,
                    "column_name": column,
                    "data_type": "Synthetic CSV field; infer with pandas",
                    "description": descriptions.get(
                        column, f"Synthetic {column.replace('_', ' ')} field."
                    ),
                    "grain": grain,
                    "whether_safe_for_model_training": safe,
                    "notes": (
                        "Do not use simulated ground truth or post-outcome fields "
                        "as model features."
                        if column in unsafe
                        or table_name == "ground_truth_simulation_parameters.csv"
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def create_readme(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar_df: pd.DataFrame,
    metrics: Dict[str, Any],
) -> str:
    return f"""# PromoPulse: Causal Promotion, Demand and Inventory Optimisation

## Fictional business scenario
Northstar Retail Group is a fictional UK mid-market grocery and convenience retailer
operating City Convenience, High Street, Suburban Supermarket and Retail Park Superstore
locations across England, Scotland and Wales.

## Data model
- `dim_store.csv`: store master data.
- `dim_product.csv`: SKU and commercial-product attributes.
- `dim_calendar.csv`: daily calendar, events and synthetic weather.
- `fact_promotions.csv`: promotion-event records.
- `fact_inventory_delivery.csv`: daily inventory movement and stockout outcomes.
- `fact_daily_store_sku.csv`: primary analytical fact table at date × store × SKU grain.
- `../ground_truth/ground_truth_simulation_parameters.csv`: known simulation parameters for
  retrospective validation only. Deliberately stored outside `data/raw/` so no training
  loader can pick it up by globbing the raw directory.

Relationships:
- Daily facts link to `dim_store` through `store_id`.
- Daily facts link to `dim_product` through `sku_id`.
- Daily facts link to `dim_calendar` through `date`.
- Promotion events can be joined to the daily fact table using store, SKU and the
  promotion-date range.

## Entity counts and coverage
- Stores: {len(stores):,}
- SKUs: {len(products):,}
- Calendar days: {len(calendar_df):,}
- Daily modelling rows: {metrics["rows"]:,}
- Date range: {calendar_df["date"].min()} to {calendar_df["date"].max()}

## Synthetic demand logic
Latent demand is generated through a noisy multiplicative process combining product base
demand, store demand factor, weekday effect, seasonality, holiday and event effects,
weather, price response, promotion uplift, marketing support, autocorrelation and
stochastic variation. Demand is generated using a Gamma-Poisson mixture to create
over-dispersion similar to retail demand.

## Promotion-selection bias
Promotions are intentionally not random. Products with stronger margin, seasonal
relevance, own-label status and weakening demand momentum are more likely to receive
promotions. That momentum is a real declining trend applied to latent demand, so the
decline is visible in observed sales history and can legitimately be adjusted for by a
propensity model. Higher-footfall stores receive more promotion events. Promotions
cluster around Christmas, Easter, heatwaves, bank holidays and payday windows.

## Staggered campaign rollout
Roughly {CAMPAIGN_SHARE:.0%} of promotion events belong to multi-store campaigns
identified by `campaign_id`. Stores join a campaign by `rollout_cohort` (assigned on
footfall rank), entering after the wave offset recorded in `campaign_wave_days`. This
produces genuine staggered adoption for difference-in-differences. The remaining events
are store-local tactical promotions. SKUs that are never promoted anywhere form a
never-treated control pool.

## Treatment-effect ground truth
`true_promo_uplift_pct` is a **structural coefficient applied per 10 percentage points of
discount**, and it compounds with the separate price-elasticity response. It is not an
average treatment effect and must not be compared directly against a DiD or PSM estimate.
Use `true_realised_att_pct`, which is the exact simulated effect on latent demand averaged
over each SKU's treated rows. Note that observed sales are stockout-censored, so an
estimate recovered from `units_sold` will sit below the latent ATT.

## Stockout censoring
`potential_demand_units` represents latent uncensored demand. `units_sold` is constrained
by available stock after deliveries, damage and waste. Therefore stockouts censor observed
sales and create `lost_sales_estimate_units`. Inventory reconciliation is:
`closing_stock = max(0, opening_stock + delivery_units - units_sold - damaged_units
- expired_or_wasted_units)`.

## Data-quality checks completed
The generator checks unique dimension keys, fact key construction, foreign-key validity,
non-negative quantities, price and cost validity, discount bounds, inventory
reconciliation, stock constraints, gross-profit reconciliation, never-treated pool
integrity, headers and blank keys. Full results are written to
`reports/data_quality_report.md`. Each check is verified by mutation testing in `tests/` —
a corrupted value must make the corresponding check fail.

## Important modelling warning
Do not use `potential_demand_units`, `lost_sales_estimate_units`, anomaly labels or
`ground_truth_simulation_parameters.csv` as predictive or causal-model features.
`ground_truth_simulation_parameters.csv` exists only to assess whether a model recovered
the known simulated truth after modelling.

## Suggested analysis tasks
- Regression analysis of price elasticity
- Difference-in-differences study of promotion effect
- Propensity-score-weighted promotion-effect study
- Time-series demand forecasting
- Stockout prediction
- Profit-maximising promotion simulation
- Replenishment recommendation
"""


def validate_data(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar_df: pd.DataFrame,
    promotions: pd.DataFrame,
    metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run automated data-quality checks and return a structured result per check.

    Two checks in the original draft could not fail and have been rebuilt:

    * The gross-profit check computed
      ``revenue - COGS - (revenue - COGS - gross_profit)``, which reduces
      algebraically to ``gross_profit``, then asserted it equalled
      ``gross_profit``. It now reconciles against the retailer-funded promotion
      cost actually carried in the fact table.
    * The inventory check built an ``expected_closing`` by subtracting a row
      *count* rather than units sold, then never used the variable, asserting only
      that closing stock was non-negative. It now joins daily ``units_sold`` on the
      full key and tests the reconciliation identity directly.

    Results are returned rather than only asserted so the caller can write
    reports/data_quality_report.md, which PROJECT_ARCHITECTURE.md §3.1 requires.
    """
    LOGGER.info("Running automated validation checks.")
    results: List[Dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            LOGGER.error("QA FAILED: %s — %s", name, detail)

    record("Unique store primary keys", stores["store_id"].is_unique)
    record("Unique SKU primary keys", products["sku_id"].is_unique)
    record("Unique calendar dates", calendar_df["date"].is_unique)
    record("Unique promotion IDs", promotions["promotion_id"].is_unique)
    record(
        "Unit cost below retail price",
        bool((products["unit_cost_gbp"] < products["regular_unit_price_gbp"]).all()),
    )
    record(
        "Promotion discount within 0-30%",
        bool(promotions["discount_pct"].between(0, 30).all()),
        f"observed range {promotions['discount_pct'].min()}-{promotions['discount_pct'].max()}",
    )
    record(
        "Promotion store foreign keys valid",
        bool(promotions["store_id"].isin(stores["store_id"]).all()),
    )
    record(
        "Promotion SKU foreign keys valid",
        bool(promotions["sku_id"].isin(products["sku_id"]).all()),
    )
    record(
        "Promoted SKUs are all promotion-eligible",
        bool(
            promotions["sku_id"]
            .isin(products.loc[products["promotion_eligible"], "sku_id"])
            .all()
        ),
        "never-treated control pool must stay untreated",
    )

    store_set = set(stores["store_id"])
    sku_set = set(products["sku_id"])
    daily_path = OUTPUT_DIR / "fact_daily_store_sku.csv"
    inventory_path = OUTPUT_DIR / "fact_inventory_delivery.csv"

    # Units sold keyed for the cross-file inventory reconciliation. Read once,
    # outside the chunk loop - the original re-read the whole daily file per chunk.
    units_sold_by_key = pd.read_csv(
        daily_path, usecols=["date", "store_id", "sku_id", "units_sold"]
    ).set_index(["date", "store_id", "sku_id"])["units_sold"]

    daily_keys = 0
    dup_daily = False
    bad_fk = bad_neg = bad_discount = bad_promo_price = bad_stock = 0
    worst_gp_diff = 0.0
    seen_daily: set[Tuple[str, str, str]] = set()

    for chunk in pd.read_csv(daily_path, chunksize=250_000):
        daily_keys += len(chunk)
        bad_fk += int((~chunk["store_id"].isin(store_set)).sum())
        bad_fk += int((~chunk["sku_id"].isin(sku_set)).sum())
        bad_fk += int(chunk[["date", "store_id", "sku_id"]].isna().any(axis=1).sum())
        bad_neg += int(
            (chunk["units_sold"] < 0).sum()
            + (chunk["sales_revenue_gbp"] < 0).sum()
            + (chunk["actual_unit_price_gbp"] < 0).sum()
        )
        bad_discount += int((~chunk["discount_pct"].between(0, 30)).sum())
        promoted = chunk["promo_flag"].astype(bool)
        bad_promo_price += int(
            (
                chunk.loc[promoted, "actual_unit_price_gbp"]
                > chunk.loc[promoted, "regular_unit_price_gbp"]
            ).sum()
        )
        bad_stock += int(
            (chunk["units_sold"] > chunk["opening_stock_units"] + chunk["delivery_units"]).sum()
        )

        # Real gross-profit identity: revenue - COGS - retailer-funded promo cost.
        expected_gp = (
            chunk["sales_revenue_gbp"]
            - chunk["cost_of_goods_sold_gbp"]
            - chunk["retailer_promo_cost_gbp"]
        )
        worst_gp_diff = max(
            worst_gp_diff, float((expected_gp - chunk["gross_profit_gbp"]).abs().max())
        )

        keys = list(zip(chunk["date"], chunk["store_id"], chunk["sku_id"]))
        if any(key in seen_daily for key in keys):
            dup_daily = True
        seen_daily.update(keys)

    record("No blank or invalid daily foreign keys", bad_fk == 0, f"{bad_fk} violations")
    record("No negative daily quantities or prices", bad_neg == 0, f"{bad_neg} violations")
    record("Daily discount within 0-30%", bad_discount == 0, f"{bad_discount} violations")
    record(
        "Promoted price never exceeds regular price",
        bad_promo_price == 0,
        f"{bad_promo_price} violations",
    )
    record("Units sold never exceed available stock", bad_stock == 0, f"{bad_stock} violations")
    record(
        "Gross profit reconciles to revenue - COGS - promo cost",
        worst_gp_diff <= 0.02,
        f"max absolute difference £{worst_gp_diff:.4f}",
    )
    record("No duplicate daily fact keys", not dup_daily)

    inv_keys = 0
    dup_inv = False
    bad_inv_neg = 0
    recon_violations = 0
    worst_recon_diff = 0
    seen_inv: set[Tuple[str, str, str]] = set()

    for chunk in pd.read_csv(inventory_path, chunksize=250_000):
        inv_keys += len(chunk)
        quantity_cols = [
            "opening_stock_units",
            "delivery_units",
            "damaged_units",
            "expired_or_wasted_units",
            "closing_stock_units",
        ]
        bad_inv_neg += int((chunk[quantity_cols] < 0).sum().sum())
        bad_inv_neg += int(chunk[["date", "store_id", "sku_id"]].isna().any(axis=1).sum())

        # Real reconciliation identity, joined on the full grain.
        sold = units_sold_by_key.reindex(
            pd.MultiIndex.from_arrays(
                [chunk["date"], chunk["store_id"], chunk["sku_id"]]
            )
        ).to_numpy()
        expected_closing = np.maximum(
            0,
            chunk["opening_stock_units"].to_numpy()
            + chunk["delivery_units"].to_numpy()
            - sold
            - chunk["damaged_units"].to_numpy()
            - chunk["expired_or_wasted_units"].to_numpy(),
        )
        diff = np.abs(expected_closing - chunk["closing_stock_units"].to_numpy())
        recon_violations += int((diff > 0).sum())
        worst_recon_diff = max(worst_recon_diff, int(np.nanmax(diff)) if len(diff) else 0)

        keys = list(zip(chunk["date"], chunk["store_id"], chunk["sku_id"]))
        if any(key in seen_inv for key in keys):
            dup_inv = True
        seen_inv.update(keys)

    record("No negative inventory quantities", bad_inv_neg == 0, f"{bad_inv_neg} violations")
    record(
        "Inventory reconciliation identity holds",
        recon_violations == 0,
        f"{recon_violations} violations, max difference {worst_recon_diff} units",
    )
    record("No duplicate inventory fact keys", not dup_inv)
    record(
        "Daily row count matches simulation",
        daily_keys == metrics["rows"],
        f"{daily_keys:,} rows",
    )
    record(
        "Inventory row count matches simulation",
        inv_keys == metrics["rows"],
        f"{inv_keys:,} rows",
    )
    record(
        "Ground truth is outside data/raw",
        not (OUTPUT_DIR / "ground_truth_simulation_parameters.csv").exists(),
        "leakage guard per §5/§7",
    )

    failed = [r for r in results if not r["passed"]]
    if failed:
        raise AssertionError(
            "Data-quality checks failed: "
            + "; ".join(f"{r['check']} ({r['detail']})" for r in failed)
        )
    LOGGER.info("All %d automated validation checks passed.", len(results))
    return results


def write_quality_report(results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Path:
    """Persist the QA outcome to reports/data_quality_report.md (§3.1)."""
    passed = sum(1 for r in results if r["passed"])
    lines = [
        "# PromoPulse — Data Quality Report",
        "",
        f"Generated from seed {SEED} in {'FULL' if FULL_MODE else 'DEV'} mode "
        f"({N_STORES} stores x {N_SKUS} SKUs, {START_DATE} to {END_DATE}).",
        "",
        f"**{passed} of {len(results)} checks passed.**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "**FAIL**"
        lines.append(f"| {result['check']} | {status} | {result['detail'] or '—'} |")

    promo_rate = metrics["promo_rows"] / max(metrics["rows"], 1)
    stockout_rate = metrics["stockouts"] / max(metrics["rows"], 1)
    lines += [
        "",
        "## Key rates",
        "",
        f"- Daily fact rows: {metrics['rows']:,}",
        f"- Promotion density: {promo_rate:.2%}",
        f"- Stockout rate: {stockout_rate:.3%}",
        f"- True average treatment effect on latent demand: "
        f"{metrics['true_att_overall_pct']:.2f}%",
        "",
        "## Modelling warning",
        "",
        "`potential_demand_units`, `lost_sales_estimate_units`, the anomaly columns and "
        "every column in `data/ground_truth/` are outcomes of the simulation, not inputs. "
        "They must never enter a model feature set. `true_promo_uplift_pct` is a structural "
        "coefficient applied per 10 percentage points of discount and compounds with the "
        "separate price response — compare causal estimates against `true_realised_att_pct` "
        "instead.",
        "",
    ]
    path = REPORTS_DIR / "data_quality_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_summary(
    metrics: Dict[str, Any], promotions: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    stockout_rate = metrics["stockouts"] / max(metrics["rows"], 1)
    promo_rate = metrics["promo_rows"] / max(metrics["rows"], 1)

    print("\n" + "=" * 78)
    print("PROMOPULSE DATASET GENERATION SUMMARY")
    print("=" * 78)
    print(f"Mode: {'FULL_MODE' if FULL_MODE else 'DEMO_MODE'}")
    print(f"Total daily fact rows: {metrics['rows']:,}")
    print(f"Date coverage: {calendar_df['date'].min()} to {calendar_df['date'].max()}")
    print(f"Promotion rate: {promo_rate:.2%}")
    print(f"Stockout rate: {stockout_rate:.2%}")
    print(f"Total sales revenue: £{metrics['revenue']:,.2f}")
    print(f"Total gross profit: £{metrics['gross_profit']:,.2f}")

    print("\nPromotion distribution:")
    for promo_type, count in sorted(metrics["promotion_distribution"].items()):
        print(f"  - {promo_type}: {count:,} daily promotional records")

    print("\nSales revenue by category:")
    for category, revenue in sorted(metrics["category_sales"].items(), key=lambda x: -x[1]):
        print(f"  - {category}: £{revenue:,.2f}")

    print("\nOutput file sizes:")
    for file_path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {file_path.name}: {file_path.stat().st_size / (1024 * 1024):,.2f} MB")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    LOGGER.info("Starting PromoPulse synthetic data generation with seed %s.", SEED)
    ensure_output_directory()

    stores = generate_stores(N_STORES)
    products = generate_products(N_SKUS)
    calendar_df = generate_calendar(START_DATE, END_DATE)
    ground_truth = generate_ground_truth(products)
    promotions = generate_promotions(stores, products, calendar_df)
    promo_lookup = build_promo_lookup(promotions)

    # Internal simulation parameters (store_demand_factor, base_demand_units,
    # demand_trend_annual, promotion_eligible) drive generation but are not
    # observable to an analyst, so they stay out of the published dimensions.
    store_dim = stores.drop(columns=["store_demand_factor"])
    product_dim = products.drop(
        columns=["base_demand_units", "demand_trend_annual", "promotion_eligible"]
    )

    write_csv(store_dim, "dim_store.csv")
    write_csv(product_dim, "dim_product.csv")
    write_csv(calendar_df, "dim_calendar.csv")
    write_csv(promotions, "fact_promotions.csv")

    metrics = generate_facts(
        stores=stores,
        products=products,
        calendar_df=calendar_df,
        ground_truth=ground_truth,
        promo_lookup=promo_lookup,
    )

    # The realised ATT is only known after simulation, so ground truth is written
    # once here - and into data/ground_truth/, never data/raw/ (§5, §7).
    ground_truth["true_realised_att_pct"] = (
        ground_truth["sku_id"].map(metrics["true_att_by_sku"]).round(3)
    )
    ground_truth_path = GROUND_TRUTH_DIR / "ground_truth_simulation_parameters.csv"
    ground_truth.to_csv(ground_truth_path, index=False)
    LOGGER.info("Ground truth written outside data/raw to %s", ground_truth_path)

    table_columns = {
        "dim_store.csv": store_dim.columns.tolist(),
        "dim_product.csv": product_dim.columns.tolist(),
        "dim_calendar.csv": calendar_df.columns.tolist(),
        "fact_promotions.csv": promotions.columns.tolist(),
        "fact_inventory_delivery.csv": metrics["inventory_columns"],
        "fact_daily_store_sku.csv": metrics["daily_columns"],
        "ground_truth_simulation_parameters.csv": ground_truth.columns.tolist(),
    }
    data_dictionary = create_data_dictionary(table_columns)
    write_csv(data_dictionary, "data_dictionary.csv")

    readme = create_readme(stores, products, calendar_df, metrics)
    (OUTPUT_DIR / "README_DATA_GENERATION.md").write_text(readme, encoding="utf-8")

    results = validate_data(stores, products, calendar_df, promotions, metrics)
    report_path = write_quality_report(results, metrics)
    LOGGER.info("Data quality report written to %s", report_path)
    print_summary(metrics, promotions, calendar_df)

    print("\nExpected folder structure:")
    print("data/")
    print("└── raw/")
    for file_path in sorted(OUTPUT_DIR.iterdir()):
        print(f"    ├── {file_path.name}")

    print("\nRun command:")
    print("python generate_retail_dataset.py")


if __name__ == "__main__":
    main()
