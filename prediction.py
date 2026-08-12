import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FOLDER = PROJECT_ROOT / "latest/model"

MODEL_PATH = MODEL_FOLDER / "model_b_floor_area_rooms.joblib"
CONFIG_PATH = MODEL_FOLDER / "model_b_floor_area_rooms_features.json"
POSTCODE_LOOKUP_PATH = MODEL_FOLDER / "postcode_lookup.csv"
HPI_LOOKUP_PATH = MODEL_FOLDER / "hpi_lookup.csv"

SCHOOL_PATH = PROJECT_ROOT / "spatial/(School)maintained_schools_wg.gpkg"
TRANSPORT_PATH = PROJECT_ROOT / "spatial/(Transport)NaPTAN_wales_feb_26.gpkg"

GREENSPACE_PATH = (
    PROJECT_ROOT
    / "spatial/OS Open Greenspace (ESRI Shape File) ST/data/ST_GreenspaceSite.shp"
)

CRIME_PATH = PROJECT_ROOT / "spatial/(Crime)dataPolice/crime.gpkg"

FLOOD_FOLDER = PROJECT_ROOT / "spatial/Flood"

RIVER_PATH = FLOOD_FOLDER / "NRW_FLOOD_RISK_FROM_RIVERS.gpkg"
SEA_PATH = FLOOD_FOLDER / "NRW_FLOOD_RISK_FROM_SEA.gpkg"
SURFACE_PATH = (
    FLOOD_FOLDER
    / "NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES.gpkg"
)

# General helpers

def check_required_files():

    required_files = [
        MODEL_PATH,
        CONFIG_PATH,
        POSTCODE_LOOKUP_PATH,
        HPI_LOOKUP_PATH,
        SCHOOL_PATH,
        TRANSPORT_PATH,
        GREENSPACE_PATH,
        CRIME_PATH,
        RIVER_PATH,
        SEA_PATH,
        SURFACE_PATH,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        print("\nRequired files are missing:\n")

        for path in missing:
            print(" -", path.relative_to(PROJECT_ROOT))

        print(
            "\nPut the missing files in the paths above "
            "before running the predictor."
        )

        sys.exit(1)


def clean_postcode(postcode):

    return (
        str(postcode)
        .upper()
        .strip()
        .replace(" ", "")
    )


def extract_postcode_district(postcode):

    match = pd.Series([postcode]).str.extract(
        r"^([A-Z]{1,2}\d[A-Z\d]?)",
        expand=False,
    )

    district = match.iloc[0]

    if pd.isna(district):
        raise ValueError(
            f"Could not extract a postcode district from '{postcode}'."
        )

    return district


def haversine(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2,
):

    earth_radius_km = 6371

    latitude_1 = np.radians(latitude_1)
    longitude_1 = np.radians(longitude_1)
    latitude_2 = np.radians(latitude_2)
    longitude_2 = np.radians(longitude_2)

    latitude_difference = latitude_2 - latitude_1
    longitude_difference = longitude_2 - longitude_1

    a = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(longitude_difference / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return earth_radius_km * c


def create_distance_band(distance_km):

    if distance_km < 2:
        return "0-2km"

    if distance_km < 5:
        return "2-5km"

    if distance_km < 10:
        return "5-10km"

    if distance_km <= 20:
        return "10-20km"

    return "20km+"


def ask_choice(prompt, allowed):

    allowed = {value.upper() for value in allowed}

    while True:
        value = input(prompt).upper().strip()

        if value in allowed:
            return value

        print(
            "Please enter one of:",
            ", ".join(sorted(allowed)),
        )


def ask_int(prompt, minimum=None, maximum=None):

    while True:
        try:
            value = int(input(prompt))

            if minimum is not None and value < minimum:
                raise ValueError

            if maximum is not None and value > maximum:
                raise ValueError

            return value

        except ValueError:
            if minimum is not None and maximum is not None:
                print(
                    f"Please enter a whole number between "
                    f"{minimum} and {maximum}."
                )
            else:
                print("Please enter a valid whole number.")


def ask_float(prompt, minimum=None):

    while True:
        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                raise ValueError

            return value

        except ValueError:
            print("Please enter a valid number.")



# Load model artifacts


def load_model_artifacts():
    prediction_model = joblib.load(MODEL_PATH)

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        model_configuration = json.load(file)

    postcode_lookup = pd.read_csv(
        POSTCODE_LOOKUP_PATH
    )

    postcode_lookup["postcode"] = (
        postcode_lookup["postcode"]
        .astype(str)
        .map(clean_postcode)
    )

    hpi_lookup = pd.read_csv(
        HPI_LOOKUP_PATH,
        parse_dates=["period"],
    )

    return (
        prediction_model,
        model_configuration,
        postcode_lookup,
        hpi_lookup,
    )



# Load spatial reference datasets


def load_spatial_data():
    target_crs = "EPSG:27700"

    school_gdf = gpd.read_file(
        SCHOOL_PATH
    )

    cardiff_school_gdf = (
        school_gdf[
            school_gdf["local_authority"] == "Cardiff"
        ][
            [
                "school_name",
                "sector",
                "school_type",
                "geometry",
            ]
        ]
        .copy()
        .to_crs(target_crs)
    )

    transport_gdf = gpd.read_file(
        TRANSPORT_PATH
    )

    transport_gdf = (
        transport_gdf[
            transport_gdf["status"] == "active"
        ][
            [
                "commonname",
                "geometry",
            ]
        ]
        .copy()
        .to_crs(target_crs)
    )

    greenspace_gdf = gpd.read_file(
        GREENSPACE_PATH
    )

    greenspace_sites = (
        greenspace_gdf[
            [
                "function",
                "distName1",
                "geometry",
            ]
        ]
        .rename(
            columns={
                "function": "greenspace_function",
                "distName1": "greenspace_name",
            }
        )
        .copy()
        .to_crs(target_crs)
    )

    crime_gdf = gpd.read_file(
        CRIME_PATH
    )

    crime_gdf["date"] = pd.to_datetime(
        crime_gdf["date"],
        errors="coerce",
    )

    crime_gdf = (
        crime_gdf
        .dropna(
            subset=[
                "date",
                "geometry",
            ]
        )
        .reset_index(drop=True)
    )

    crime_gdf["crime_id"] = crime_gdf.index

    crime_gdf["crime_month"] = (
        crime_gdf["date"]
        .dt.to_period("M")
    )

    river_flood_gdf = (
        gpd.read_file(RIVER_PATH)
        .to_crs(target_crs)
    )

    sea_flood_gdf = (
        gpd.read_file(SEA_PATH)
        .to_crs(target_crs)
    )

    surface_flood_gdf = (
        gpd.read_file(SURFACE_PATH)
        .to_crs(target_crs)
    )

    return {
        "schools": cardiff_school_gdf,
        "transport": transport_gdf,
        "greenspace": greenspace_sites,
        "crime": crime_gdf,
        "river": river_flood_gdf,
        "sea": sea_flood_gdf,
        "surface": surface_flood_gdf,
    }



# Build entered property


def collect_property_input(
    postcode_lookup,
    hpi_lookup,
):
    print("\nCARDIFF PROPERTY VALUE PREDICTOR")
    print("-" * 35)

    entered_postcode = clean_postcode(
        input("Enter postcode: ")
    )

    postcode_match = postcode_lookup.loc[
        postcode_lookup["postcode"]
        == entered_postcode
    ]

    if postcode_match.empty:
        raise ValueError(
            f"Postcode '{entered_postcode}' was not found "
            "in postcode_lookup.csv."
        )

    entered_latitude = float(
        postcode_match.iloc[0]["latitude"]
    )

    entered_longitude = float(
        postcode_match.iloc[0]["longitude"]
    )

    minimum_prediction_date = (
        hpi_lookup["period"].min()
    )

    maximum_prediction_date = (
        hpi_lookup["period"].max()
    )

    entered_year = ask_int(
        "Enter sale year: "
    )

    entered_month = ask_int(
        "Enter sale month (1-12): ",
        minimum=1,
        maximum=12,
    )

    entered_prediction_date = pd.Timestamp(
        year=entered_year,
        month=entered_month,
        day=1,
    )

    if not (
        minimum_prediction_date
        <= entered_prediction_date
        <= maximum_prediction_date
    ):
        raise ValueError(
            "Prediction date must be between "
            f"{minimum_prediction_date:%B %Y} and "
            f"{maximum_prediction_date:%B %Y}."
        )

    entered_property_type = ask_choice(
        "Property type (D/S/T/F): ",
        {"D", "S", "T", "F"},
    )

    entered_old_new = ask_choice(
        "New build (Y/N): ",
        {"Y", "N"},
    )

    entered_duration = ask_choice(
        "Duration (F/L): ",
        {"F", "L"},
    )

    entered_sale_category = ask_choice(
        "Sale category "
        "(A = standard, B = additional): ",
        {"A", "B"},
    )

    entered_total_floor_area = ask_float(
        "Total floor area in square metres: ",
        minimum=0,
    )

    entered_number_rooms = ask_int(
        "Number of habitable rooms: ",
        minimum=0,
    )

    entered_postcode_district = (
        extract_postcode_district(
            entered_postcode
        )
    )

    entered_decimal_year = (
        entered_year
        + (entered_month - 1) / 12
    )

    entered_house_location = gpd.GeoDataFrame(
        {
            "postcode": [
                entered_postcode
            ],
            "property_type": [
                entered_property_type
            ],
            "old_new": [
                entered_old_new
            ],
            "duration": [
                entered_duration
            ],
            "ppd_category_type": [
                entered_sale_category
            ],
            "postcode_district": [
                entered_postcode_district
            ],
            "decimal_year": [
                entered_decimal_year
            ],
            "latitude": [
                entered_latitude
            ],
            "longitude": [
                entered_longitude
            ],
            "total_floor_area": [
                entered_total_floor_area
            ],
            "number_habitable_rooms": [
                entered_number_rooms
            ],
        },
        geometry=gpd.points_from_xy(
            [entered_longitude],
            [entered_latitude],
        ),
        crs="EPSG:4326",
    )

    entered_house_location = (
        entered_house_location
        .to_crs("EPSG:27700")
    )

    return (
        entered_house_location,
        entered_prediction_date,
        entered_property_type,
    )



# Spatial features


def add_cardiff_centre_features(
    entered_house_location,
):
    cardiff_latitude = 51.4816
    cardiff_longitude = -3.1791

    entered_house_location[
        "distance_to_cardiff_centre_km"
    ] = haversine(
        entered_house_location["latitude"],
        entered_house_location["longitude"],
        cardiff_latitude,
        cardiff_longitude,
    )

    entered_house_location[
        "distance_band"
    ] = (
        entered_house_location[
            "distance_to_cardiff_centre_km"
        ]
        .apply(create_distance_band)
    )

    return entered_house_location


def keep_first_nearest(
    joined_gdf,
    distance_column,
):
    return (
        joined_gdf
        .sort_values(distance_column)
        .head(1)
        .drop(
            columns="index_right",
            errors="ignore",
        )
        .reset_index(drop=True)
    )


def add_nearest_features(
    entered_house_location,
    spatial_data,
):
    entered_house_location = (
        entered_house_location.drop(
            columns="index_right",
            errors="ignore",
        )
    )

    entered_house_location = gpd.sjoin_nearest(
        entered_house_location,
        spatial_data["schools"],
        how="left",
        distance_col=
            "distance_to_nearest_school_m",
    )

    entered_house_location = keep_first_nearest(
        entered_house_location,
        "distance_to_nearest_school_m",
    )

    entered_house_location = gpd.sjoin_nearest(
        entered_house_location,
        spatial_data["transport"],
        how="left",
        distance_col=
            "distance_to_nearest_transport_m",
    )

    entered_house_location = keep_first_nearest(
        entered_house_location,
        "distance_to_nearest_transport_m",
    )

    entered_house_location = gpd.sjoin_nearest(
        entered_house_location,
        spatial_data["greenspace"],
        how="left",
        distance_col=
            "nearest_greenspace_distance_m",
    )

    entered_house_location = keep_first_nearest(
        entered_house_location,
        "nearest_greenspace_distance_m",
    )

    return entered_house_location


def add_crime_feature(
    entered_house_location,
    prediction_date,
    crime_gdf,
):
    prediction_month = (
        prediction_date.to_period("M")
    )

    crime_start_month = (
        prediction_month - 12
    )

    crime_end_month = (
        prediction_month - 1
    )

    crime_during_period = crime_gdf.loc[
        crime_gdf["crime_month"].between(
            crime_start_month,
            crime_end_month,
        ),
        [
            "crime_id",
            "geometry",
        ],
    ].copy()

    if crime_during_period.empty:
        entered_house_location[
            "crime_count_1km_12m"
        ] = 0

        return entered_house_location

    crime_during_period = (
        crime_during_period.to_crs(
            entered_house_location.crs
        )
    )

    entered_house_buffer = (
        entered_house_location[
            ["geometry"]
        ].copy()
    )

    entered_house_buffer[
        "geometry"
    ] = (
        entered_house_buffer
        .geometry
        .buffer(1000)
    )

    nearby_crimes = gpd.sjoin(
        crime_during_period,
        entered_house_buffer,
        how="inner",
        predicate="within",
    )

    entered_house_location[
        "crime_count_1km_12m"
    ] = (
        nearby_crimes["crime_id"]
        .nunique()
    )

    return entered_house_location


def check_flood_risk(
    entered_house_location,
    flood_gdf,
):
    flood_join = gpd.sjoin(
        entered_house_location[
            ["geometry"]
        ].drop(
            columns="index_right",
            errors="ignore",
        ),
        flood_gdf[
            ["geometry"]
        ],
        how="left",
        predicate="intersects",
    )

    return int(
        flood_join["index_right"]
        .notna()
        .any()
    )


def add_flood_features(
    entered_house_location,
    spatial_data,
):
    river_risk = check_flood_risk(
        entered_house_location,
        spatial_data["river"],
    )

    sea_risk = check_flood_risk(
        entered_house_location,
        spatial_data["sea"],
    )

    surface_risk = check_flood_risk(
        entered_house_location,
        spatial_data["surface"],
    )

    entered_house_location[
        "any_flood_risk"
    ] = int(
        river_risk == 1
        or sea_risk == 1
        or surface_risk == 1
    )

    return (
        entered_house_location,
        river_risk,
        sea_risk,
        surface_risk,
    )



# Prediction


def predict_property_value(
    prediction_model,
    model_configuration,
    hpi_lookup,
    entered_house_location,
    entered_prediction_date,
    entered_property_type,
):
    categorical_features = (
        model_configuration[
            "categorical_features"
        ]
    )

    numeric_features = (
        model_configuration[
            "numeric_features"
        ]
    )

    required_features = (
        categorical_features
        + numeric_features
    )

    missing_features = [
        column
        for column in required_features
        if column
        not in entered_house_location.columns
    ]

    if missing_features:
        raise ValueError(
            "Required model features are missing: "
            f"{missing_features}"
        )

    input_data = (
        entered_house_location[
            required_features
        ]
        .copy()
    )

    predicted_log_price = (
        prediction_model.predict(
            input_data
        )[0]
    )

    predicted_adjusted_price = float(
        np.exp(predicted_log_price)
    )

    entered_hpi_match = hpi_lookup.loc[
        (
            hpi_lookup["period"]
            == entered_prediction_date
        )
        &
        (
            hpi_lookup["property_type"]
            == entered_property_type
        )
    ]

    if entered_hpi_match.empty:
        raise ValueError(
            "No HPI value was found for "
            f"{entered_prediction_date:%B %Y} "
            f"and property type {entered_property_type}."
        )

    entered_month_hpi = float(
        entered_hpi_match.iloc[0][
            "hpi_at_sale"
        ]
    )

    reference_hpi_lookup = (
        model_configuration[
            "reference_hpi"
        ]
    )

    property_reference_hpi = float(
        reference_hpi_lookup[
            entered_property_type
        ]
    )

    predicted_price = (
        predicted_adjusted_price
        * entered_month_hpi
        / property_reference_hpi
    )

    prediction_margin_reference = float(
        model_configuration[
            "mean_cv_mae_actual"
        ]
    )

    hpi_scale = (
        entered_month_hpi
        / property_reference_hpi
    )

    prediction_margin_entered = (
        prediction_margin_reference
        * hpi_scale
    )

    lower_price = max(
        0,
        predicted_price
        - prediction_margin_entered,
    )

    upper_price = (
        predicted_price
        + prediction_margin_entered
    )

    reference_month = pd.Timestamp(
        model_configuration[
            "reference_month"
        ]
    )

    return {
        "predicted_price": predicted_price,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "predicted_adjusted_price":
            predicted_adjusted_price,
        "reference_month": reference_month,
    }



# Output


def print_results(
    result,
    entered_house_location,
    entered_prediction_date,
    river_risk,
    sea_risk,
    surface_risk,
):
    row = entered_house_location.iloc[0]

    print("\n" + "=" * 55)
    print("CARDIFF AVM PREDICTION")
    print("=" * 55)

    print(
        f"Postcode: "
        f"{row['postcode']}"
    )

    print(
        f"Prediction month: "
        f"{entered_prediction_date:%B %Y}"
    )

    print(
        f"\nEstimated property value: "
        f"£{result['predicted_price']:,.2f}"
    )

    print(
        f"Estimated price range: "
        f"£{result['lower_price']:,.2f} "
        f"to £{result['upper_price']:,.2f}"
    )

    print(
        f"Model reference-month equivalent "
        f"({result['reference_month']:%B %Y}): "
        f"£{result['predicted_adjusted_price']:,.2f}"
    )

    print("\nSpatial features")
    print("-" * 55)

    print(
        f"Nearest school: "
        f"{row.get('school_name', 'N/A')}"
    )

    print(
        f"School distance: "
        f"{row['distance_to_nearest_school_m']:,.1f} m"
    )

    print(
        f"Nearest transport stop: "
        f"{row.get('commonname', 'N/A')}"
    )

    print(
        f"Transport distance: "
        f"{row['distance_to_nearest_transport_m']:,.1f} m"
    )

    print(
        f"Nearest greenspace: "
        f"{row.get('greenspace_name', 'N/A')}"
    )

    print(
        f"Greenspace distance: "
        f"{row['nearest_greenspace_distance_m']:,.1f} m"
    )

    print(
        f"Crime count within 1 km "
        f"(previous 12 complete months): "
        f"{int(row['crime_count_1km_12m'])}"
    )

    print(
        f"River flood risk: {river_risk}"
    )

    print(
        f"Sea flood risk: {sea_risk}"
    )

    print(
        f"Surface-water flood risk: "
        f"{surface_risk}"
    )

    print(
        f"Any flood risk: "
        f"{int(row['any_flood_risk'])}"
    )

    print("=" * 55)



# Main program


def main():
    check_required_files()

    print("Loading trained model and lookup files...")

    (
        prediction_model,
        model_configuration,
        postcode_lookup,
        hpi_lookup,
    ) = load_model_artifacts()

    print("Loading spatial datasets...")

    spatial_data = load_spatial_data()

    (
        entered_house_location,
        entered_prediction_date,
        entered_property_type,
    ) = collect_property_input(
        postcode_lookup,
        hpi_lookup,
    )

    entered_house_location = (
        add_cardiff_centre_features(
            entered_house_location
        )
    )

    entered_house_location = (
        add_nearest_features(
            entered_house_location,
            spatial_data,
        )
    )

    entered_house_location = (
        add_crime_feature(
            entered_house_location,
            entered_prediction_date,
            spatial_data["crime"],
        )
    )

    (
        entered_house_location,
        river_risk,
        sea_risk,
        surface_risk,
    ) = add_flood_features(
        entered_house_location,
        spatial_data,
    )

    result = predict_property_value(
        prediction_model,
        model_configuration,
        hpi_lookup,
        entered_house_location,
        entered_prediction_date,
        entered_property_type,
    )

    print_results(
        result,
        entered_house_location,
        entered_prediction_date,
        river_risk,
        sea_risk,
        surface_risk,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

    except Exception as error:
        print(
            f"\nPrediction failed: "
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)