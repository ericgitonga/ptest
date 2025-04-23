# -*- coding: utf-8 -*-
"""
Nest Monitoring Report Generator

This script processes nest monitoring data from EarthRanger for conservation areas.
It retrieves nest IDs and nest check events, processes them based on species and status,
and generates a consolidated Excel report with multiple sheets for conservation management.

The script uses environmental variables for configuration and connects to EarthRanger
to retrieve data, then processes it to create a single Excel file with multiple sheets.
"""

import os
import sys
import getpass
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

import ecoscope
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerConnection
from ecoscope.io.earthranger_utils import normalize_column

from utils import (
    EnvironmentConfig,
    ProcessingContext,
    setup_logging,
    process_nest_checks,
    process_multiple_species,
    long_to_wide_with_totals,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import helper as helper

# Initialize ecoscope
ecoscope.init()


def main():
    """
    Execute the nest monitoring report generation process.

    This function orchestrates the end-to-end workflow for generating nest monitoring reports:

    1. Configuration and Setup:
    2. EarthRanger Connection:
    3. Data Retrieval:
    4. Data Transformation:
    5. Species Processing:
    6. Status Classification:
    7. Report Generation:

    All reports are now consolidated into a single Excel file with multiple sheets.

    Returns:
        int: Exit code (0 for success, non-zero for error)
            0 - Process completed successfully
            1 - Error occurred during processing

    Raises:
        No exceptions are raised outside the function; all exceptions are caught
        and logged within the function.
    """
    # Set up initial console logger for startup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    startup_logger = logging.getLogger("startup")

    # Load configuration from environment
    try:
        config = EnvironmentConfig.from_env()
    except Exception as e:
        startup_logger.error(f"Error loading configuration: {e}")
        return 1

    # Get user input if needed
    if not config.output_dir:
        config.output_dir = input("Enter the output directory: ") or "Ecoscope-Outputs/KBoPT/"

    os.makedirs(config.output_dir, exist_ok=True)

    # Check conservancies
    if not config.conservancies:
        startup_logger.error("No conservancies loaded from environment, check KB_CONSERVANCIES format")
        return 1

    startup_logger.info("Conservancy code: lc mmnr mnc mt nc oc okc omc pca sc snp nca")
    conservancy_key = input("Enter the conservancy code from the list above: ")
    conservancy = config.conservancies[conservancy_key]

    # Create store_dir based on date range
    store_dir = os.path.join(
        config.output_dir,
        "Analysis Output",
        config.start_date.date().isoformat() + " to " + config.end_date.date().isoformat(),
    )
    os.makedirs(store_dir, exist_ok=True)

    # Create processing context
    context = ProcessingContext(config=config, store_dir=Path(store_dir))

    # Set up full logging with file output
    logger = setup_logging(context, conservancy)
    context.logger = logger

    # Log script start
    logger.info("=" * 80)
    logger.info(f"NEST MONITORING REPORT STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Conservancy: {conservancy[:-1]}")
    logger.info(f"Date Range: {config.start_date.isoformat()} to {config.end_date.isoformat()}")
    logger.info("=" * 80)

    # Get EarthRanger credentials if needed
    if not config.er_username:
        config.er_username = input("Enter your EarthRanger username: ")

    if not config.er_password:
        config.er_password = getpass.getpass("Please enter your ER password: ")

    logger.info(f"Connecting to EarthRanger server: {config.er_server}")
    er_io = EarthRangerConnection(
        server=config.er_server,
        username=config.er_username,
        password=config.er_password,
        tcp_limit=5,
        sub_page_size=100,
    ).get_client()

    context.er_io = er_io

    # Check for required event type
    if not config.er_nest_id_event:
        logger.error("Required environment variable ER_NEST_ID_EVENT is not set!")
        return 1

    logger.info(f"Retrieving nest IDs for event type: {config.er_nest_id_event}")
    nest_ids = helper.get_events_from_type(er_io, config.er_nest_id_event)
    logger.info(f"Retrieved {len(nest_ids)} nest ID events")

    try:
        logger.info("Retrieving event types from EarthRanger")
        event_types = er_io.get_event_types()

        logger.info(f"Looking for nest check event type: {config.er_nest_check_event}")

        nest_check_matches = event_types[event_types["value"] == config.er_nest_check_event]
        if nest_check_matches.empty:
            logger.error(f"Nest check event type '{config.er_nest_check_event}' not found in event types!")
            return 1

        nest_check = nest_check_matches["id"].values[0]
        logger.info(f"Found nest check event type with ID: {nest_check}")

        logger.info("Retrieving nest check events...")
        nest_checks = er_io.get_events(
            event_type=nest_check, since=config.start_date.isoformat(), until=config.end_date.isoformat()
        )
        logger.info(f"Successfully downloaded {len(nest_checks)} Nest Check events.")

    except Exception as e:
        logger.error(f"Nest Check events download failed: {str(e)}")
        raise

    # Transform nest_ids DataFrame
    logger.info("Normalizing nest IDs data...")
    normalize_column(nest_ids, "event_details")

    nest_ids.rename(columns={"event_details__nest_id_location": "nest_location"}, inplace=True)
    nest_ids.columns = [col.replace("event_details__", "") for col in nest_ids.columns]
    nest_ids.columns = [col.replace("nest_id_", "") for col in nest_ids.columns]

    nest_ids = nest_ids[~nest_ids["nest_id"].isna()]
    logger.info(f"After removing NaN nest_ids, {len(nest_ids)} records remain")

    nests = nest_ids[nest_ids["nest_id"].str.startswith(conservancy)][config.nest_id_columns]
    logger.info(f"Found {len(nests)} nest IDs for conservancy {conservancy}")

    # Transform nest_checks DataFrame
    logger.info("Normalizing nest checks data...")
    normalize_column(nest_checks, "event_details")

    nest_checks.columns = [col.replace("event_details__", "") for col in nest_checks.columns]
    nest_checks.columns = [col.replace("nest_check_", "") for col in nest_checks.columns]

    logger.info(f"nest_checks shape after normalizing: {nest_checks.shape}")

    # Check to see if nest_checks has an observer column. If it does not, then add the column.
    if "observer" not in nest_checks.columns:
        normalize_column(nest_checks, "reported_by")
        nest_checks.rename(columns={"reported_by__user__id": "observer_id"}, inplace=True)

        # Extract observer_ids from which we shall get names to fill in the observer column
        observer_ids = nest_checks["observer_id"].unique()
        observer_ids = observer_ids[~pd.isna(observer_ids)]

        users = pd.DataFrame(er_io.get_users())
        users = users[users["id"].isin(observer_ids)]
        users["observer"] = users["first_name"] + " " + users["last_name"]
        users.set_index("id")

        observer_map = users.set_index("id")["observer"].to_dict()
        nest_checks["observer"] = nest_checks["observer_id"].map(observer_map)

    nest_checks["latitude"] = nest_checks["geometry"].y
    nest_checks["longitude"] = nest_checks["geometry"].x

    logger.info(f"nest_checks shape after adding lat/lon: {nest_checks.shape}")

    def separate(timestamp):
        """
        Split a timestamp into separate date and time components.

        This function takes a timestamp in any format recognized by pandas and
        converts it into two separate string components: a date in ISO format
        (YYYY-MM-DD) and a time in 24-hour format (HH:MM).

        Args:
            timestamp (str, datetime, Timestamp): A timestamp value that can be
            parsed by pandas.to_datetime(). This can be a string in various
            formats, a Python datetime object, or a pandas Timestamp.

        Returns:
            tuple: A tuple containing two strings:
                - date (str): The date portion in ISO format (YYYY-MM-DD)
                - time (str): The time portion in 24-hour format (HH:MM)
                  Note that seconds are truncated from the time output.
        """
        date = pd.to_datetime(timestamp).date().isoformat()
        time = pd.to_datetime(timestamp).time().isoformat()[:5]

        return date, time

    nest_checks[["date", "time"]] = nest_checks["time"].apply(lambda x: pd.Series(separate(x)))

    logger.info(f"nest_checks shape after separating date and time: {nest_checks.shape}")

    nest_checks = nest_checks[config.nc_columns]

    logger.info(f"nest_checks shape after subsetting: {nest_checks.shape}")

    nest_checks = nest_checks.dropna(subset=["nest_id", "status", "condition"])
    logger.info(f"After dropping rows with NaN nest_id, status, or condition, {len(nest_checks)} records remain")

    # Pre-Process nest checks
    for column in ["nest_id", "status", "condition"]:
        nest_checks[column] = nest_checks[column].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Nest checks for selected conservancy
    conservancy_checks = nest_checks[nest_checks["nest_id"].str.startswith(conservancy)].copy()
    logger.info(f"Found {len(conservancy_checks)} nest checks for conservancy {conservancy}")

    conservancy_checks.rename(columns={"serial_number": "checks_serial_number"}, inplace=True)

    # Nest check for selected conservancy merged with nest_ids to get lat/lon
    nest_check = pd.merge(nests, conservancy_checks, on="nest_id", how="right")
    logger.info(f"After merging with nest IDs, nest_check has {len(nest_check)} rows")

    if nest_check.empty:
        logger.warning(f"There was no data available for {conservancy} conservancy for the period specified")
        logger.info("=" * 80)
        logger.info(
            f"NEST MONITORING REPORT COMPLETED - NO DATA FOUND - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("=" * 80)
        return 0

    # Process Active Nests
    try:
        logger.info("Retrieving species choices from EarthRanger...")
        species_df = pd.DataFrame(er_io.get_objects_multithreaded(object="choices/"))

        species_df = species_df[species_df["field"].str.contains("nest_check_species")]
        master_species_list = species_df["value"].unique()
        logger.info(f"Retrieved {len(master_species_list)} species from master list")

    except Exception as e:
        logger.error(f"Error retrieving species choices: {e}")
        master_species_list = []

    # Create species list
    species_list = [
        item
        for item in nest_check["species"].unique()
        if item in master_species_list and item not in ["Not Applicable", "Other"] and pd.notna(item)
    ]
    logger.info(f"Found {len(species_list)} species to process")

    # Check for empty species list
    if not species_list:
        logger.error("No valid species found for processing!")
        return 1

    # Process species data
    all_results = process_multiple_species(df=nest_check, species_list=species_list, context=context)

    # Concatenate results for all species
    check_dfs = []
    species_data_dict = {}  # To store individual species DataFrames

    for item in species_list:
        if item in all_results:
            df = all_results[item].in_progress_sorted
            if not df.empty:
                check_dfs.append(df)
                species_data_dict[item] = df

    if check_dfs:
        checks_all = pd.concat(check_dfs)
        logger.info(f"Successfully concatenated results with {len(checks_all)} total rows")

    else:
        logger.warning("No data to concatenate, creating empty DataFrame")
        checks_all = pd.DataFrame(columns=["species", "nest_id", "date", "status"])

    # Create totals summary
    if checks_all.empty:
        logger.warning("No active nests found. Creating empty totals dataframe.")
        checks_totals = pd.DataFrame(columns=["Species", "Nest_building", "Egg", "Chick", "Success", "Fail", "Total"])
    else:
        try:
            logger.info("Creating totals table...")

            checks_totals = long_to_wide_with_totals(
                df=checks_all, category_col="species", value_col="status", count_col="count"
            )
            logger.info(f"Totals table created with shape: {checks_totals.shape}")

            checks_totals.rename(columns={"species": "Species", "Failed": "Fail"}, inplace=True)

            expected_columns = ["Species", "Nest_building", "Egg", "Chick", "Success", "Fail", "Total"]
            for col in expected_columns:
                if col not in checks_totals.columns:
                    checks_totals[col] = 0

            checks_totals = checks_totals[expected_columns]

        except Exception as e:
            logger.error(f"Error creating totals table: {e}")
            checks_totals = pd.DataFrame(
                columns=["Species", "Nest_building", "Egg", "Chick", "Success", "Fail", "Total"]
            )

    # Process Inactive Nests
    if checks_all.empty:
        logger.info("No active nests found")
        active_nests = []
    else:
        active_nests = checks_all["nest_id"].unique()
        logger.info(f"Found {len(active_nests)} active nests")

    logger.info("Processing inactive nests...")
    inactive_nests_result = process_nest_checks(
        group=nest_check, status_column="status", condition_column="condition", inactive_cols=config.inactive_columns
    )

    inactive_nests_df = inactive_nests_result.inactives
    logger.info(f"Initial inactive nests dataframe has {len(inactive_nests_df)} rows")

    if inactive_nests_df.empty:
        logger.warning("No inactive nests found.")
        inactives = pd.DataFrame(columns=config.inactive_columns)
    else:
        inactive_nests = inactive_nests_df["nest_id"].unique()
        logger.info(f"Found {len(inactive_nests)} unique inactive nest IDs")

        inactive_nests_list = list(set(inactive_nests) - set(active_nests))
        logger.info(f"After removing active nests, {len(inactive_nests_list)} inactive nests remain")

        inactive_nests_to_keep = inactive_nests_df[inactive_nests_df["nest_id"].isin(inactive_nests_list)]
        logger.info(f"Filtered inactive nests dataframe has {len(inactive_nests_to_keep)} rows")

        inactives = (
            inactive_nests_to_keep.sort_values(by=["nest_id", "date"])
            .drop_duplicates(subset="nest_id", keep="last")
            .reset_index(drop=True)
        )
        logger.info(f"Final inactive nests dataframe has {len(inactives)} rows after deduplication")

    # Write all dataframes to a single Excel file with multiple sheets
    excel_filename = os.path.join(store_dir, f"{conservancy}nest_monitoring_report.xlsx")
    logger.info(f"Writing consolidated report to {excel_filename}")

    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        # Write totals sheet
        checks_totals.to_excel(writer, sheet_name="totals", index=False)

        # Write inactives sheet
        inactives.to_excel(writer, sheet_name="inactives", index=False)

        # Write individual species sheets
        for species_name, species_df in species_data_dict.items():
            # Format sheet name to be valid for Excel (max 31 chars, no special chars)
            sheet_name = species_name.lower().replace("_", "-")
            # Truncate if necessary
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            species_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Successfully wrote consolidated report to {excel_filename}")
    logger.info("=" * 80)
    logger.info(f"NEST MONITORING REPORT COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    main()
