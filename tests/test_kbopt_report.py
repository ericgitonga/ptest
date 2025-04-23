import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "KBOPT", "nest_monitoring_report.py")


def test_nest_monitoring_report(tmp_path):
    """
    Test the nest monitoring report generation.

    This test verifies that:
    1. The script can be found and loaded
    2. The data can be properly accessed
    3. The output file is generated with all required sheets
    4. Each sheet has the correct column structure

    The test provides "nc" as the conservancy code when prompted via stdin.
    """
    # Set up environment variables
    env = os.environ.copy()  # This copies existing env vars including secrets

    # Add test-specific environment variables
    env["OUTPUT_DIR"] = str(tmp_path)
    env["START"] = "2024-01-01T00:00:00Z"
    env["END"] = "2024-12-31T11:59:59Z"
    env["ER_NEST_ID_EVENT"] = "nest_id"
    env["ER_NEST_CHECK_EVENT"] = "nest_check"
    env["KB_CONSERVANCIES"] = json.dumps({"nc": "NC_"})
    env["INPUT_DATA"] = "nc\n"

    env["NC_COLS"] = json.dumps(
        [
            "nest_id",
            "observer",
            "date",
            "time",
            "latitude",
            "longitude",
            "method",
            "status",
            "species",
            "condition",
            "incubating",
            "nest_outcome",
            "adult_id_male",
            "picture_taken",
            "age_of_chick_1",
            "age_of_chick_2",
            "age_of_chick_3",
            "age_of_chick_4",
            "number_of_eggs",
            "adult_id_female",
            "number_of_chicks",
            "number_of_fledglings",
            "eggs_chicks_fledglings",
            "number_of_adults_present",
            "breeding_attempt",
            "comments",
            "serial_number",
        ]
    )

    env["NEST_ID_COLUMNS"] = json.dumps(["nest_id", "serial_number", "nest_location"])
    env["NEST_CHECK_COLUMNS"] = json.dumps(["nest_id", "species", "observer", "date", "time", "status", "condition"])
    env["SUCCESS_FAIL_COLUMNS"] = json.dumps(
        [
            "nest_id",
            "date",
            "species",
            "status",
            "latitude",
            "longitude",
            "altitude",
            "habitat",
            "tree_species",
            "tree_or_cliff_height",
            "height",
            "position",
            "condition",
        ]
    )
    env["IN_PROGRESS_COLUMNS"] = json.dumps(
        [
            "nest_id",
            "date",
            "species",
            "status",
            "latitude",
            "longitude",
            "altitude",
            "habitat",
            "tree_species",
            "tree_or_cliff_height",
            "height",
            "position",
            "condition",
            "incubating",
            "number_of_eggs",
            "number_of_chicks",
        ]
    )

    # Define expected sheets
    env["REQUIRED_SHEETS"] = json.dumps(
        [
            "totals",
            "inactives",
            "vulture-white-backed",
            "vulture-lappet-faced",
            "bateleur",
            "eagle-tawny",
            "eagle-owl-verreaux's",
        ]
    )

    # Define expected columns for each type of sheet
    env["TOTALS_COLUMNS"] = json.dumps(["Species", "Nest_building", "Egg", "Chick", "Success", "Fail", "Total"])
    env["INACTIVE_COLUMNS"] = json.dumps(
        ["nest_id", "observer", "nest_location", "date", "time", "latitude", "longitude"]
    )
    env["SUCCESS_FAIL_COLUMNS"] = json.dumps(
        [
            "nest_id",
            "date",
            "species",
            "status",
            "latitude",
            "longitude",
            "altitude",
            "habitat",
            "tree_species",
            "tree_or_cliff_height",
            "height",
            "position",
            "condition",
        ]
    )

    # Run the script with input data provided via stdin
    process = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)], input=env["INPUT_DATA"], text=True, capture_output=True, env=env
    )

    # Check if the script executed successfully
    assert process.returncode == 0, f"Script failed with error: {process.stderr}"

    # Check if output file was created
    expected_output_file = os.path.join(
        tmp_path, "Analysis Output", "2024-01-01 to 2024-12-31", "NC_nest_monitoring_report.xlsx"
    )
    assert os.path.exists(expected_output_file), f"Output file not found at {expected_output_file}"

    # Check if the Excel file has all required sheets
    try:
        excel_file = pd.ExcelFile(expected_output_file)
        sheet_names = excel_file.sheet_names

        # Check for required sheets
        required_sheets = json.loads(env["REQUIRED_SHEETS"])

        for sheet in required_sheets:
            assert sheet in sheet_names, f"Required sheet '{sheet}' not found in output file"

        # Get expected columns from environment variables
        totals_expected_columns = json.loads(env["TOTALS_COLUMNS"])
        inactives_expected_columns = json.loads(env["INACTIVE_COLUMNS"])
        species_expected_columns = json.loads(env["SUCCESS_FAIL_COLUMNS"])

        # Validate totals sheet structure
        totals_df = pd.read_excel(expected_output_file, sheet_name="totals")
        for column in totals_expected_columns:
            assert column in totals_df.columns, f"Expected column '{column}' not found in totals sheet"

        # Validate inactives sheet structure
        inactives_df = pd.read_excel(expected_output_file, sheet_name="inactives")
        for column in inactives_expected_columns:
            assert column in inactives_df.columns, f"Expected column '{column}' not found in inactives sheet"

        # Validate each species sheet structure
        species_sheets = [sheet for sheet in required_sheets if sheet not in ["totals", "inactives"]]
        for sheet in species_sheets:
            species_df = pd.read_excel(expected_output_file, sheet_name=sheet)
            for column in species_expected_columns:
                assert column in species_df.columns, f"Expected column '{column}' not found in '{sheet}' sheet"

    except Exception as e:
        assert False, f"Error validating Excel file: {str(e)}"
