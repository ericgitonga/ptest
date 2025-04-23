from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import json
import os


# Pydantic Models
class EnvironmentConfig(BaseModel):
    """
    Configuration model loaded from environment variables.

    This class represents the configuration settings for the nest monitoring report,
    including connection details, date ranges, and column specifications.

    Attributes:
        output_dir (str): Directory path where outputs will be stored
        conservancies (Dict[str, str]): Mapping of conservancy codes to full names
        start_date (Optional[datetime]): Start date for data filtering
        end_date (Optional[datetime]): End date for data filtering
        er_server (str): EarthRanger server URL
        er_username (Optional[str]): EarthRanger username for authentication
        er_password (Optional[str]): EarthRanger password for authentication
        er_nest_id_event (str): Event type identifier for nest ID events
        er_nest_check_event (str): Event type identifier for nest check events
        nest_id_columns (List[str]): Column names to include from nest ID data
        nc_columns (List[str]): Column names to include from normalized data
        nest_check_columns (List[str]): Column names for nest check processing
        inactive_columns (List[str]): Column names for inactive nest reporting
        success_fail_columns (List[str]): Column names for success/fail reporting
        in_progress_columns (List[str]): Column names for in-progress nest reporting
    """

    output_dir: str = Field(default="Ecoscope-Outputs/KBoPT/")
    conservancies: Dict[str, str] = Field(default_factory=dict)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    er_server: str = Field(default="https://kbopt.pamdas.org")
    er_username: Optional[str] = None
    er_password: Optional[str] = None
    er_nest_id_event: str
    er_nest_check_event: str = Field(default="nest_check")

    # Column specifications
    nest_id_columns: List[str] = Field(default_factory=list)
    nc_columns: List[str] = Field(default_factory=list)
    nest_check_columns: List[str] = Field(default_factory=list)
    inactive_columns: List[str] = Field(default_factory=list)
    success_fail_columns: List[str] = Field(default_factory=list)
    in_progress_columns: List[str] = Field(default_factory=list)

    @classmethod
    def from_env(cls):
        """
        Create an EnvironmentConfig instance from environment variables.

        This method loads environment variables from .env files if available,
        and parses JSON and date string values to create a properly configured instance.

        Returns:
            EnvironmentConfig: Fully configured instance with settings from environment

        Raises:
            json.JSONDecodeError: If JSON environment variables are malformed
        """
        load_env_files()

        # Parse JSON environment variables
        try:
            conservancies = json.loads(os.getenv("KB_CONSERVANCIES", "{}"))
            nest_id_cols = json.loads(os.getenv("NEST_ID_COLUMNS", "[]"))
            nc_cols = json.loads(os.getenv("NC_COLS", "[]"))
            nest_check_cols = json.loads(os.getenv("NEST_CHECK_COLUMNS", "[]"))
            inactive_cols = json.loads(os.getenv("INACTIVE_COLUMNS", "[]"))
            success_fail_cols = json.loads(os.getenv("SUCCESS_FAIL_COLUMNS", "[]"))
            in_progress_cols = json.loads(os.getenv("IN_PROGRESS_COLUMNS", "[]"))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON environment variables: {e}")
            raise

        # Parse date strings
        start_date_str = os.getenv("START")
        end_date_str = os.getenv("END")

        start_date = pd.to_datetime(start_date_str) if start_date_str and start_date_str.lower() != "none" else None
        end_date = pd.to_datetime(end_date_str) if end_date_str and end_date_str.lower() != "none" else None

        return cls(
            output_dir=os.getenv("OUTPUT_DIR", "Ecoscope-Outputs/KBoPT/"),
            conservancies=conservancies,
            start_date=start_date,
            end_date=end_date,
            er_server=os.getenv("ER_SERVER"),
            er_username=os.getenv("ER_USERNAME"),
            er_password=os.getenv("ER_PASSWORD"),
            er_nest_id_event=os.getenv("ER_NEST_ID_EVENT", ""),
            er_nest_check_event=os.getenv("ER_NEST_CHECK_EVENT", "nest_check"),
            nest_id_columns=nest_id_cols,
            nc_columns=nc_cols,
            nest_check_columns=nest_check_cols,
            inactive_columns=inactive_cols,
            success_fail_columns=success_fail_cols,
            in_progress_columns=in_progress_cols,
        )


class NestCheckResult(BaseModel):
    """
    Results model for nest check processing.

    This class holds the processed outputs of nest check data categorized
    into inactive nests and in-progress (active) nests.

    Attributes:
        inactives (Any): DataFrame containing inactive nest records
        in_progress (Any): DataFrame containing in-progress nest records
    """

    inactives: Any = Field(default_factory=lambda: pd.DataFrame())
    in_progress: Any = Field(default_factory=lambda: pd.DataFrame())

    class Config:
        arbitrary_types_allowed = True


class SpeciesResult(BaseModel):
    """
    Results model for species-specific processing.

    This class holds the processed and sorted data for a specific species.

    Attributes:
        in_progress_sorted (Any): Sorted DataFrame containing in-progress nests for a species
    """

    in_progress_sorted: Any = Field(default_factory=lambda: pd.DataFrame())

    class Config:
        arbitrary_types_allowed = True


class ProcessingContext(BaseModel):
    """
    Context for processing nest monitoring data.

    This class encapsulates the configuration, logging, and I/O components
    needed throughout the data processing workflow.

    Attributes:
        config (EnvironmentConfig): Configuration settings
        logger (Optional[Any]): Logger instance for recording execution information
        store_dir (Path): Directory path where outputs will be stored
        er_io (Optional[Any]): EarthRanger I/O client for data retrieval
    """

    config: EnvironmentConfig
    logger: Optional[Any] = None
    store_dir: Path = Field(default=Path("output"))
    er_io: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True


def load_env_files():
    """
    Load environment variables from .env files.

    This function attempts to find and load environment variables from a .env file
    in the current working directory. If not found, it falls back to checking for
    a paste.txt file in the script directory.

    Returns:
        None
    """
    # Try to find .env file
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
    else:
        # Try paste.txt as fallback
        paste_path = Path(os.path.dirname(os.path.abspath(__file__))) / "paste.txt"
        if paste_path.exists():
            load_dotenv(paste_path)


def setup_logging(context: ProcessingContext, conservancy: str) -> logging.Logger:
    """
    Set up logging to both console and file.
    This function configures a logger with handlers for both console and file output.
    The log file is named based on the conservancy and date range, and is stored
    in a logs directory within the configured output directory.
    Args:
        context (ProcessingContext): The processing context containing configuration
        conservancy (str): The name of the conservancy for log file naming
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log filename based on conservancy and date range
    since = context.config.start_date
    until = context.config.end_date
    date_range = f"{since.strftime('%Y%m%d')}_{until.strftime('%Y%m%d')}"
    log_filename = f"{conservancy}_{date_range}.log"
    # Create log directory
    log_dir = os.path.join(context.config.output_dir, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Configure logger
    logger = logging.getLogger("nest_monitoring")
    logger.setLevel(logging.DEBUG)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)  # More detailed in the file
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


def process_nest_checks(
    group: pd.DataFrame,
    status_column: str = "status",
    condition_column: str = "condition",
    inactive_cols: Optional[List[str]] = None,
    success_fail_cols: Optional[List[str]] = None,
    in_progress_cols: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> NestCheckResult:
    """
    Process nest checks based on status and condition.

    This function processes a group of nest check data (typically for a single species)
    and categorizes records based on their status and condition. It returns separate
    DataFrames for inactive nests and in-progress/success/fail nests.

    Args:
        group (pd.DataFrame): DataFrame containing nest check records to process
        status_column (str, optional): Column name for nest status. Defaults to "status".
        condition_column (str, optional): Column name for nest condition. Defaults to "condition".
        inactive_cols (Optional[List[str]], optional): Columns to include in inactive result. Defaults to None.
        success_fail_cols (Optional[List[str]], optional): Columns to include in success/fail result. Defaults to None.
        in_progress_cols (Optional[List[str]], optional): Columns to include in in-progress result. Defaults to None.
        logger (Optional[logging.Logger], optional): Logger for recording processing information. Defaults to None.

    Returns:
        NestCheckResult: Object containing separate DataFrames for inactive and in-progress nests
    """
    # For Inactive status
    inactive_mask = group[status_column] == "Inactive"
    condition_mask = (group[condition_column] == "Fallen") | (group[condition_column] == "Destroyed")
    inactives = group[inactive_mask | condition_mask][inactive_cols] if inactive_cols else pd.DataFrame()

    # For Success, Fail, Nest_building or Nest_lining status
    return_as_is = ["Success", "Failed", "Nest_building", "Nest_lining"]
    success_fail_nest_build = (
        group[group[status_column].isin(return_as_is)][success_fail_cols] if success_fail_cols else pd.DataFrame()
    )

    # For In_progress status
    in_progress = group[group[status_column] == "In_progress"][in_progress_cols] if in_progress_cols else pd.DataFrame()

    def is_castable_to_int(value):
        try:
            int(float(value))
            return True
        except (ValueError, TypeError):
            return False

    if not in_progress.empty:
        in_progress = in_progress.copy()  # Create a copy to avoid SettingWithCopyWarning
        for index, row in in_progress.iterrows():
            if is_castable_to_int(row["number_of_eggs"]) or row["incubating"] == "Presumed":
                in_progress.loc[index, status_column] = "Egg"
            if is_castable_to_int(row["number_of_chicks"]):
                in_progress.loc[index, status_column] = "Chick"
            if row["status"] == "Nest_lined":
                in_progress.loc[index, "status"] = "Nest_building"

        in_progress = in_progress[success_fail_cols] if success_fail_cols else pd.DataFrame()

    # Combine the dataframes
    combined_dfs = []
    if not success_fail_nest_build.empty:
        combined_dfs.append(success_fail_nest_build)
    if not in_progress.empty:
        combined_dfs.append(in_progress)

    combined_df = pd.concat(combined_dfs) if combined_dfs else pd.DataFrame(columns=success_fail_cols or [])

    return NestCheckResult(inactives=inactives, in_progress=combined_df)


def process_group(group: pd.DataFrame, context: ProcessingContext) -> SpeciesResult:
    """
    Process a species group and prepare it for output.

    This function takes a DataFrame representing a single species group, processes
    the nest checks, and prepares the data for output.

    Args:
        group (pd.DataFrame): DataFrame containing nest records for a single species
        context (ProcessingContext): Processing context with configuration and logging

    Returns:
        SpeciesResult: Processed and sorted results for the species
    """
    species_name = group.name

    # Process nest checks
    summary = process_nest_checks(
        group=group,
        status_column="status",
        condition_column="condition",
        inactive_cols=context.config.nest_check_columns,
        success_fail_cols=context.config.success_fail_columns,
        in_progress_cols=context.config.in_progress_columns,
        logger=context.logger,
    )

    # Extract and process in_progress from summary
    species_in_progress = summary.in_progress

    if species_in_progress.empty:
        return SpeciesResult(in_progress_sorted=pd.DataFrame(columns=context.config.success_fail_columns))

    species_in_progress.sort_values(by=["nest_id", "date"], inplace=True)
    species_in_progress.drop_duplicates(subset="nest_id", keep="last", inplace=True)
    species_in_progress["species"] = species_name.replace("_", " ").title()

    # Sort the final result
    species_in_progress_sorted = species_in_progress.sort_values(by=["nest_id", "date"])

    return SpeciesResult(in_progress_sorted=species_in_progress_sorted)


def process_multiple_species(
    df: pd.DataFrame, species_list: List[str], context: ProcessingContext
) -> Dict[str, SpeciesResult]:
    """
    Process multiple species from the dataframe.

    This function iterates through a list of species and processes the nest data
    for each one separately, returning a dictionary of results keyed by species name.

    Args:
        df (pd.DataFrame): DataFrame containing nest records for multiple species
        species_list (List[str]): List of species names to process
        context (ProcessingContext): Processing context with configuration and logging

    Returns:
        Dict[str, SpeciesResult]: Dictionary mapping species names to their processed results
    """
    # Filter the dataframe for the specified species and columns
    species_data = df[df["species"].isin(species_list)][context.config.nest_check_columns]

    # Apply the processing function to each species group
    results = {}
    for species in species_list:
        species_group = species_data[species_data["species"] == species]
        if not species_group.empty:
            # Create a mock pd.Series with name attribute for compatibility
            species_group.name = species
            try:
                results[species] = process_group(species_group, context)
            except Exception as e:
                if context.logger:
                    context.logger.error(f"Error processing species {species}: {e}")
                # Create default empty result with expected structure
                results[species] = SpeciesResult(
                    in_progress_sorted=pd.DataFrame(columns=context.config.success_fail_columns)
                )
        else:
            # Create default empty result with expected structure
            results[species] = SpeciesResult(
                in_progress_sorted=pd.DataFrame(columns=context.config.success_fail_columns)
            )

    return results


# Compile totals
def long_to_wide_with_totals(df, category_col, value_col, count_col):
    """
    Convert a long format DataFrame to wide format with totals.

    Parameters:
    df (pd.DataFrame): Input DataFrame in long format
    category_col (str): Name of the column containing categories (e.g., 'species')
    value_col (str): Name of the column containing values to be pivoted (e.g., 'status')
    count_col (str): Name of the column containing counts

    Returns:
    pd.DataFrame: Wide format DataFrame with totals
    """

    # Create long format DataFrame with value counts
    long = pd.DataFrame(df[[category_col, value_col]].value_counts()).reset_index()
    long.columns = [category_col, value_col, count_col]

    # Pivot to wide format
    wide = long.pivot(index=category_col, columns=value_col, values=count_col).fillna(0).reset_index()

    # Convert float columns to int
    wide = wide.astype({col: "int" for col in wide.select_dtypes(include=["float64"]).columns})

    wide_cols = list(wide.columns)[1:]
    checks_cols = ["Nest_building", "Egg", "Chick", "Success", "Failed"]
    diff = list(set(checks_cols) - set(wide_cols))
    if len(diff) > 0:
        for item in diff:
            wide[item] = 0

    # Separate category column
    category_series = wide[category_col]
    wide.drop(category_col, axis="columns", inplace=True)

    # Calculate totals
    column_totals = wide.sum()
    wide["Total"] = wide.sum(axis="columns")
    column_totals["Total"] = column_totals.sum()
    wide.loc["Total"] = column_totals

    # Combine category column with the rest of the DataFrame
    result = pd.concat([category_series, wide], axis="columns").fillna("Total")

    return result
