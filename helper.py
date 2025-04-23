import getpass
import logging
import os
import sys
import typing
import uuid
from typing import Tuple, Union
from urllib.parse import urlparse

import ecoscope
import ecoscope.analysis
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ecoscope.analysis.seasons import seasonal_windows, std_ndvi_vals, val_cuts
from ecoscope_workflows_ext_ecoscope.tasks import analysis
from erclient.client import ERClientException
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from tqdm import tqdm

from SMART.smartio import SmartAPI, SmartAPIConfig

# from typing import Optional
# import time
# import shapely
# from datetime import datetime
# from ecoscope.mapping import EcoMap
# from lonboard.colormap import apply_categorical_cmap
# from ecoscope.analysis.feature_density import calculate_feature_density
# from typing import Optional
# import matplotlib as mpl


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def earthengine_init():
    try:
        SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
        PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE")
        if SERVICE_ACCOUNT and PRIVATE_KEY_FILE:
            credentials = ee.ServiceAccountCredentials(
                email=SERVICE_ACCOUNT,
                key_file=PRIVATE_KEY_FILE,
            )
            ee.Initialize(credentials)
        else:
            EE_PROJECT = os.getenv("EE_PROJECT")
            ee.Authenticate()
            ee.Initialize(project=EE_PROJECT)

        logger.info("Successfully connected to EarthEngine.")

    except ee.EEException as e:
        logger.error(f"Failed to connect to EarthEngine: {e}")
        sys.exit(1)


class SmartEnvironConfig(BaseModel):
    """Environment configuration for SMART API"""

    server_url: HttpUrl = Field(
        default="https://smartapitest.smartconservationtools.org/smartapi/",
        description="SMART server URL",
    )
    username: str = Field(..., description="SMART username")
    password: SecretStr = Field(..., description="SMART password")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    @classmethod
    def from_env(cls) -> "SmartEnvironConfig":
        """
        Create configuration from environment variables.
        Falls back to interactive input if environment variables are not set.
        """
        try:
            server_url = (
                os.getenv("SM_SERVER")
                or input(
                    "Enter the SMART server URL "
                    "(default: https://smartapitest.smartconservationtools.org/smartapi/): "
                )
                or "https://smartapitest.smartconservationtools.org/smartapi/"
            )
            if not server_url.endswith("/"):
                server_url += "/"

            parsed_url = urlparse(server_url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError(f"Invalid server URL format: {server_url}")

            username = os.getenv("SM_USERNAME") or input("Enter your SMART username: ")
            password = os.getenv("SM_PASSWORD") or getpass.getpass("Please enter your SM password: ")

            return cls(
                server_url=server_url,
                username=username,
                password=password,
                verify_ssl=True,
            )
        except Exception as e:
            logger.error(f"Failed to create configuration: {str(e)}")
            raise


class SmartInitializer:
    """Handles initialization of SMART API connection"""

    @staticmethod
    def initialize() -> Union[SmartAPI, None]:
        """
        Initialize SMART API connection with configuration from environment or user input.

        Returns:
            SmartAPI: Initialized SMART API instance

        Raises:
            SystemExit: If connection fails
        """
        try:
            config = SmartEnvironConfig.from_env()

            api_config = SmartAPIConfig(
                url_base=str(config.server_url),
                username=config.username,
                password=config.password.get_secret_value(),
                verify_ssl=config.verify_ssl,
            )

            smart_io = SmartAPI(api_config)
            smart_io.login()

            logger.info("Successfully connected to SMART.")
            return smart_io

        except Exception as e:
            logger.error(f"Failed to connect to SMART: {str(e)}")
            sys.exit(1)


def smart_init() -> Union[SmartAPI, None]:
    """
    Convenience function to initialize SMART API.

    Returns:
        SmartAPI: Initialized SMART API instance

    Raises:
        SystemExit: If connection fails
    """
    return SmartInitializer.initialize()


def filter_trajectory(
    traj,
    min_length_meters=0.0,
    max_length_meters=float("inf"),
    min_time_secs=0.0,
    max_time_secs=4 * 60 * 60,
    min_speed_kmhr=0.0,
    max_speed_kmhr=8.0,
):
    """filter trajectory"""
    try:
        traj_seg_filter = ecoscope.base.TrajSegFilter(
            min_length_meters=min_length_meters,
            max_length_meters=max_length_meters,
            min_time_secs=min_time_secs,
            max_time_secs=max_time_secs,
            min_speed_kmhr=min_speed_kmhr,
            max_speed_kmhr=max_speed_kmhr,
        )
        """apply filter to trajectory"""
        traj.apply_traj_filter(traj_seg_filter, inplace=True)

        """remove the filtered segment"""
        traj.remove_filtered(inplace=True)
        logger.info("Trajectory filter was successful.")

        return traj

    except Exception as e:
        logger.error(f"Failed to filter the trajectory: {e}")
        sys.exit(1)


def define_raster_profile(pixel_size=250.0, crs="ESRI:102022", nodata_value=np.nan, band_count=1):
    return ecoscope.io.raster.RasterProfile(
        pixel_size=pixel_size,
        crs=crs,
        nodata_value=nodata_value,
        band_count=band_count,
    )


def determine_season_windows(aoi, since, until):
    windows = None
    try:
        # Merge to a larger Polygon
        aoi = aoi.copy()
        aoi = aoi.to_crs(4326)
        aoi = aoi.dissolve()
        aoi = aoi.iloc[0]["geometry"]

        # Determine wet/dry seasons
        logger.info(f"Attempting download of NDVI values since: {since.isoformat()} until: {until.isoformat()}")
        date_chunks = (
            pd.date_range(start=since, end=until, periods=5, inclusive="both")
            .to_series()
            .apply(lambda x: x.isoformat())
            .values
        )
        ndvi_vals = []
        for t in range(1, len(date_chunks)):
            logger.info(f"Downloading NDVI Values from EarthEngine......({t}/5)")
            ndvi_vals.append(
                std_ndvi_vals(
                    img_coll="MODIS/061/MCD43A4",
                    nir_band="Nadir_Reflectance_Band2",
                    red_band="Nadir_Reflectance_Band1",
                    aoi=aoi,
                    start=date_chunks[t - 1],
                    end=date_chunks[t],
                )
            )
        ndvi_vals = pd.concat(ndvi_vals)

        logger.info(f"Calculating seasonal windows based on {str(len(ndvi_vals))} NDVI values....")

        # Calculate the seasonal transition point
        cuts = val_cuts(ndvi_vals, 2)

        # Determine the seasonal time windows
        windows = seasonal_windows(ndvi_vals, cuts, season_labels=["dry", "wet"])

    except Exception as e:
        logger.error(f"Failed to calculate seasonal windows {e}")

    return windows


def create_temporal_label(df=None, col_name="temporal_label", time_col="fixtime", directive="%m"):
    if directive and time_col:
        df[col_name] = df[time_col].dt.strftime(directive)
        return df


def create_seasonal_label(traj, output_dir="./", desired_percentile=99.9):
    try:
        logger.info("Calculating seasonal ETD percentiles: calculating total percentiles")
        total_percentiles = (
            traj.groupby(["groupby_col"])
            .apply(lambda df: analysis.calculate_time_density(df, pixel_size=250.0, percentiles=[desired_percentile]))
            .to_crs(traj.crs)
        )

        # Apply a seasonal label to each trajectory segment by individual based on the total_percentile area
        logger.info("Calculating seasonal ETD percentiles: applying seasonal labels to trajectory")

        def apply_seasonal_label(t):
            # logger.info(t.name)
            # aoi=total_percentiles[total_percentiles['subject_id']==t.name]
            aoi = total_percentiles.loc[t.name]
            seasonal_wins = determine_season_windows(
                aoi=aoi, since=t["segment_start"].min(), until=t["segment_end"].max()
            )

            season_bins = pd.IntervalIndex(
                data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1)
            )

            labels = seasonal_wins.season

            t["season"] = pd.cut(t["segment_start"], bins=season_bins).map(dict(zip(season_bins, labels)))
            return t

        traj = (
            traj.groupby("groupby_col")
            .apply(apply_seasonal_label, include_groups=False)
            .reset_index(level=["groupby_col"])
        )

        return traj

    except Exception as e:
        logger.error(f"Failed to apply seasonal label to trajectory {e}")


def combine_percentiles(percentiles):
    """Combine percentiles of the same level across multiple subjects"""
    try:
        combined_percentiles = gpd.GeoDataFrame(
            geometry=percentiles.groupby("percentile").apply(lambda x: x["geometry"].unary_union),
            crs=percentiles.crs,
        ).reset_index()
        combined_percentiles["area_sqkm"] = combined_percentiles.area / 1000000
        combined_percentiles.sort_values(by="percentile", ascending=False, inplace=True)
        return combined_percentiles
    except Exception as e:
        logger.error(f"Failed to combine percentiles {e}")


def fix_struct_type(df):
    if "extra__source__additional" in df.columns:
        if df["extra__source__additional"].dtype == "object":
            df["extra__source__additional"] = df["extra__source__additional"].apply(
                lambda x: {"dummy": None} if pd.isna(x) or x == {} else x
            )
    return df


def transform_df_columns(df: pd.DataFrame = None, column_map_dict: dict = None):
    """
    A function to first subset a dataframe's columns based on the keys in the supplied dictionary, then
    secondly to re-name the columns to the values provided in the dictionary. If the user doesn't supply
    a column_map_dict then just pass back the original dataframe.
    """

    if column_map_dict is not None:
        # subset the columns
        df = df[list(column_map_dict.keys())]

        # rename the columns
        df = df.rename(mapper=column_map_dict, axis=1)

    return df


def clean_geodataframe(gdf):
    return gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]


def extract_voltage(s: typing.Dict):
    """
    Extracts voltage from different source-provider in EarthRanger
    Parameters
    ----------
    s: typing.Dict

    Returns
    -------
    typing.Any

    """
    additional = s["extra__observation_details"] or {}
    voltage = additional.get("battery", None)  # savannah tracking
    if not voltage:
        voltage = additional.get("mainVoltage", None)  # vectronics
    if not voltage:
        voltage = additional.get("batt", None)  # AWT
    if not voltage:
        voltage = additional.get("power", None)  # Followit
    return voltage


def plot_collar_voltage(
    relocations,
    start_time,
    extract_fn=extract_voltage,
    output_folder=None,
    layout_kwargs=None,
    hline_kwargs=None,
):
    assigned_range = (
        relocations["extra__subjectsource__assigned_range"]
        .apply(pd.Series)
        .add_prefix("extra.extra.subjectsource__assigned_range.")
    )
    relocations = relocations.merge(assigned_range, right_index=True, left_index=True)

    groups = relocations.groupby(by=["extra__subject__id", "extra__subjectsource__id"])

    for group, dataframe in groups:
        subject_name = relocations.loc[relocations["extra__subject__id"] == group[0]]["extra__subject__name"].unique()[
            0
        ]

        dataframe["extra__subjectsource__assigned_range__upper"] = pd.to_datetime(
            dataframe["extra__subjectsource__assigned_range"].str["upper"],
            errors="coerce",
        )

        # changing to correct the issue
        subjectsource_upperbound = dataframe["extra__subjectsource__assigned_range__upper"].unique()
        is_source_active = subjectsource_upperbound >= start_time or pd.isna(subjectsource_upperbound)[0]

        if is_source_active:
            logger = logging.getLogger(__name__)
            logger.info(subject_name)

            dataframe = dataframe.sort_values(by=["fixtime"])
            dataframe["voltage"] = np.array(dataframe.apply(extract_fn, axis=1), dtype=np.float64)

            time = dataframe[dataframe.fixtime >= start_time].fixtime.tolist()
            voltage = dataframe[dataframe.fixtime >= start_time].voltage.tolist()

            # Calculate the historic voltage
            hist_voltage = dataframe[dataframe.fixtime <= start_time].voltage.tolist()
            if hist_voltage:
                volt_upper, volt_lower = np.nanpercentile(hist_voltage, [97.5, 2.5])
                hist_voltage_mean = np.nanmean(hist_voltage)
            else:
                volt_upper, volt_lower = np.nan, np.nan
                hist_voltage_mean = None
            volt_diff = volt_upper - volt_lower
            volt_upper = np.full((len(time)), volt_upper, dtype=np.float32)
            volt_lower = np.full((len(time)), volt_lower, dtype=np.float32)

            if np.all(volt_diff == 0):
                # jitter = np.random.random_sample((len(volt_upper,)))
                volt_upper = volt_upper + 0.025 * max(volt_upper)
                volt_lower = volt_lower - 0.025 * max(volt_lower)

            if not any(hist_voltage or voltage):
                continue

            try:
                lower_y = min(np.nanmin(np.array(hist_voltage)), np.nanmin(np.array(voltage)))
                upper_y = max(np.nanmax(np.array(hist_voltage)), np.nanmax(np.array(voltage)))
            except ValueError:
                lower_y = min(hist_voltage or voltage)
                upper_y = max(hist_voltage or voltage)
            finally:
                lower_y = lower_y - 0.1 * lower_y
                upper_y = upper_y + 0.1 * upper_y

            if not len(voltage):
                continue

            if not layout_kwargs:
                layout = go.Layout(
                    xaxis={"title": "Time"},
                    yaxis={"title": "Collar Voltage"},
                    margin={"l": 40, "b": 40, "t": 50, "r": 50},
                    hovermode="closest",
                )
            else:
                layout = go.Layout(**layout_kwargs)

            # Add the current voltage
            trace = go.Scatter(
                x=time,
                y=voltage,
                fill=None,
                showlegend=True,
                mode="lines",
                line={
                    "width": 1,
                    "shape": "spline",
                },
                line_color="rgb(0,0,246)",
                marker={
                    "colorscale": "Viridis",
                    "color": voltage,
                    "colorbar": dict(title="Colorbar"),
                    "cmax": np.max(voltage),
                    "cmin": np.min(voltage),
                },
                name=subject_name,
            )

            # Add the historical lower HPD value
            trace_lower = go.Scatter(
                x=time,
                y=volt_lower,
                fill=None,
                line_color="rgba(255,255,255,0)",
                mode="lines",
                showlegend=False,
            )

            # Add the historical max HPD value
            trace_upper = go.Scatter(
                x=time,
                y=volt_upper,
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                fillcolor="rgba(0,176,246,0.2)",
                line_color="rgba(255,255,255,0)",
                showlegend=True,
                name="Historic 2.5% - 97.5%",
            )

            fig = go.Figure(layout=layout)
            fig.add_trace(trace_lower)
            fig.add_trace(trace_upper)
            fig.add_trace(trace)
            if hist_voltage_mean:
                if not hline_kwargs:
                    fig.add_hline(
                        y=hist_voltage_mean,
                        line_dash="dot",
                        line_width=1.5,
                        line_color="Red",
                        annotation_text="Historic Mean",
                        annotation_position="right",
                    )
                else:
                    fig.add_hline(**hline_kwargs)
            fig.update_layout(yaxis=dict(range=[lower_y, upper_y]))
            if output_folder:
                fig.write_image(os.path.join(f"{output_folder}/_{group}_{str(uuid.uuid4())[:4]}.png"))
                fig.show()
            else:
                fig.show()


def get_date_range() -> Tuple[str, str]:
    """Retrieve and validate the date range from user input."""
    while True:
        since = os.getenv("START") or input("Enter the start date (YYYY-MM-DD): ")
        until = os.getenv("END") or input("Enter the end date (YYYY-MM-DD): ")
        try:
            since_iso = pd.Timestamp(since).isoformat()
            until_iso = pd.Timestamp(until).isoformat()
            return since_iso, until_iso
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")


def get_er_subject(
    er_io: ecoscope.io.EarthRangerIO = None,
    name: str = None,
    id: str = None,
    use_id: bool = True,
) -> pd.Series:
    """
    Download a pd.Series for a single subject in ER based either on the ID or the name of the subject
    """
    try:
        if use_id:
            # Look up subject based on matching id only
            subject = er_io.get_subjects(id=id, include_inactive=True)
        else:
            # Look up subject based on matching name only
            subject = er_io.get_subjects(name=name, include_inactive=True)
    except Exception as e:
        # handle case where there is no existing subject by creating an empty dataframe
        print(f"Subject does not exist, thus empty dataframe shall be created. {e}")
        return pd.Series()

    # test for identifiability
    if len(subject) > 1:
        raise Exception("There is more than one Subject in ER with the same name")
    if subject.empty:
        return pd.Series()
    else:
        return subject.iloc[0]


def create_or_update_subjects(
    er_io: ecoscope.io.EarthRangerIO = None,
    subjects_df: pd.DataFrame = None,
    use_id: bool = True,
    provider: str = "ecoscope",
) -> None:
    """
    This function will first check for an existing subject on the ER instance with the same
    name or id (depending on the use_id flag) as the input subject. If a subject already
    exists then it be updated. If it does not already exist then it will be created.
    """
    try:
        for _, row in subjects_df.iterrows():
            if use_id:
                subject = get_er_subject(er_io=er_io, id=row["id"], use_id=use_id)
            else:
                subject = get_er_subject(er_io=er_io, name=row["name"], use_id=use_id)

            if subject.empty:
                # create new subject or patch existing subject
                er_io.post_subject(
                    id=row["id"],
                    subject_name=row["name"],
                    subject_type=row["subject_type"],
                    subject_subtype=row["subject_subtype"],
                    additional=row["additional"],
                    is_active=row["is_active"],
                    provider=provider,  # todo: is this still necessary?
                )
                logger.info(f"subject: {row['id']} created.")
            else:
                # update existing subject
                er_io._patch(
                    f'subject/{row["id"]}',
                    payload=dict(
                        subject_name=row["name"],
                        subject_type=row["subject_type"],
                        subject_subtype=row["subject_subtype"],
                        additional=row["additional"],
                        is_active=row["is_active"],
                    ),
                )
                logger.info(f"subject: {row['id']} updated.")

    except Exception as e:
        logger.error(f"An error occurred trying to create or patch the subject: {e}")


def get_er_source(
    er_io: ecoscope.io.EarthRangerIO = None,
    manufacturer_id: str = "",
    id: str = None,
    use_id: bool = True,
) -> pd.Series:
    """
    Download a pd.Series for a single source in ER based either on the ID or the name of the source
    """
    try:
        if use_id:
            # Look up source based on matching id only
            source = er_io.get_sources(id=id)  # TODO: this is passing back all sources
        else:
            # source = pd.DataFrame(er_io.get_source_by_manufacturer_id(manufacturer_id)) #TODO: this is not working
            source = er_io.get_sources(manufacturer_id=manufacturer_id)  # TODO: this is passing back all sources
            if not source.empty:
                source = source[source["manufacturer_id"] == manufacturer_id]

    except Exception as e:
        # handle case where there is no existing source by creating an empty dataframe
        print(f"Source does not exist, thus empty dataframe shall be created. {e}")
        return pd.Series()

    # test for identifiability
    if len(source) > 1:
        raise Exception("There is more than one Source in ER with the same manufacturer_id")
    if source.empty:
        return pd.Series()
    else:
        return source.iloc[0]


def create_or_update_sources(
    er_io: ecoscope.io.EarthRangerIO = None,
    sources_df: pd.DataFrame = None,
    use_id: bool = True,
) -> pd.DataFrame:
    """
    Create or update a new source in the database and reflect those changes in the dataframe.
    Note that the ER API doesn't preserve the ID value and will generate a new one unlike the events, subjects APIs
    """
    try:
        # Iterate over rows of the dataframe
        for idx, row in sources_df.iterrows():
            # Try to fetch the source using the given ID or manufacturer ID
            if use_id:
                source = get_er_source(er_io=er_io, id=row["id"], use_id=use_id)
            else:
                source = get_er_source(er_io=er_io, manufacturer_id=row["manufacturer_id"], use_id=use_id)

            # If the source doesn't exist, create a new one
            if source.empty:
                source_response = er_io.post_source(
                    id=row["id"],  # TODO: this doesn't work - can't set the ID value like we do with other tables in ER
                    manufacturer_id=row["manufacturer_id"],
                    source_type=row["source_type"],
                    model_name=row["model_name"],
                    additional=row["additional"],
                )

                sources_df.at[idx, "id"] = source_response["id"].values[0]

                logger.info(f"Source {row['id']} created!")  # TODO: this isn't correct

            else:
                # If the source exists, update only the additional info
                source_response = er_io._patch(
                    f'source/{source["id"]}',
                    payload=dict(
                        # manufacturer_id=row['manufacturer_id'],
                        # source_type=row['source_type'],
                        # model_name=row['model_name'],
                        additional=row["additional"],
                    ),
                )
                sources_df.at[idx, "id"] = source["id"]
                logger.info(f"source: {source['id']} updated.")

        # Return the created dataframe
        return sources_df

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e


def create_or_update_subjectsources(
    er_io: ecoscope.io.EarthRangerIO = None,
    subject_sources_df: pd.DataFrame = None,
) -> None:
    """
    A function to create/update subject-source records on the target ER site. Currentely only
    working to 'Create' records rather than update them.
    """
    try:
        for _, row in subject_sources_df.iterrows():
            #  Check whether the subjectsource exists already. The only way to do this reliably is using a uuid.
            #  Because of the fact that a subject-source can easily have multiple times then we have no way of truly
            #  knowing whether it has already been created.
            subjectsources = er_io.get_subjectsources(subjects=row["subject"], sources=row["source"])

            if subjectsources.empty:
                # TODO: this logic should be moved into the er_io.post_subjectsource() method
                # as we're currently not posting location there
                location = row["location"]
                if location is None:
                    # location = dict(latitude=1.5, longitude=1.5)
                    location = ""

                payload = dict(
                    id=row[
                        "id"
                    ],  # TODO: this doesn't work - we can't set the subjectsource record with a specific uuid
                    subject=row["subject"],
                    source=row["source"],
                    assigned_range=row["assigned_range"],
                    additional=row["additional"],
                    location=location,  # this works to set the location even though the server response is None
                )
                response = er_io._post(
                    path=f"subject/{row['subject']}/sources",
                    payload=payload,
                )
                logger.info(f"subjects_source created: {response}")
            else:
                logger.info("subject-source records already exist on the ER site for this subject+source")

    except Exception as e:
        logger.error(f"An error occurred trying to get or create the subjectsources: {e}")
        raise e


def get_er_event(
    er_io: ecoscope.io.EarthRangerIO = None,
    id: str = None,
) -> pd.Series:
    """
    Download a pd.Series for a single event in ER based on the ID
    """
    logger.info(f"Requesting event {id}")
    try:
        # Look up event based on id
        event = er_io.get_events(
            event_ids=[id],
            include_details=True,
        )
    except Exception as e:
        # handle case where there is no existing event by creating an empty series
        print(f"Event does not exist, thus empty series shall be created. {e}")
        return pd.Series()
    if event.empty:
        return pd.Series()
    else:
        return event.iloc[0]


def get_events_from_type(er_io: ecoscope.io.EarthRangerIO, event_type_value: str = None) -> gpd.GeoDataFrame:
    """Download events of a given event type"""
    try:
        # get the event_types
        df_event_types = er_io.get_event_types()

        # figure out the event type uuid
        event_id = df_event_types[df_event_types["value"] == event_type_value]["id"]

        # use the event type uuid to query the events API
        events_df = er_io.get_events(event_type=[event_id])

        logger.info("events download success!")
        return events_df
    except Exception as e:
        logger.error(f"events download failed!: {e}")
        sys.exit(1)


def create_or_update_events(er_io: ecoscope.io.EarthRangerIO, events_df=None) -> None:
    """
    Requires that the Event Category 'monitoring' exists on the ER instance
    """
    try:
        # Check to make sure that the event_types exists. If not, create.
        event_types_df = er_io.get_event_types()
        event_types = event_types_df["value"].unique()

        for e in events_df["event_type"].unique():
            if e not in event_types:
                payload = {
                    "value": e,
                    "display": e,
                    "ordernum": 0,
                    "is_collection": False,
                    "category": "monitoring",
                    "is_active": True,
                    "default_priority": 0,
                    "default_state": "new",
                    "geometry_type": "Point",
                    "resolve_time": 32767,
                    "auto_resolve": True,
                }
                er_io.post_event_type(payload)

        # make a copy of the df
        events_df = events_df.copy()

        # Make sure that title & comments data are filled in
        events_df["comment"] = events_df["comment"].fillna("")
        events_df["title"] = events_df["title"].fillna("")

        # drop columns that don't make sense to try to copy over
        events_df.drop(
            columns=[
                "serial_number",
                "related_subjects",
                # 'patrols',
                # 'patrol_segments',
            ],
            inplace=True,
        )

        # reset the index so we can make use of the uuid more easily
        events_df.reset_index(inplace=True)

        # turn any NaN values to mpty strings
        # events_df = events_df.fillna('')

        """
        Determine the list of events that already exists on the target ER server based on their ...? We
        need to make the request in chunks when the id list is long as we can easily hit a URI length
        limit in the request for multiple IDs. Note that there is currently a bug in ER (discovered during
        the testing of this code) whereby only the last event is returned by the activity/events api even
        if more then one event ID is requested. So we need to chunk=1 to make sure we're getting each event
        otherwise we miss some this issue is being tracked here: https://allenai.atlassian.net/browse/ERA-10527
        """

        df_chunk_size = 1
        existing_events_df = pd.concat(
            [
                er_io.get_events(event_ids=chunk["id"].astype(str).values.flatten().tolist())
                for chunk in chunk_df(events_df, df_chunk_size)
            ]
        ).reset_index()

        """
        split the events_df into two:
           1) events we want to create that need creating
           2) events that already exist that we want to update
        """
        event_responses = []
        if existing_events_df.empty:
            event_responses.append(er_io.post_event(events_df))

        else:
            events_df_create = events_df[~events_df["id"].isin(existing_events_df["id"].to_list())]
            events_df_patch = events_df[events_df["id"].isin(existing_events_df["id"].to_list())]

            if not events_df_create.empty:
                # post the events
                event_responses.append(er_io.post_event(events_df_create))

            if not events_df_patch.empty:
                # patch the events one by one (because ER does not support bulk updates via the activity\events API)
                for i, row in events_df_patch.iterrows():
                    event_responses.append(er_io._patch(f"activity/event/{row['id']}", payload=row.to_dict()))
        # return event_responses

    except ERClientException as e:
        logger.error(f"Error posting events: {e}")


def create_or_update_observations(
    er_io: ecoscope.io.EarthRangerIO, observations: pd.DataFrame = None, chunk_size=1000
) -> None:
    """
    A function to create or update existing observations. Assumes the input dataframe has these columns:
        "id"
        "location"
        "recorded_at"
        "source"
        "exclusion_flags"
        "additional"
    """

    if not observations.empty:
        # chunk the df so we get a sense of progress
        for chunk in tqdm(chunk_df(observations, chunk_size=chunk_size)):
            try:
                er_io.post_observation(chunk.to_dict("records"))
            except Exception as e:
                if "already exists" in e.args[0]:
                    pass
                else:
                    raise e


def chunk_df(df, chunk_size):
    chunks = [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    return chunks


# ----------------------------------------------------------------------------------------------

"""Download Patrols"""


def get_patrols_in_range(er_io: ecoscope.io.EarthRangerIO, start_date, end_date):
    """Fetch patrols in a specified date range"""
    try:
        patrol_df = er_io.get_patrols(
            since=pd.Timestamp(start_date).isoformat(),
            until=pd.Timestamp(end_date).isoformat(),
        )
        logger.info("patrols downloaded successfully!.")
        return patrol_df
    except Exception as e:
        logger.info(f"patrols download failed!: {e}")
        sys.exit(1)


"""Download patrols segments"""


def get_patrols_segment(er_io: ecoscope.io.EarthRangerIO):
    try:
        patrols_segments_df = er_io.get_patrol_segments()
        logger.info("Patrols segments downloaded successfully!")
        return patrols_segments_df
    except Exception as e:
        logger.info(f"Patrol segments download failed!:{e}")
        sys.exit(1)


"""Download patrols segments events"""


def get_patrols_segment_events(er_io: ecoscope.io.EarthRangerIO, patrol_segment_id: str):
    try:
        patrols_segments_event_df = er_io.get_patrol_segment_events(patrol_segment_id=patrol_segment_id)
        logger.info("Patrols events downloaded successfully!")
        return patrols_segments_event_df
    except Exception as e:
        logger.info(f"Patrols events download failed!: {e}")
        sys.exit(1)


"""Post patrols"""


def post_patrol_data(er_upload: ecoscope.io.EarthRangerIO, patrol_df):
    for row in patrol_df.itertuples(index=False):
        try:
            er_upload.post_patrol(priority=100, state="done")
            print("patrols posted successfully!")
        except ERClientException as e:
            print(f"Failed to post patrols!: {e}")


"""Post patrol segments"""


def post_patrol_segments(er_upload, patrol_segments_list, patrol_types, patrols, subjects_name=""):
    # Get the patrols and subjects
    patrols_df = get_patrols_in_range(er_upload)
    subjects = er_upload.get_subjects()

    # Ensure subject exists
    subject_id = subjects[subjects["name"] == subjects_name]["id"].values
    if len(subject_id) == 0:
        raise ValueError(f"Subject '{subjects_name}' not found in subjects data.")
    subject_id = subject_id[0]

    # Iterate through patrols and post segments
    for i, patrol in enumerate(patrols_df):
        if i >= len(patrol_segments_list) or i >= len(patrol_types):
            raise IndexError(f"Patrols or patrol segment list out of range. Patrol index {i} is invalid.")

        for j in range(len(patrol)):
            patrol_segment_id = patrol_segments_list[i].loc[j, "id"]
            patrol_id = patrol.loc[j, "id"]
            patrol_type = patrol_types[i]
            start_time = patrols[i].loc[j, "patrol_segments"][0]["time_range"]["start_time"]
            end_time = patrols[i].loc[j, "patrol_segments"][0]["time_range"]["end_time"]

            # Post patrol segment
            er_upload.post_patrol_segment(
                patrol_segment_id=patrol_segment_id,
                patrol_id=patrol_id,
                patrol_type=patrol_type,
                tracked_subject_id=subject_id,
                start_time=start_time,
                end_time=end_time,
            )

    logger.info("Patrol segments posted successfully.")


def export_gpkg(df, dir=None, outname=None, lyrname=None):
    df = df.copy()
    check_columns = df.columns.to_list()
    check_columns.remove("geometry")
    df.drop(
        columns=df[check_columns].columns[df[check_columns].applymap(lambda x: isinstance(x, list)).any()],
        errors="ignore",
        inplace=True,
    )
    df.to_file(os.path.join(dir or ".", outname or "df.gpkg"), layer=lyrname or "df")
