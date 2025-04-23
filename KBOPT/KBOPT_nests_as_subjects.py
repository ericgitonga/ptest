import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
import ecoscope

# import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# flake8: noqa
import helper as helper

# flake8: noqa
from helper import logger

# Initialize ecoscope
ecoscope.init()

# Load environment variables
load_dotenv()

# Output DIR
output_dir = "Ecoscope-Outputs/KBOPT"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# define the provider key
provider_key = os.getenv("PROVIDER_KEY") or "ecoscope"


def create_nest_check_observation(
    er_io: ecoscope.io.EarthRangerIO = None,
    nest_name="",
    nest_loc=(0, 0),
    nest_check_time=pd.NaT,
    nest_check_status=pd.NA,
    nest_check_condition=pd.NA,
):
    """
    Post an observation with the sensor observation API. This has the benefit of setting the subject status,
    and if the same observation exists, that the status will update even if the observation doesn't get stored.
    """
    logger.info(f"Posting latest nest check as an observation {nest_name} ")
    try:
        """
        Fallen nests will be displayed with the colour “grey”
        Present and inactive nests will be displayed with the colour “grey”
        Present and active nests will be displayed with the colour “green”

        If a fallen or inactive nest hasn’t been checked in three months, then turn the colour of the nest from grey to red.
        If an active nest hasn’t been checked in 6 weeks, then turn the colour of the  nest from green to red.
        If a nest has been destroyed, then the nest subject should be inactivated.

        nest_check_status: [nan, 'Failed', 'Inactive', 'In_progress', 'Nest_building', 'Nest_lined', 'Not Applicable', 'Post_fledging', 'Success']
        nest_check_condition: [nan 'Present' 'Fallen' 'Destroyed']

        """

        # Nest Last Checked:
        # 0 - 6weeks: Green
        # 6 weeks - 6 months: Amber
        # > 6 months: Red

        radio_state = "alarm"  # default to alarm status

        # “online-gps”: “green”, “online”: “blue”, “offline”: “gray”, “alarm”: “red”, “na”: “black”
        days_since = pd.Timedelta(pd.Timestamp.utcnow() - nest_check_time).days

        if days_since <= 90:
            if nest_check_status == "Fallen":
                radio_state = "offline"
            if (nest_check_condition == "Present") and (
                (nest_check_status == "Inactive") or (nest_check_status == "Success")
            ):
                radio_state = "offline"
            if (nest_check_condition == "Present") and (
                (nest_check_status == "In_progress")
                or (nest_check_status == "Nest_building")
                or (nest_check_status == "Nest_lined")
            ):
                radio_state = "online-gps"

        if days_since > 42:
            if (nest_check_condition == "Present") and (
                (nest_check_status == "In_progress")
                or (nest_check_status == "Nest_building")
                or (nest_check_status == "Nest_lined")
            ):
                radio_state = "alarm"

        observation = {
            "message_key": "observation",
            "location": {"lat": nest_loc[1], "lon": nest_loc[0]},
            "recorded_at": nest_check_time.isoformat(),
            "manufacturer_id": nest_name,
            "subject_name": nest_name,
            "subject_subtype": "nest",
            "model_name": "Nest T1000",
            "source_type": "gps-radio",
            "additional": {
                "event_action": "device_state_changed",
                "radio_state": radio_state,
                "radio_state_at": nest_check_time.isoformat(),
            },
        }
        er_io.post_sensor_observation(observation, sensor_type="generic")

    except Exception as e:
        logger.error(f"An error occurred creating an observation: {e}")


def process_nests(
    nest: pd.DataFrame = None,
    er_io: ecoscope.io.EarthRangerIO = None,
):
    # the name of the subject/NestID
    name = nest.name

    logger.info(f"Processing nest: {name}")

    # sort the data according to the most recent nest check
    most_recent_nest = nest.sort_values(by="nest_check_time", ascending=False).iloc[0]

    # when was the nest last checked? Use the nest_check_time if it exists or whenever the nest was first recorded
    last_check_time = most_recent_nest["nest_check_time"]
    if pd.isnull(last_check_time):
        last_check_time = most_recent_nest["nest_id_time"]

    # set the nest status and condition
    nest_check_condition = most_recent_nest["nest_check_condition"]
    nest_check_status = most_recent_nest["nest_check_status"]

    # set the subject active status
    subject_active = True
    if nest_check_condition == "Destroyed":
        subject_active = False

    # Nest Location
    nest_loc = most_recent_nest["nest_id_geometry"].x, most_recent_nest["nest_id_geometry"].y

    # To-Do: do we want to store any nest data in the subject additional info?
    nest_data = {}

    # # convert any timestamps to strings
    # nest_data = { key: value if not isinstance(value, dt.datetime) else value.isoformat() for key, value in nest_data.items() }

    # # convert any null values to blank text
    # nest_data = {k: "" if pd.isna(v) else v for k, v in nest_data.items() if type(v) is not list}

    # Get or create the nest subject
    subject = helper.create_or_update_subject(
        er_io=er_io,
        subject_name=name,
        subject_type="wildlife",
        subject_subtype="nest",
        is_active=subject_active,
        provider=provider_key,
        additional=nest_data,
    )

    # Get or create the nest source
    source = helper.create_or_update_source(
        er_io=er_io,
        manufacturer_id=name,
        source_type="gps-radio",
        model_name="Nest T1000",
        provider=provider_key,
        additional={},
    )

    # Create the Subject-Source if it doesn't exist
    _ = helper.create_or_update_subjectsource(
        er_io=er_io,
        subject_id=subject["id"],
        source_id=source["id"],
        provider=provider_key,
        static_loc=None,
    )

    # Process the nest check as an observation
    create_nest_check_observation(
        er_io=er_io,
        nest_name=name,
        nest_loc=nest_loc,
        nest_check_time=last_check_time,
        nest_check_condition=nest_check_condition,
        nest_check_status=nest_check_status,
    )


# main.py
def main():
    # Initialize EarthRanger
    er_io = helper.earthranger_init()

    # get a dataframe of event types
    event_types_df = pd.DataFrame(er_io.get_event_types())

    # read all nest ID events
    nest_id_df = er_io.get_events(
        event_type=[event_types_df.query("value == 'nest_id'")["id"].values[0]],
        since=None,
        until=None,
    ).reset_index()

    # unpack the event_details column
    ecoscope.io.earthranger_utils.normalize_column(nest_id_df, "event_details")

    # transform the nest_id_df columns
    nest_id_column_transform = json.loads(os.getenv("NEST_ID_COLUMN_TRANSFORM"))
    nest_id_df = helper.transform_df_columns(df=nest_id_df, column_map_dict=nest_id_column_transform)

    # read all nest check events
    nest_check_df = er_io.get_events(
        event_type=[event_types_df.query("value == 'nest_check'")["id"].values[0]],
        since=None,
        until=None,
    ).reset_index()

    # unpack the nest check events
    ecoscope.io.earthranger_utils.normalize_column(nest_check_df, "event_details")

    # transform the nest_check_df columns
    nest_check_column_transform = json.loads(os.getenv("NEST_CHECK_COLUMN_TRANSFORM"))
    nest_check_df = helper.transform_df_columns(df=nest_check_df, column_map_dict=nest_check_column_transform)

    # Left Join Nest IDs and Nest Checks
    nests = nest_id_df.merge(nest_check_df, how="left", left_on="nest_id_id", right_on="nest_check_nest_id")

    # Process each nest
    nests.groupby(["nest_id_id"]).apply(process_nests, er_io=er_io, include_groups=False)


if __name__ == "__main__":
    main()
    quit()
