import os
import sys
import ast
import pandas as pd
import geopandas as gpd
import numpy as np

from dotenv import load_dotenv

load_dotenv()  # Load environment variables

import ecoscope

from ecoscope_workflows_ext_ecoscope.connections import EarthRangerConnection
import ecoscope.plotting as plotting

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# flake8: noqa
import helper as helper

# Output DIR
output_dir = os.getenv("OUTPUT_DIR") or input("Enter the output directory: ") or "Ecoscope-Outputs/KBOPT/Polylines"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize ecoscope
ecoscope.init()


def main():
    # Setup a connection to EarthRanger
    er_server = os.getenv("ER_SERVER")
    er_username = os.getenv("ER_USERNAME")
    er_password = os.getenv("ER_PASSWORD")

    er_io = EarthRangerConnection(
        server=er_server,
        username=er_username,
        password=er_password,
        tcp_limit=5,
        sub_page_size=4000,
    ).get_client()

    # get a dataframe of patrols based on the parameters
    er_patrol_type = os.getenv("ER_PATROL_TYPE")
    since_filter = pd.to_datetime(os.getenv("START"))
    until_filter = pd.to_datetime(os.getenv("END"))
    er_patrol_serials_filter = ast.literal_eval(os.getenv("ER_PATROL_SERIALS_FILTER", "None"))
    er_subject_names_filter = ast.literal_eval(os.getenv("ER_SUBJECT_FILTER", "None"))

    patrols_df = er_io.get_patrols(
        since=since_filter.isoformat(),
        until=until_filter.isoformat(),
        patrol_type=er_patrol_type,
    )

    # TODO: filter patrol observations based on serial number https://github.com/wildlife-dynamics/ecoscope-workflows/issues/605
    if er_patrol_serials_filter:
        patrols_df = patrols_df[patrols_df["serial_number"].isin(er_patrol_serials_filter)]

    if not patrols_df.empty:
        # -------------------Relocs/Traj

        # download observations related to the patrols
        patrol_relocs = er_io.get_patrol_observations(
            patrols_df=patrols_df,
            include_patrol_details=True,
            include_subject_details=True,
        )

        # filter based on subject_name
        # TODO: https://github.com/wildlife-dynamics/ecoscope-workflows/issues/607
        if er_subject_names_filter:
            patrol_relocs = patrol_relocs[patrol_relocs["extra__subject__name"].isin(er_subject_names_filter)]

        # Turn the relocations into trajectory segments and union the segments into single polylines
        # TODO: https://github.com/wildlife-dynamics/ecoscope-workflows/issues/606
        def tmp(x):
            if len(x) > 1:
                geo = ecoscope.base.Trajectory.from_relocations(x)["geometry"].unary_union
                if geo:
                    return gpd.GeoSeries({"geometry": geo}, crs=4326)

        patrol_polylines = patrol_relocs.groupby("patrol_serial_number").apply(tmp, include_groups=True)
        # TODO: https://github.com/wildlife-dynamics/ecoscope-workflows/issues/406

        helper.export_gpkg(
            df=patrol_polylines, dir=output_dir, outname="Patrol_Polylines.gpkg", lyrname="patrol_polylines"
        )


if __name__ == "__main__":
    main()
