import logging
import sys


def get_events_from_type(er_io, event_type_value):
    """
    Download events of a given event type from an EarthRanger IO instance

    Args:
        er_io (ecoscope.io.EarthRangerIO): EarthRanger IO instance
        event_type_value (str): Event type value to filter events

    Returns:
        GeoDataFrame: DataFrame containing events of the specified type

    Raises:
        SystemExit: If event download fails
    """
    try:
        # Get the event types
        df_event_types = er_io.get_event_types()

        # Find the event type UUID for the given value
        event_id = df_event_types[df_event_types["value"] == event_type_value]["id"]

        # Query events using the event type UUID
        events_df = er_io.get_events(event_type=[event_id])

        logging.info("Events download successful!")
        return events_df
    except Exception as e:
        logging.error(f"Events download failed: {e}")
        sys.exit(1)
