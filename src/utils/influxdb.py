"""Module for writing data to InfluxDB."""
import os
import logging
import requests
from dotenv import load_dotenv
from influxdb_client_3 import InfluxDBClient3, Point

load_dotenv("./secrets/.env")

INFLUXDB_HOST = os.environ["INFLUXDB_HOST"]
INFLUXDB_TOKEN = os.environ["INFLUXDB_TOKEN"]
INFLUXDB_DB = os.environ["INFLUXDB_DB"]

logger = logging.getLogger(__name__)

class InfluxDBService:
    """Service for saving data to influxdb"""
    def __init__(self):
        self.client = InfluxDBClient3(host=INFLUXDB_HOST, token=INFLUXDB_TOKEN, database=INFLUXDB_DB)


    def close(self):
        """Closes client connection"""
        self.client.close()


    def delete_table(self, table: str, hard: bool = False):
        """Deletes a table from InfluxDB, optionally performing a hard delete to
        instantly remove underlying data."""
        params = {"db": INFLUXDB_DB, "table": table}
        if hard:
            params["hard_delete_at"] = "now"
        resp = requests.delete(
            f"{INFLUXDB_HOST}/api/v3/configure/table",
            params=params,
            headers={"Authorization": f"Bearer {INFLUXDB_TOKEN}"},
            timeout=20
        )
        if resp.status_code == 404:
            logger.info("Table '%s' doesn't exist, nothing to delete.", table)
            return
        resp.raise_for_status()


    def write_point(self, measurement: str, tags: dict, fields: dict, time=None):
        """Writes a single point to InfluxDB"""
        p = Point(measurement)
        for tag_key, tag_value in tags.items():
            p = p.tag(tag_key, tag_value)
        for field_key, field_value in fields.items():
            p = p.field(field_key, field_value)
        if time is not None:
            p = p.time(time)

        self.client.write(p)
        self.client.close()


    def write_points(self, records: list[dict]):
        """Writes multiple points to InfluxDB using a single connection.

        Each record in 'records' must be a dict with keys:
            - measurement: str
            - tags: dict
            - fields: dict
            - time: optional (datetime, epoch int, or ISO string)
        """
        points = []
        for record in records:
            p = Point(record["measurement"])
            for tag_key, tag_value in record.get("tags", {}).items():
                p = p.tag(tag_key, tag_value)
            for field_key, field_value in record.get("fields", {}).items():
                p = p.field(field_key, field_value)
            if record.get("time") is not None:
                p = p.time(record["time"])
            points.append(p)

        if points:
            self.client.write(points)
        self.client.close()


    def write_dataframe(self, df, measurement: str, tag_cols: list[str], field_cols: list[str], time_col: str):
        """Writes a DataFrame to InfluxDB."""
        df.foreachPartition(
            lambda rows: write_partition(rows, measurement, tag_cols, field_cols, time_col)
        )


def row_to_point(row, measurement: str, tag_cols: list[str], field_cols: list[str], time_col: str) -> Point:
    """Converts a row to an InfluxDB Point."""
    p = Point(measurement)
    for tag in tag_cols:
        p = p.tag(tag, getattr(row, tag))
    for field in field_cols:
        p = p.field(field, getattr(row, field))
    if time_col:
        p = p.time(getattr(row, time_col))
    return p

def write_partition(rows, measurement: str, tag_cols: list[str], field_cols: list[str], time_col: str):
    """Writes a partition of rows to InfluxDB."""
    client = InfluxDBClient3(host=INFLUXDB_HOST, token=INFLUXDB_TOKEN, database=INFLUXDB_DB)
    points = [row_to_point(r, measurement, tag_cols, field_cols, time_col) for r in rows]
    if points:
        client.write(points)
    client.close()
