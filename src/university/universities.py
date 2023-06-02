import logging
from contextlib import suppress
from copy import copy

import numpy as np
import pandas as pd
from university.utils import fetch_data, fetch_id, fields

INDEX_COL = "id"
DATATYPES = {
    "int": int,
    "float": float,
    "str": object,
    "bool": bool,
}
logger = logging.getLogger(__name__)

def dist(coord1: np.ndarray, coord2: np.ndarray):
    """Obtain the manhattan distance between two coordinates."""
    return sum(abs(coord1 - coord2))


class Universities:
    FIELDS_FILEPATH = "university/fields.csv"

    def __init__(self, universities: pd.DataFrame):
        self.universities = universities
        logger.info(f"Loaded {len(self.universities)} universities.")

    def __len__(self):
        return len(self.universities)

    @classmethod
    def from_file(cls, filepath: str) -> "Universities":
        """
        Load data for many schools from a file.

        Args:
            filepath: Filepath to load the schools txt list from. Each school should be on
                its own line, with no trailing line in the file.

        Returns:
            A new Universities object.
        """
        universities = cls(pd.read_csv(filepath, index_col=INDEX_COL))
        logger.info(f"Loaded {len(universities)} universities from {filepath}.")
        return universities

    def to_file(self, filepath: str):
        """Save the data for many schools to a file."""
        self.universities.to_csv(filepath, index_label=INDEX_COL)
        logger.info(f"Saved {len(self.universities)} universities to {filepath}.")

    @classmethod
    def from_schools(cls, filepath: str, cache: str | None = None):
        """
        Load data for many schools from a file.

        filepath: Filepath to load the schools txt list from. Each school should be on its
            own line, with no trailing line in the file.
        cache: Optional file to use as a cache for obtaining schools.
        """
        # The cache is either an empty dataframe or a loaded-from-file dataframe
        cache = Universities.from_file(cache).universities if cache else pd.DataFrame()

        with open(filepath) as schools_file:
            # Read the names of all the schools that we are to load
            schools = schools_file.readlines()
            logger.debug(f"Loaded {len(schools)} schools from {filepath}.")
            # Fetch the id for each school
            school_ids = []
            for school_name in schools:
                with suppress(ValueError):
                    school_ids.append(fetch_id(school_name.strip()))

            # Create a dictionary of fields to numpy arrays which will later
            # be converted into a pd DataFrame.
            data = {"id": np.zeros(len(school_ids), dtype=int)}
            for index, field in fields.iterrows():
                datatype = DATATYPES[field["type"]]
                data[field["name"]] = np.empty(len(school_ids), dtype=datatype)

            for school_index, school_id in enumerate(school_ids):
                try:
                    # If the data already exists in the cache use it
                    school_data = cache.loc[school_id]
                    logger.debug(f"Loaded data for {school_id} from cache.")
                except KeyError:
                    # Otherwise look it up and if it is nowhere to be found then skip the
                    # school and move on
                    try:
                        school_data = fetch_data(school_id)
                        logger.debug(f"Loaded data for {school_id}.")
                    except ValueError:
                        logger.warning(f"Could not find data for {school_id}.")

                # Locate the proper data based on its path
                data["id"][school_index] = school_id
                for field in school_data.keys():
                    data[field][school_index] = school_data[field]

            return cls(pd.DataFrame(data))

    def standardized(self) -> pd.DataFrame:
        """
        Map each datapoint in the Universities' dataframe to be a value between 0 and 1.

        Returns:
            A new and identical dataframe to ours where each datapoint is mapped between
                0 and 1.
        """
        output_df = copy(self.universities)
        # Iterate through the columns of the dataset
        for column in output_df.columns:
            # For all numerical (and thus divide-able) columns, divide all items in the
            # column by the greatest item in the column
            with suppress(TypeError):
                output_df[column] = output_df[column] / max(output_df[column])
        return output_df

    def cluster(self, cluster_count: int, datapoints: tuple, resolution: int = 3):
        """
        Cluster the universities based on certain datapoints.

        Args:
            cluster_count: The number of clusters
            datapoints: A tuple of the names of the datapoints to consider.
            resolution: Number of times to iterate k-means.
        """
        datapoints = [
            self.universities.columns.get_loc(datapoint) for datapoint in datapoints
        ]
        clustered_df = self.standardized().dropna()
        initial_centroids_df = clustered_df.sample(cluster_count)
        centroids = []
        for centroid_index, centroid in initial_centroids_df.iterrows():
            centroids.append(centroid.iloc[datapoints])
        dists = np.zeros(cluster_count)
        clustered_df["cluster"] = -1

        for i in range(resolution):
            for datapoint_index, datapoint in clustered_df.iterrows():
                for centroid_index, centroid in enumerate(centroids):
                    datapoint_coord = datapoint.iloc[datapoints]
                    dists[centroid_index] = dist(datapoint_coord, centroid)
                clustered_df.loc[datapoint_index, "cluster"] = dists.argmin()

            if i != resolution - 1:
                centroids.clear()
                for cluster_index, cluster_df in clustered_df.groupby("cluster"):
                    cluster_data = cluster_df.values[:, datapoints]
                    midpoint = sum(cluster_data) / len(cluster_df)
                    centroids.append(midpoint)

        return clustered_df, np.array(centroids)
