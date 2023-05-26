import random
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


class Universities:
    FIELDS_FILEPATH = "university/fields.csv"

    def __init__(self, universities: pd.DataFrame):
        self.universities = universities

    @classmethod
    def from_file(cls, filepath: str) -> "Universities":
        return cls(pd.read_csv(filepath, index_col=INDEX_COL))

    def to_file(self, filepath: str):
        self.universities.to_csv(filepath, index_label=INDEX_COL)

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
                except KeyError:
                    # Otherwise look it up and if it is nowhere to be found then skip the
                    # school and move on
                    with suppress(ValueError):
                        school_data = fetch_data(school_id)

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
        clustered_df = self.standardized().dropna()
        centroids = random.sample(tuple(clustered_df.index), k=cluster_count)
        dists = np.zeros(cluster_count)

        for i in range(resolution):
            clusters = np.zeros(len(clustered_df))

            for datapoint_index, datapoint_id in enumerate(clustered_df.index):
                datapoint_coord = clustered_df.loc[datapoint_id].values[1:]
                for centroid_index, centroid_id in enumerate(centroids):
                    centroid_coord = clustered_df.loc[centroid_id].values[1:]
                    dists[centroid_index] = sum(abs(datapoint_coord - centroid_coord))
                clusters[datapoint_index] = dists.argmin()
            clustered_df["cluster"] = clusters

            centroids.clear()
            for cluster_index, cluster in clustered_df.groupby("cluster"):
                cluster_data = cluster.values[:, 1:-1]

                midpoint = sum(cluster_data) / len(cluster)
                closest_datapoint_dist, closest_datapoint_id = 0, -1
                for datapoint_index, datapoint_id in enumerate(cluster.index):
                    datapoint_coord = cluster.loc[datapoint_id].values[1:-1]
                    if sum(abs(datapoint_coord - midpoint)) < closest_datapoint_dist:
                        closest_datapoint_id = datapoint_id

                if closest_datapoint_id != -1:
                    centroids.append(cluster.loc[closest_datapoint_id])
                else:
                    centroids.append(random.choice(tuple(clustered_df.index)))

        return clustered_df, clustered_df.loc[centroids]
