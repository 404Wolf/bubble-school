from contextlib import suppress

import numpy as np
import pandas

from university.utils import fetch_data, fetch_id, fields

DATATYPES = {
    "int": int,
    "float": float,
    "str": object,
    "bool": bool,
}


class Universities(pandas.DataFrame):
    FIELDS_FILEPATH = "university/fields.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with suppress(KeyError):
            self.set_index("id", inplace=True, drop=True)

    @classmethod
    def from_file(cls, filepath: str) -> "Universities":
        return cls(pandas.read_csv(filepath))

    def to_file(self, filepath: str):
        self.to_csv(filepath, index_label=False)

    @classmethod
    def from_schools(cls, filepath: str, cache: str | None = None):
        """
        Load data for many schools from a file.

        filepath: Filepath to load the schools txt list from. Each school should be on its
            own line, with no trailing line in the file.
        cache: Optional file to use as a cache for obtaining schools.
        """
        cache = Universities.from_file(cache) if cache else {}

        with open(filepath) as schools_file:
            schools = schools_file.readlines()
            school_ids = []
            for school_name in schools:
                with suppress(ValueError):
                    school_ids.append(fetch_id(school_name.strip()))

            # Create a dictionary of fields to numpy arrays which will later
            # be converted into a pandas DataFrame.
            data = {"id": np.zeros(len(school_ids), dtype=int)}
            for index, field in fields.iterrows():
                datatype = DATATYPES[field['type']]
                data[field["name"]] = np.empty(len(school_ids), dtype=datatype)

            for school_index, school_id in enumerate(school_ids):
                try:
                    school_data = cache.loc[school_id]
                except KeyError:
                    try:
                        school_data = fetch_data(school_id)
                    except ValueError:
                        continue

                data["id"][school_index] = school_id
                for field in school_data.keys():
                    data[field][school_index] = school_data[field]

            return cls(data)
