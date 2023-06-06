import logging

import pandas as pd

from university import Universities

logging.basicConfig(level=logging.DEBUG)


def main():
    existing = Universities.from_file("university/dataset.csv")
    universities = Universities.from_schools("test.txt")
    universities.universities = pd.concat(
        (existing.universities, universities.universities)
    )
    universities.to_file("university/dataset.csv")


if __name__ == "__main__":
    main()
