import json

import openai

import config

import requests

# The path to the College Score Card API.
collegeScoreCardApiPath = "https://api.data.gov/ed/collegescorecard/v1"

# All the fields we care about
with open("pertinent_fields.txt", "r") as file:
    pertinent_fields = file.read().split("\n")


class University:
    """
    A university.

    Attributes:
        identifier (int): The id of the university.
        name (str): The name of the university.
        zip_code (str): The zip code of the university.
        state (str): The state of the university.
        city (str): The city of the university.
        retention_rate (float): The retention rate of the university.
    """

    def __init__(self, identifier: int):
        self.identifier = identifier

    def harvest_data(self):
        """
        Harvest data from the university.
        """
        scorecard_data = requests.get(
            f"{collegeScoreCardApiPath}/schools",
            params={
                "api_key": config.apiKeys["collegeScoreCard"],
                "id": self.identifier,
                "fields": ",".join(pertinent_fields),
            },
        )


    @classmethod
    def from_name(cls, name: str) -> "University":
        """
        Create a university instance from the name of the university.

        1) Search for the university on the College Score Card API and get a list of
            possible names similar to the query.
        2) Use OpenAI's GPT-3 to find the most relevant name from the list of possible
            names and return the university instance for the school with that name.

        Args:
            name (str): The name of the university.

        Returns:
            University: The university instance.
        """
        # Get a list of possible names similar to the query.
        college_search = requests.get(
            f"{collegeScoreCardApiPath}/schools",
            params={
                "api_key": config.apiKeys["collegeScoreCard"],
                "per_page": "100",
                "sort": "school.name",
                "fields": "school.name,school.alias,id",
                "school.search": name,
            },
        )
        print(college_search.json())
        # Parse the list of possible names into a list of names.
        parsed_college_names = []
        for college in college_search.json()["results"]:
            parsed_college_names.append(
                f"NAME={college['school.name']}, "
                f"ALIAS={college['school.alias']}, "
                f"ID={college['id']}"
            )

        # Use OpenAI's GPT-3 to find the most relevant name from the list of possible
        # names.
        name_finder = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=(
                {
                    "role": "system",
                    "content": "Given a list of possible college aliases/names and a "
                    "query, you respond with the ID of the most relevant college "
                    "name. Your response should only be the ID of the school, and "
                    "should have no prefix.",
                },
                {"role": "user", "content": f"QUERY='{name}'; {parsed_college_names}"},
            ),
            temperature=0.0,
            max_tokens=12,
            api_key=config.apiKeys["openAi"],
        )

        # Build the university instance from the id of the most relevant name.
        return cls(name_finder.choices[0]["message"]["content"])


test = University.from_name("UC Berkeley").harvest_data()
