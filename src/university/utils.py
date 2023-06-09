import atexit
import json
import logging
from time import sleep

import config
import openai
import pandas
import requests

COLLEGE_SCORECARD_API_PATH = "https://api.data.gov/ed/collegescorecard/v1"
FIELDS_FILEPATH = "university/fields.csv"
NAMES_FILEPATH = "university/names.json"

logger = logging.getLogger(__name__)

with open(FIELDS_FILEPATH) as paths_file:
    fields = pandas.read_csv(paths_file)
    logger.debug(f"Loaded {len(fields)} fields from {FIELDS_FILEPATH}")

with open(NAMES_FILEPATH) as names_file:
    names: dict[str, int] = json.load(names_file)
    logger.debug(f"Loaded {len(names)} names from {NAMES_FILEPATH}")
    atexit.register(lambda: json.dump(names, open(NAMES_FILEPATH, "w"), indent=3))
    logger.debug(f"Registered {NAMES_FILEPATH} to be saved on exit.")


def fetch_data(identifier: int) -> dict:
    """
    Fetch the data of a university from its identifier.

    Args:
        identifier (str): The identifier of the university.

    Returns:
        dict: The data of the university.
    """
    resp = requests.get(
        url=f"{COLLEGE_SCORECARD_API_PATH}/schools",
        params={
            "api_key": config.apiKeys["collegeScoreCard"],
            "id": identifier,
        },
    ).json()["results"]
    logger.debug(f"Received {len(resp)} results from the College Scorecard API.")

    if resp:
        resp = resp[0]["latest"]
    else:
        raise ValueError(
            f"Could not find a university with the identifier {identifier}."
        )

    # Parse the data into a dictionary with fields that are standardized path names.
    output = {}
    for index, field in fields.iterrows():
        output[field["key"]] = resp
        for key in field["path"].split("."):
            try:
                output[field["key"]] = output[field["key"]][key]
            except KeyError:
                output[field["key"]] = None
            except TypeError:
                pass
        logger.debug(f"Field {field['key']}'s value set to {output[field['key']]}")

    logger.info(f"Obtained data for {field['name']}")
    return output


def fetch_id(name: str, gtp: bool=True) -> int:
    """
    Fetch the ID of a university from its name.

    Args:
        name (str): The name of the university.
        gtp (bool): Whether or not to use GPT-3 to find the university if needed.
    """
    global names
    if identifier := names.get(name):
        logger.info(f"Found \"{name}\" in the cache of names to ids.")
        return identifier

    # Get a list of possible names similar to the query.
    college_search = requests.get(
        f"{COLLEGE_SCORECARD_API_PATH}/schools",
        params={
            "api_key": config.apiKeys["collegeScoreCard"],
            "per_page": "100",
            "sort": "school.name",
            "fields": "school.name,school.alias,id",
            "school.search": name,
        },
    )

    # Parse the list of possible names into a list of names.
    parsed_college_names = []
    for i in range(10):
        try:
            for college in college_search.json()["results"]:
                # If there is a perfect match, return the university instance for that
                aliases = [college["school.name"]]
                if college["school.alias"]:
                    for alias in college["school.alias"].split(","):
                        aliases.append(alias.strip())
                for match in aliases:
                    if name == match:
                        names[name] = int(college["id"])
                        logger.info(f"Found ID {names[name]} for {name} w/o having to use GTP.")
                        return names[name]

                # Otherwise, add the name to the list of possible names.
                parsed_college_names.append(
                    f"NAME={college['school.name']}, "
                    f"ALIAS={college['school.alias']}, "
                    f"ID={college['id']}"
                )
        except KeyError:
            logger.error(f"Error parsing JSON from {college_search.url}")
            logger.info(f"Retrying in 15 seconds.")
            sleep(15)

    if not gtp:
        return

    # Use OpenAI's GPT-3 to find the most relevant name from the list of possible
    # names.
    for i in range(3):
        try:
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
            break
        except openai.error.RateLimitError:
            logger.warning(f"Rate limit reached for OpenAI's API.")
            sleep(8)
    logger.debug(f"Received college name to use from GPT-3.")

    names[name] = int(name_finder.choices[0]["message"]["content"].replace("ID=", ""))

    logger.info(f"Found ID {names[name]} for {name}.")
    # Build the university instance from the id of the most relevant name.
    return names[name]
