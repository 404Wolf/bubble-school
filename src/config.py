from dotenv import load_dotenv, find_dotenv
from os import getenv

load_dotenv(find_dotenv())

apiKeys = {
    "openAi": getenv("openAiAPI"),
    "collegeScoreCard": getenv("collegeScoreCardAPI"),
}
