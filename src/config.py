from os import getenv

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

apiKeys = {
    "openAi": getenv("openAiAPI"),
    "collegeScoreCard": getenv("collegeScoreCardAPI"),
}
