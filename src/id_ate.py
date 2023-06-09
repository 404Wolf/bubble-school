import pandas as pd
from university.utils import fetch_id
import logging

logging.basicConfig(level=logging.DEBUG)

# Load the CSV file
df = pd.read_csv('university/dataset.csv')

# Iterate over the rows
try:
    for index, row in df.iterrows():
        # Check if the ID is missing
        if pd.isnull(row['id']):
            # Fetch the ID based on the name
            id_value = fetch_id(row['schoolName'])
            # Fill in the missing ID
            df.at[index, 'id'] = id_value
finally:
    # Save the modified DataFrame to a new CSV file
    df.to_csv('output_file.csv', index=False)
