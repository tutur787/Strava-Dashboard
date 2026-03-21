from stravalib.client import Client
import json
from helpers.fetch_new_activities import fetch_new_activities
from helpers.refresh_token import refresh_token
from helpers.authenticate import authenticate
from datetime import datetime

CLIENT_ID = int(input("Enter your Strava CLIENT_ID: "))
CLIENT_SECRET = input("Enter your Strava CLIENT_SECRET: ")

option = input("Enter the option you want to perform (0: authenticate, 1: refresh token, 2: fetch new activities): ")
if option == "0":
    new_access_token, new_refresh_token = authenticate(CLIENT_ID, CLIENT_SECRET)
    print(f"New ACCESS_TOKEN: {new_access_token}")
    print(f"New REFRESH_TOKEN: {new_refresh_token}")
if option == "1":
    REFRESH_TOKEN = input("Enter your Strava REFRESH_TOKEN: ")
    new_access_token, new_refresh_token = refresh_token(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
    print(f"New ACCESS_TOKEN: {new_access_token}")
    print(f"New REFRESH_TOKEN: {new_refresh_token}")
elif option == "2":
    ACCESS_TOKEN = input("Enter your Strava ACCESS_TOKEN: ")
    START_FROM = input("Enter the timestamp of the first activity to fetch (YYYY-MM-DD): ")
    if START_FROM == "":
        START_FROM = None
    else:
        START_FROM = datetime.strptime(START_FROM, "%Y-%m-%d").strftime("%Y-%m-%d")
    client = Client(access_token=ACCESS_TOKEN)
    n_details_added, n_streams_added = fetch_new_activities(client, athlete_path="data/athlete.json", detailed_path="data/strava_runs_detailed.json", streams_path="data/strava_runs_streams.json", start_from=START_FROM)
    print(f"Added {n_details_added} detailed activities and {n_streams_added} stream sets.")