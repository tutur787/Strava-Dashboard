from stravalib import Client
from typing import Tuple

def refresh_token(client_id: int, client_secret: str, refresh_token: str) -> Tuple[str, str]:
    client = Client()
    token_response = client.refresh_access_token(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
    )
    return token_response["access_token"], token_response["refresh_token"]