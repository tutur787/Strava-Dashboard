from stravalib import Client

def authenticate(client_id: int, client_secret: str) -> str:
    client = Client()
    url = client.authorization_url(
        client_id=client_id,
        redirect_uri="http://127.0.0.1:5000/authorization",
        approval_prompt="force",
        scope=["activity:read_all", "profile:read_all", "activity:read_all", "profile:read_all"],
    )
    print(url)
    code = input("Enter the code from the url: ")
    token_response = client.exchange_code_for_token(
        client_id=client_id, client_secret=client_secret, code=code
    )
    return token_response["access_token"], token_response["refresh_token"]