from stravalib import Client
import webbrowser

def authenticate(client_id: str, client_secret: str) -> str:
    client = Client()
    print(client_id)
    print(client_secret)
    print("--------------------------------")
    request_scope = ["read_all", "profile:read_all", "activity:read_all"]
    redirect_url = "http://127.0.0.1:5000/authorization"
    url = client.authorization_url(
        client_id=client_id,
        redirect_uri=redirect_url,
        scope=request_scope,
        approval_prompt="force",
    )
    webbrowser.open(url)
    code = input("Enter the code from the url: ")
    token_response = client.exchange_code_for_token(
        client_id=client_id, client_secret=client_secret, code=code
    )
    return token_response["access_token"], token_response["refresh_token"]