# mail_read_poc.py

# Install these packages before running it:
# pip install msal requests

import sys, msal, requests, os

TENANT   = "uipath.onmicrosoft.com"
# App registration created in Entra portal with "Mail.Read" permission granted by admin
# https://entra.microsoft.com/#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/CallAnAPI/quickStartType~/null/sourceType/Microsoft_AAD_IAM/appId/b912a89b-3beb-486e-915b-526e2299dae8/objectId/6f624b1d-e1b3-4d49-a577-c3e4e727e15a/isMSAApp~/false/defaultBlade/Overview/appSignInAudience/AzureADMyOrg/servicePrincipalCreated~/true
CLIENT_ID= "b912a89b-3beb-486e-915b-526e2299dae8"
SCOPES = ["Mail.Read"]
CACHEFILE = "msal_cache.json"

# ---- one global cache + one global app ----
cache = msal.SerializableTokenCache()
if os.path.exists(CACHEFILE):
    cache.deserialize(open(CACHEFILE, "r").read())

app = msal.PublicClientApplication(
    CLIENT_ID,
    authority=f"https://login.microsoftonline.com/{TENANT}",
    token_cache=cache,  # attach cache to app
)

def save_cache():
    if cache.has_state_changed:
        open(CACHEFILE, "w").write(cache.serialize())

def get_token():
    # 1) try silent (access token or refresh via refresh_token)
    accts = app.get_accounts()
    result = app.acquire_token_silent(SCOPES, account=accts[0] if accts else None)

    # 2) device code if nothing cached yet
    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise SystemExit(flow.get("error_description", "device flow failed"))
        print(f"\nSign in at {flow['verification_uri']} and enter code: {flow['user_code']}\n")
        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise SystemExit(result.get("error_description", "auth error"))

    save_cache()
    return result["access_token"]

# ---- simple Graph helpers ----
session = requests.Session()  # reuse TCP
def graph_get(path, **kw):
    tok = get_token()
    session.headers.update({"Authorization": f"Bearer {tok}"})
    r = session.get(f"https://graph.microsoft.com/v1.0{path}", timeout=30, **kw)

    # If token expired mid-run, try once more silently
    if r.status_code in (401, 403):
        tok = get_token()
        session.headers.update({"Authorization": f"Bearer {tok}"})
        r = session.get(f"https://graph.microsoft.com/v1.0{path}", timeout=30, **kw)

    r.raise_for_status()
    return r.json()

def main():
    params = {
        "$select": "subject,from,receivedDateTime,isRead,hasAttachments,bodyPreview",
        "$orderby": "receivedDateTime DESC",
        "$top": 10
    }
    data = graph_get("/me/mailFolders/Inbox/messages", params=params)
    for m in data.get("value", []):
        sender = (m.get("from") or {}).get("emailAddress", {})
        flag = "Unread" if not m.get("isRead") else "Read"
        preview = (m.get("bodyPreview") or "").replace("\r"," ").replace("\n"," ")[:100]
        print(f"[{flag:6}] {m.get('receivedDateTime')} | {sender.get('name')} <{sender.get('address')}>")
        print(f"Subject: {m.get('subject')}")
        print(f"Body(100): {preview!r}")
        if m.get("hasAttachments"):
            atts = graph_get(f"/me/messages/{m['id']}/attachments",
                             params={"$select":"name,contentType,size"})
            if atts.get("value"):
                print("Attachments:")
                for a in atts["value"]:
                    print(f"  - {a.get('name')} ({a.get('contentType')}, {a.get('size')} bytes)")
        print("-"*80)

if __name__ == "__main__":
    main()

