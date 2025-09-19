# mail_read_poc.py

# Install these packages before running it:
# pip install msal requests

import sys, msal, requests

TENANT   = "uipath.onmicrosoft.com"
# App registration created in Entra portal with "Mail.Read" permission granted by admin
# https://entra.microsoft.com/#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/CallAnAPI/quickStartType~/null/sourceType/Microsoft_AAD_IAM/appId/b912a89b-3beb-486e-915b-526e2299dae8/objectId/6f624b1d-e1b3-4d49-a577-c3e4e727e15a/isMSAApp~/false/defaultBlade/Overview/appSignInAudience/AzureADMyOrg/servicePrincipalCreated~/true
CLIENT_ID= "b912a89b-3beb-486e-915b-526e2299dae8"
SCOPES = ["Mail.Read"]

def token():
    app = msal.PublicClientApplication(CLIENT_ID, authority=f"https://login.microsoftonline.com/{TENANT}")
    accs = app.get_accounts()
    r = app.acquire_token_silent(SCOPES, account=accs[0]) if accs else None
    if not (r and "access_token" in r):
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow: raise SystemExit(flow.get("error_description", "device flow failed"))
        print(f"\nSign in at {flow['verification_uri']} and enter code: {flow['user_code']}\n")
        r = app.acquire_token_by_device_flow(flow)
        if "access_token" not in r: raise SystemExit(r.get("error_description", "auth error"))
    return r["access_token"]

def graph_get(path, **kw):
    h = {"Authorization": f"Bearer {token()}"}
    r = requests.get(f"https://graph.microsoft.com/v1.0{path}", headers=h, timeout=30, **kw)
    r.raise_for_status()
    return r.json()

def main():
    # Grab latest 10 messages with metadata needed + bodyPreview + hasAttachments
    params = {
        "$select": "subject,from,receivedDateTime,isRead,hasAttachments,bodyPreview",
        "$orderby": "receivedDateTime DESC",
        "$top": 10
    }
    data = graph_get("/me/mailFolders/Inbox/messages", params=params)

    for m in data.get("value", []):
        sender = (m.get("from") or {}).get("emailAddress", {})
        flag = "Unread" if not m.get("isRead") else "Read"
        preview = (m.get("bodyPreview") or "").replace("\r", " ").replace("\n", " ")
        preview = preview[:100]  # first 100 chars

        print(f"[{flag:6}] {m.get('receivedDateTime')} | {sender.get('name')} <{sender.get('address')}>")
        print(f"Subject: {m.get('subject')}")
        print(f"Body(100): {preview!r}")

        if m.get("hasAttachments"):
            # Fetch attachment metadata for this message
            atts = graph_get(f"/me/messages/{m['id']}/attachments",
                             params={"$select": "name,contentType,size"})
            if atts.get("value"):
                print("Attachments:")
                for a in atts["value"]:
                    print(f"  - {a.get('name')} ({a.get('contentType')}, {a.get('size')} bytes)")
        print("-" * 80)

if __name__ == "__main__":
    main()


