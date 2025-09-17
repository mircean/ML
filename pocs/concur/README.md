# Concur Expense Receipt Tracker

A Python application that connects to your SAP Concur account to find active expense reports and identify expenses that require receipts.

## Features

- Authenticates with SAP Concur using OAuth 2.0
- Retrieves all active expense reports for your account
- Identifies expenses that require receipts (image or paper)
- Displays detailed information about reports and missing receipts using structured logging

## Prerequisites

1. **SAP Concur Access**: You need a SAP Concur account with API access
2. **Client Web Services License**: Required for API access
3. **OAuth 2.0 Application**: Must be registered in Concur's Application Management

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your SAP Concur credentials:

```bash
# SAP Concur API Configuration
CONCUR_CLIENT_ID=your_client_id_here
CONCUR_CLIENT_SECRET=your_client_secret_here
CONCUR_BASE_URL=https://us2.api.concursolutions.com  # or your data center URL

# For regular users - Username/Password
CONCUR_USERNAME=your_username_here
CONCUR_PASSWORD=your_password_here

# For SSO users - Company Request Token (expires every 24 hours)
CONCUR_COMPANY_UUID=your_company_uuid_here
CONCUR_REQUEST_TOKEN=your_request_token_here

# Logging Level
LOG_LEVEL=INFO
```

### 3. Get Your Concur API Credentials

1. Log into SAP Concur as a Web Services Admin
2. Go to **Administration > Company > Authentication Admin**
3. Click **OAuth 2.0 Application Management**
4. Click **Create New App**
5. Fill in the required fields and note your Client ID and Client Secret
6. Make sure your app has the necessary scopes:
   - `expense.report.read`
   - `expense.report.readwrite` (if needed)

## Usage

Run the application:

```bash
python main.py
```

The application will:

1. Authenticate with SAP Concur
2. Fetch all your active expense reports
3. Check each report for expenses requiring receipts
4. Log detailed information about any missing receipts

