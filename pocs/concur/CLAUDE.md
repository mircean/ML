# Concur Expense Receipt Tracker

## Architecture

This application automatically finds airline expense receipts in Outlook emails and matches them to Concur expenses that require receipts.

### Core Components

- **`concur_auth.py`**: Handles OAuth 2.0 PKCE authentication with SAP Concur API
  - Browser-based OAuth flow with local callback server
  - Token caching in `.concur_token_cache.json`
  - Automatic token refresh

- **`concur_client.py`**: Pure API client for Concur operations
  - `get_active_expense_reports()` - fetches expense reports
  - `get_expenses(report_id)` - fetches expenses for a specific report
  - `upload_receipt_image(entry_id, image_data)` - uploads receipt to expense entry via Image v1 API
  - `_make_request()` - handles authenticated API requests
  - **Does NOT contain business logic** - only API interactions

- **`outlook_auth.py`**: Handles Microsoft Graph API authentication
  - MSAL-based OAuth flow for Outlook/Graph API access
  - Token caching in `msal_cache.json`
  - Manages user authentication state

- **`outlook_client.py`**: Pure API client for Outlook/Graph operations
  - `search_messages()` - searches emails with Microsoft Graph queries
  - `get_message_attachments()` - retrieves attachment metadata
  - `get_attachment_content()` - downloads attachment content
  - **Does NOT contain business logic** - only API interactions

- **`main.py`**: Application logic and orchestration
  - Configuration loading and validation
  - Business logic for matching receipts to expenses
  - Azure Form Recognizer integration for PDF parsing
  - Invoice matching algorithm
  - Report processing and statistics

- **`config.py`**: Configuration and logging setup
  - Centralized logging configuration
  - Microsoft Graph API scopes

### Design Principles

- **Separation of Concerns**: API clients only do API calls, main handles business logic
- **Single Responsibility**: Each module has one clear purpose
- **Module-level logger**: `logger = logging.getLogger(__name__)` at module level, no per-function loggers
- **Error Handling**: Use asserts for unexpected conditions, minimal try/catch
- **Caching**: PDF parse results cached in-memory to avoid redundant Azure API calls

### Key Functions

- `format_expense_info(expense)` - formats expense data for logging
- `parse_pdf_with_azure(pdf_content, filename)` - extracts invoice data from PDFs using Azure Form Recognizer
- `match_invoice(total, date, invoice)` - matches expense to parsed invoice by amount and date
- `search(outlook_client, total, date, days_to_search)` - searches Outlook for matching receipts
- `main()` - orchestrates the entire workflow

### Workflow

1. Authenticate with Concur and Outlook APIs
2. Fetch active expense reports from Concur
3. Find airline expenses that require receipts but don't have them
4. For each missing receipt:
   - Search Outlook emails from Amex travel with PDF attachments
   - Parse PDFs with Azure Form Recognizer
   - Match invoice amounts and dates to expense
   - Upload matched receipt to Concur (unless `--dry-run`)
5. Display statistics (total expenses, matched, not matched)

### CLI Options

- `--dry-run` - Find and match receipts but do not upload them to Concur

### Dependencies

- Uses `uv` for dependency management
- Key deps: `requests`, `python-dotenv`, `msal`, `azure-ai-formrecognizer`, `tqdm`
- Install with: `uv sync`

### Environment Configuration

**Concur API:**
- `CONCUR_CLIENT_ID` - OAuth client ID
- `CONCUR_CLIENT_SECRET` - OAuth client secret
- `CONCUR_BASE_URL` - API base URL (default: https://us2.api.concursolutions.com)
- `CONCUR_SCOPE` - OAuth scopes (default: openid profile user.read identity.user.ids.read expense.report.read IMAGE)
- `REDIRECT_URI` - OAuth callback (default: http://localhost:53682/callback)

**Microsoft Graph API:**
- `GRAPH_TENANT_ID` - Azure AD tenant ID
- `GRAPH_CLIENT_ID` - Azure AD application client ID

**Azure Form Recognizer:**
- `AZURE_ENDPOINT` - Form Recognizer endpoint URL
- `AZURE_API_KEY` - Form Recognizer API key

### Token Caching

- `.concur_token_cache.json` - Concur OAuth tokens (gitignored)
- `msal_cache.json` - Microsoft Graph tokens (gitignored)