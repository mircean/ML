# Concur Expense Receipt Tracker

## Architecture

This application follows a clean separation of concerns:

### Core Components

- **`auth.py`**: Handles OAuth 2.0 authentication with SAP Concur API
  - Supports both username/password and company request token authentication
  - Manages token refresh and headers

- **`concur_client.py`**: Pure API client for Concur operations
  - `get_active_expense_reports()` - fetches expense reports
  - `get_expenses(report_id)` - fetches expenses for a specific report
  - `_make_request()` - handles authenticated API requests
  - **Does NOT contain business logic** - only API interactions

- **`main.py`**: Application logic and orchestration
  - Configuration loading and validation
  - Business logic for filtering expenses requiring receipts
  - Report processing loops
  - Logging and output formatting

### Design Principles

- **Separation of Concerns**: API client only does API calls, main handles business logic
- **Single Responsibility**: Each module has one clear purpose
- **Module-level logger**: `logger = logging.getLogger(__name__)` at module level, no per-function loggers
- **Error Handling**: Use asserts for unexpected conditions, minimal try/catch

### Key Functions

- `filter_expenses_requiring_receipts(expenses)` - utility function in main.py
- `get_reports_with_missing_receipts(client)` - orchestrates the main workflow
- `format_expense_info(expense)` - formats expense data for logging

### Dependencies

- Uses `uv` for dependency management
- Key deps: `requests`, `python-dotenv`
- Install with: `uv sync`

### Environment Configuration

Supports two auth methods:
1. Username/Password: `CONCUR_USERNAME` + `CONCUR_PASSWORD`
2. Company Token: `CONCUR_COMPANY_UUID` + `CONCUR_REQUEST_TOKEN`