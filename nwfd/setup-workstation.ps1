<# setup-workstation.ps1
Run elevated. Idempotent best-effort for:
1) Install Windows Terminal & set default
2) Set active network to Private
3) Enable Ping (ICMPv4) on Private
4) Power: screen off 10m, sleep 30m; no password on wake
5) Admins: mircean@outlook.com, tate@nwfamilydental.net; local op (password: "password") -> Admins
6) Enable Remote Desktop
7) Startup script to map IPC$ shares with saved creds
#>

$ErrorActionPreference = 'Stop'

function Require-Elevation {
  if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
      ).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    throw "Please run this script as Administrator."
  }
}
Require-Elevation

# --- 1) Windows Terminal -------------------------------------------------------
Write-Host "Installing Windows Terminal (winget)..." -ForegroundColor Cyan
try {
  winget install --id Microsoft.WindowsTerminal -e --source winget `
    --accept-source-agreements --accept-package-agreements | Out-Null
} catch {
  Write-Warning "winget install failed or Terminal already installed. Continuing..."
}

# Set Windows Terminal as default console host
try {
  $wt = Get-AppxPackage -Name "Microsoft.WindowsTerminal" -ErrorAction Stop
  $pfn = $wt.PackageFamilyName + "!App"
  
  # Simplified registry approach - single key instead of multiple
  $regPath = "HKCU:\Console"
  if (-not (Test-Path $regPath)) { New-Item -Path $regPath -Force | Out-Null }
  
  Set-ItemProperty -Path $regPath -Name "DelegationConsole" -Value $pfn -Force
  Set-ItemProperty -Path $regPath -Name "DelegationTerminal" -Value $pfn -Force
  Set-ItemProperty -Path $regPath -Name "ForceV2" -Value 1 -Force
  
  Write-Host "Windows Terminal set as default." -ForegroundColor Green
} catch {
  Write-Warning "Could not set Terminal as default - Windows Terminal may not be installed."
}

# --- 2) Set active network to Private -----------------------------------------
Write-Host "Setting active network profile to Private..." -ForegroundColor Cyan
$profile = Get-NetConnectionProfile | Where-Object { $_.IPv4Connectivity -ne 'Disconnected' } | Select-Object -First 1
if ($null -ne $profile) {
  Set-NetConnectionProfile -InterfaceIndex $profile.InterfaceIndex -NetworkCategory Private
  Write-Host "Network '$($profile.Name)' set to Private." -ForegroundColor Green
} else {
  Write-Warning "No active network profile found. Skipped Private switch."
}

# --- 3) Firewall: enable Ping (ICMPv4) on Private ------------------------------
Write-Host "Enabling ICMPv4 Echo (Ping) on Private..." -ForegroundColor Cyan
try {
  Enable-NetFirewallRule -DisplayName "File and Printer Sharing (Echo Request - ICMPv4-In)" -ErrorAction Stop
} catch {
  Enable-NetFirewallRule -DisplayGroup "File and Printer Sharing" -ErrorAction SilentlyContinue
}
Get-NetFirewallRule -DisplayGroup "File and Printer Sharing" |
  Where-Object { $_.Enabled -eq 'True' } |
  Set-NetFirewallRule -Profile Private


# --- 4) Power settings ---------------------------------------------------------
Write-Host "Configuring power settings (AC): screen off 10m, sleep 30m; no password on wake..." -ForegroundColor Cyan
powercfg -x -monitor-timeout-ac 10 | Out-Null
powercfg -x -standby-timeout-ac 30 | Out-Null
try {
  powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_NONE CONSOLELOCK 0 | Out-Null
  powercfg /SETDCVALUEINDEX SCHEME_CURRENT SUB_NONE CONSOLELOCK 0 | Out-Null
  powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
  Write-Host "Disabled 'require sign-in after sleep' (best effort; policy may override)." -ForegroundColor Green
} catch {
  Write-Warning "Could not change 'require sign-in after sleep' (possibly policy-controlled)."
}

# --- 5) Users & Admins ---------------------------------------------------------
###
### TODO This doesn't work anymore, Microsoft Accounts need to be linked through Windows Settings first.
###
Write-Host "Adding specified users to Administrators (local machine)..." -ForegroundColor Cyan

function Add-ToLocalAdmins {
  param([Parameter(Mandatory)][string]$AccountName)
  
  if ($AccountName -like "MicrosoftAccount\*") {
    # Handle Microsoft Account differently
    $email = $AccountName -replace "MicrosoftAccount\\", ""
    Write-Host "  Attempting to add Microsoft Account: $email" -ForegroundColor Cyan
    
    # First try to add the Microsoft Account user to the system
    try {
      # Check if the user exists in the system
      $user = Get-LocalUser -Name $email -ErrorAction SilentlyContinue
      if (-not $user) {
        Write-Host "  '$email' not found on this machine." -ForegroundColor Yellow
        Write-Host "  Opening Settings -> Accounts -> Other users..." -ForegroundColor Yellow
        Start-Process "ms-settings:otherusers"
        Write-Host "  Add '$email' via 'Add someone else to this PC', then press Enter to continue..." -ForegroundColor Yellow
        Read-Host | Out-Null
      }
      
      # Now try to add to Administrators group
      $result = cmd /c "net localgroup administrators `"$email`" /add 2>&1"
      if ($LASTEXITCODE -eq 0) {
        Write-Host "  Added $email to Administrators." -ForegroundColor Green
      } elseif ($result -like "*already a member*" -or $result -like "*already exists*") {
        Write-Host "  $email already in Administrators." -ForegroundColor DarkGray
      } else {
        Write-Warning "  Could not add $email to Administrators: $result"
        Write-Host "  Suggestion: Link the Microsoft Account through Windows Settings first." -ForegroundColor Yellow
      }
    } catch {
      Write-Warning "  Error processing Microsoft Account '$email': $($_.Exception.Message)"
      Write-Host "  Suggestion: Link the Microsoft Account through Windows Settings first." -ForegroundColor Yellow
    }
  } else {
    # Handle regular local users
    try {
      Add-LocalGroupMember -Group "Administrators" -Member $AccountName -ErrorAction Stop
      Write-Host "  Added $AccountName to Administrators." -ForegroundColor Green
    } catch {
      if ($_.Exception.Message -like "*already a member*") {
        Write-Host "  $AccountName already in Administrators." -ForegroundColor DarkGray
      } else {
        Write-Warning "  Could not add ${AccountName}: $($_.Exception.Message)"
      }
    }
  }
}

# Add Microsoft account admins
Add-ToLocalAdmins -AccountName "MicrosoftAccount\mircean@outlook.com"
Add-ToLocalAdmins -AccountName "MicrosoftAccount\tate@nwfamilydental.net"

# Create and configure local user
$opUsername = "op"
$opPassword = "Family$"
$opUser = Get-LocalUser -Name $opUsername -ErrorAction SilentlyContinue
if (-not $opUser) {
  $securePassword = ConvertTo-SecureString $opPassword -AsPlainText -Force
  New-LocalUser -Name $opUsername -Password $securePassword -AccountNeverExpires `
    -UserMayNotChangePassword:$false -PasswordNeverExpires:$true -ErrorAction Stop
  Write-Host "  Created local user '$opUsername' with password '$opPassword'." -ForegroundColor Green
} else {
  Write-Host "  Local user '$opUsername' already exists." -ForegroundColor DarkGray
}

# Add local user to Administrators
Add-ToLocalAdmins -AccountName $opUsername

# --- 6) Enable Remote Desktop --------------------------------------------------
Write-Host "Enabling Remote Desktop and firewall rules..." -ForegroundColor Cyan
Set-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\Terminal Server" -Name "fDenyTSConnections" -Value 0
Enable-NetFirewallRule -DisplayGroup "Remote Desktop"

# --- 7) Startup script for IPC$ connections -----------------------------------
Write-Host "Creating startup cmd for persistent IPC$ connections..." -ForegroundColor Cyan
$startupAllUsers = "${env:ProgramData}\Microsoft\Windows\Start Menu\Programs\StartUp"
if (-not (Test-Path $startupAllUsers)) { New-Item -ItemType Directory -Path $startupAllUsers -Force | Out-Null }

$startupCmd = Join-Path $startupAllUsers 'startup.cmd'
$startupContent = @"
@echo off
REM Map IPC$ with saved credentials (prompts first run, then stored).
net use \\server\ipc$ /savecred /persistent:yes
"@.Trim()

$startupContent | Out-File -FilePath $startupCmd -Encoding ASCII -Force
Write-Host "Startup script created: $startupCmd" -ForegroundColor Green
Write-Host "Run once now to cache creds (optional): `"$startupCmd`"" -ForegroundColor DarkCyan

& $startupCmd

# --- summary -------------------------------------------------------------------
Write-Host "`nAll requested steps attempted." -ForegroundColor Green
