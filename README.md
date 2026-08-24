# PneumoVision

## Environment Setup

### Requirements

- Python 3.11.x
- Git
- Windows PowerShell, macOS, or Linux

The project was verified with Python 3.11.9.

### Windows PowerShell

```powershell
git clone https://github.com/Ussefemad1/PneumoVision
cd PneumoVision

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts\verify_environment.py