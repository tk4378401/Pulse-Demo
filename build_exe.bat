@echo off
REM build_exe.bat - आयुर्वेदिक नाड़ी परीक्षण EXE बिल्डर
REM Build script for Ayurvedic Nadi Pariksha EXE

echo ========================================
echo Ayurvedic Nadi Pariksha DSP Sandbox
echo EXE Build Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo [Step 1/4] Checking Python installation...
python -c "import sys; print(f'Python {sys.version}')"
echo.

REM Install PyInstaller if not already installed
echo [Step 2/4] Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)
echo PyInstaller installed successfully
echo.

REM Install required dependencies
echo [Step 3/4] Installing dependencies...
pip install numpy scipy PyQt6 pyqtgraph --quiet
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install
)
echo Dependencies installed
echo.

REM Build EXE using spec file
echo [Step 4/4] Building EXE file...
echo This may take several minutes...
echo.

pyinstaller --clean nadi.spec

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Build failed!
    echo ========================================
    echo Common issues:
    echo - Missing Visual Studio C++ Build Tools
    echo - Insufficient disk space
    echo - Antivirus blocking build process
    echo.
    echo Try running as Administrator
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo Your EXE file is located at:
echo dist\AyurvedicNadiPariksha\AyurvedicNadiPariksha.exe
echo.
echo You can distribute this folder to users.
echo They need to install these prerequisites:
echo   - Python 3.10+ runtime (if not bundled)
echo   - Or use the standalone .exe from dist folder
echo.
pause
