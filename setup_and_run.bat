@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
color 0A
title A2SB Audio Restoration - Automatic Setup and Run

:: ============================================================================
:: A2SB Audio Restoration - Automatic Setup and Run Script
:: ============================================================================

:: IMPORTANT WARNING - Check if already in project folder
set "CURRENT_DIR=%~dp0"
if exist "%CURRENT_DIR%gradio_app.py" (
    :: Already in project folder, skip warning
    goto MENU
)

cls
echo.
echo ========================================================================
echo                             WARNING!
echo ========================================================================
echo.
echo   IMPORTANT: For first-time setup, place this BAT file in an EMPTY folder!
echo.
echo   This script will download the project and create multiple files
echo   and folders in the same directory where this BAT file is located.
echo.
echo   Recommended: Create a new empty folder and place this file there.
echo.
echo   Note: If you already have the project, place this BAT file inside
echo   the project folder and use option 2 or 3.
echo.
echo ========================================================================
echo.
set /p proceed="Do you want to continue? (Y/N): "

if /i NOT "%proceed%"=="Y" (
    echo.
    echo Setup cancelled.
    echo.
    timeout /t 3 >nul
    exit /b 0
)

:MENU
cls
echo.
echo ========================================================================
echo           A2SB Audio Restoration - Setup and Run
echo ========================================================================
echo.
echo.
echo   [1] Full Setup and Run (First Time)
echo       Download project, create venv, install dependencies and run
echo.
echo   [2] Run Only (Already Installed)
echo       Launch gradio_app.py with existing installation
echo.
echo   [3] Resume Installation (Continue from error/interruption)
echo       Check and complete any missing installation steps
echo.
echo   [4] Exit
echo.
echo ========================================================================
echo.
set /p choice="Your choice (1, 2, 3 or 4): "

if "%choice%"=="1" goto FULL_SETUP
if "%choice%"=="2" goto RUN_ONLY
if "%choice%"=="3" goto RESUME_SETUP
if "%choice%"=="4" goto EXIT
echo.
echo Invalid choice! Please enter 1, 2, 3 or 4.
timeout /t 2 >nul
goto MENU

:: ============================================================================
:: OPTION 1: FULL SETUP AND RUN
:: ============================================================================
:FULL_SETUP
cls
echo.
echo ========================================================================
echo                        STARTING FULL SETUP
echo ========================================================================
echo.

:: Python check
echo [1/7] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.10 or higher:
    echo https://www.python.org/downloads/
    echo.
    echo During installation, check "Add Python to PATH"!
    echo.
    pause
    goto MENU
)
python --version
echo Python found!
echo.

:: Git check
echo [2/7] Checking Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Git not found!
    echo.
    echo Please install Git:
    echo https://git-scm.com/download/win
    echo.
    pause
    goto MENU
)
git --version
echo Git found!
echo.

:: Project directory
set "PROJECT_DIR=%~dp0diffusion-audio-restoration-colab-Kaggle-\"
set "CURRENT_DIR=%~dp0"

:: If project folder already exists, ask
if exist "%PROJECT_DIR%" (
    echo.
    echo Project folder already exists: "%PROJECT_DIR%"
    echo.
    set /p overwrite="Delete and re-download? (Y/N): "
    if /i "!overwrite!"=="Y" (
        echo.
        echo Deleting existing folder...
        rd /s /q "%PROJECT_DIR%"
        echo Deleted!
    ) else (
        echo.
        echo Using existing folder.
        goto SKIP_CLONE
    )
)

:: Download project
echo [3/7] Downloading project from GitHub...
echo.
git clone https://github.com/test4373/diffusion-audio-restoration-colab-Kaggle-.git "%PROJECT_DIR%"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to download project!
    echo.
    pause
    goto MENU
)
echo.
echo Project downloaded successfully!
echo.

:SKIP_CLONE
:: Change to project directory
cd /d "%PROJECT_DIR%"

:: Create virtual environment (venv)
echo [4/7] Creating virtual environment (venv)...
echo.
if exist "venv" (
    echo venv folder already exists. Recreating...
    rd /s /q "venv"
)
python -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment!
    echo.
    pause
    goto MENU
)
echo Virtual environment created successfully!
echo.

:: Activate virtual environment
echo [5/7] Activating virtual environment...
call venv\Scripts\activate.bat
setlocal enabledelayedexpansion
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment!
    echo.
    pause
    goto MENU
)
echo Virtual environment active!
echo.

:: Upgrade pip
echo [6/7] Upgrading pip...
echo.
python -m pip install --upgrade pip
echo.
echo pip upgraded!
echo.

:: Install dependencies
echo [7/7] Installing dependencies (This may take a few minutes)...
echo.

:: Install PyTorch with CUDA (for faster inference)
echo Installing PyTorch CUDA version...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

:: Install other dependencies
echo Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed to install.
    echo Do you want to continue?
    echo.
    set /p continue="Continue? (Y/N): "
    if /i "!continue!" NEQ "Y" (
        pause
        goto MENU
    )
)
echo.
echo All dependencies installed!
echo.

:: Create output folder
if not exist "gradio_outputs" mkdir gradio_outputs

:: Model download
if not exist "ckpt" mkdir ckpt

set "MODEL1=ckpt\A2SB_onesplit_0.0_1.0_release.ckpt"
set "MODEL2=ckpt\A2SB_twosplit_0.5_1.0_release.ckpt"
set "MODEL1_URL=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt"
set "MODEL2_URL=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt"

set "NEED_DOWNLOAD=0"
if not exist "%MODEL1%" set "NEED_DOWNLOAD=1"
if not exist "%MODEL2%" set "NEED_DOWNLOAD=1"

if "!NEED_DOWNLOAD!"=="1" goto FULL_SETUP_DOWNLOAD_PROMPT
goto FULL_SETUP_MODELS_OK

:FULL_SETUP_DOWNLOAD_PROMPT
echo.
echo ========================================================================
echo                    MODEL FILES NOT FOUND
echo ========================================================================
echo.
echo AI models required! Each ~2.3 GB (Total ~4.6 GB)
echo.
echo Do you want to download automatically?
echo (Download may take 10-30 minutes, depending on your internet speed)
echo.
set /p "download_models=Download models? (Y/N): "

if /i "!download_models!"=="Y" (
    echo.
    echo Starting model download...
    call :DOWNLOAD_MODELS
    if errorlevel 1 (
        echo.
        echo Model download failed!
        echo.
        echo For manual download:
        echo 1. https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge
        echo 2. Save two .ckpt files to "ckpt" folder
        echo.
        pause
        goto MENU
    )
    echo.
    echo Model download completed successfully!
    goto FULL_SETUP_MODELS_OK
) else (
    echo.
    echo Model download cancelled.
    echo.
    echo For manual download:
    echo 1. https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge
    echo 2. Save two .ckpt files to "ckpt" folder
    echo.
    echo The program will not work without model files!
    echo.
    pause
    goto MENU
)

:FULL_SETUP_MODELS_OK
echo.
echo ========================================================================
echo.
echo Model files present!
echo.
echo ========================================================================
echo.
echo SETUP REPORT:
echo.
echo Python installation: OK
echo Git installation: OK
echo Project download: OK
echo Virtual environment (venv): OK
echo Dependencies: OK
echo Model files: OK
echo.
echo ========================================================================
echo.
echo SETUP COMPLETED!
echo.
echo Now launching Gradio web interface...
echo.
timeout /t 3 >nul

goto RUN_APP

:: ============================================================================
:: OPTION 3: RESUME INSTALLATION
:: ============================================================================
:RESUME_SETUP
cls
echo.
echo ========================================================================
echo                    RESUMING INSTALLATION
echo ========================================================================
echo.
echo Checking installation status and completing missing steps...
echo.

:: Detect project directory (either current dir or subdirectory)
set "CURRENT_DIR=%~dp0"
set "PROJECT_DIR=%~dp0diffusion-audio-restoration-colab-Kaggle-\"

:: Step 1: Check if project exists
echo [Step 1/6] Checking project folder...

:: Check if we're already in the project folder
if exist "%CURRENT_DIR%gradio_app.py" (
    set "PROJECT_DIR=%CURRENT_DIR%"
    echo Already in project folder!
    cd /d "!PROJECT_DIR!"
    goto CONTINUE_RESUME
)

if exist "%PROJECT_DIR%gradio_app.py" (
    echo Project folder found in subdirectory!
    cd /d "%PROJECT_DIR%"
    goto CONTINUE_RESUME
)

:: If we get here, need to download
echo Project folder not found. Downloading from GitHub...
echo.

:: Git check
git --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Git not found! Please install Git first.
    echo https://git-scm.com/download/win
    echo.
    pause
    goto MENU
)

git clone https://github.com/test4373/diffusion-audio-restoration-colab-Kaggle-.git "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to download project!
    pause
    goto MENU
)
echo Project downloaded successfully!
cd /d "%PROJECT_DIR%"

:CONTINUE_RESUME
echo.

:: Step 2: Check Python
echo [Step 2/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+
    echo https://www.python.org/downloads/
    pause
    goto MENU
)
python --version
echo Python OK!
echo.

:: Step 3: Check/Create venv
echo [Step 3/6] Checking virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        goto MENU
    )
    echo Virtual environment created!
) else (
    echo Virtual environment found!
)
echo.

:: Activate venv
echo [Step 4/6] Activating virtual environment...
call venv\Scripts\activate.bat
setlocal enabledelayedexpansion
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    goto MENU
)
echo Virtual environment active!
echo.

:: Step 5: Check dependencies
echo [Step 5/6] Checking dependencies...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo PyTorch not found. Installing dependencies...
    echo.
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo.
    echo Installing PyTorch CUDA version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo.
    echo Installing other dependencies...
    pip install -r requirements.txt
    echo.
    echo Dependencies installed!
) else (
    echo PyTorch found! Verifying all dependencies...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        pause
        goto MENU
    )
    echo.
    echo All dependencies verified and updated!
)
echo.

:: Create output folder
if not exist "gradio_outputs" mkdir gradio_outputs

:: Step 5: Check models
echo [Step 6/6] Checking model files...
if not exist "ckpt" mkdir ckpt

set "MODEL1=ckpt\A2SB_onesplit_0.0_1.0_release.ckpt"
set "MODEL2=ckpt\A2SB_twosplit_0.5_1.0_release.ckpt"
set "MODEL1_URL=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt"
set "MODEL2_URL=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt"

set "NEED_DOWNLOAD=0"
if not exist "%MODEL1%" set "NEED_DOWNLOAD=1"
if not exist "%MODEL2%" set "NEED_DOWNLOAD=1"

if "!NEED_DOWNLOAD!"=="1" goto RESUME_SETUP_DOWNLOAD_PROMPT
goto RESUME_SETUP_MODELS_OK

:RESUME_SETUP_DOWNLOAD_PROMPT
echo.
echo Model files not found!
echo.
set /p "download_models=Download models now? (Y/N): "

if /i "!download_models!"=="Y" (
    echo.
    echo Starting model download...
    call :DOWNLOAD_MODELS
    if errorlevel 1 (
        echo.
        echo Model download failed!
        pause
        goto MENU
    )
    echo.
    echo Model download completed successfully!
    goto RESUME_SETUP_MODELS_OK
) else (
    echo.
    echo WARNING: The program will not work without model files!
    echo You can download them later or manually.
    echo.
    pause
    goto MENU
)

:RESUME_SETUP_MODELS_OK
echo Model files found!
echo.

echo ========================================================================
echo.
echo RESUME COMPLETED!
echo.
echo All installation steps verified and completed.
echo.
echo ========================================================================
echo.
timeout /t 2 >nul

goto RUN_APP

:: ============================================================================
:: OPTION 2: RUN ONLY
:: ============================================================================
:RUN_ONLY
cls
echo.
echo ========================================================================
echo                     LAUNCHING APPLICATION
echo ========================================================================
echo.

set "CURRENT_DIR=%~dp0"
set "PROJECT_DIR=%~dp0diffusion-audio-restoration-colab-Kaggle-\"

if exist "%CURRENT_DIR%gradio_app.py" (
    set "PROJECT_DIR=%CURRENT_DIR%"
) else if not exist "%PROJECT_DIR%gradio_app.py" (
    echo ERROR: Project folder not found!
    pause
    goto MENU
)

if not exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    pause
    goto MENU
)

cd /d "%PROJECT_DIR%"
call "%PROJECT_DIR%venv\Scripts\activate.bat"
setlocal enabledelayedexpansion

set "CKPT_DIR=%PROJECT_DIR%ckpt"
set "MODEL1=%CKPT_DIR%\A2SB_onesplit_0.0_1.0_release.ckpt"
set "MODEL2=%CKPT_DIR%\A2SB_twosplit_0.5_1.0_release.ckpt"
set "MODEL1_URL=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt"
set "MODEL2_URL=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt"

if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"

set "NEED_DOWNLOAD=0"
if not exist "%MODEL1%" set "NEED_DOWNLOAD=1"
if not exist "%MODEL2%" set "NEED_DOWNLOAD=1"

if "!NEED_DOWNLOAD!"=="1" goto MODEL_DOWNLOAD_PROMPT
goto MODELS_FOUND

:MODEL_DOWNLOAD_PROMPT
echo.
echo ========================================================================
echo                    MODEL FILES NOT FOUND!
echo ========================================================================
echo.
echo AI models required! Each ~2.3 GB (Total ~4.6 GB)
echo.
echo Do you want to download automatically?
echo.
set /p "download_models=Download models? (Y/N): "

if /i "!download_models!"=="Y" (
    echo.
    echo Starting model download...
    call :DOWNLOAD_MODELS
    if errorlevel 1 (
        echo.
        echo Model download failed!
        echo.
        pause
        goto MENU
    )
    echo.
    echo Model download completed successfully!
    goto MODELS_FOUND
) else (
    echo.
    echo The program will not work without model files!
    echo.
    pause
    goto MENU
)

:MODELS_FOUND
echo.
echo Model files found! Starting application...
echo.

:RUN_APP
echo.
echo Checking GPU...
python -c "import torch; print('GPU Found:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('GPU not found, will run in CPU mode')"
echo.
echo ========================================================================
echo   Launching web interface...
echo   Browser will open automatically. If not, click the URL shown below.
echo   TIP: Hold CTRL and left-click the URL to open in browser
echo ========================================================================
echo.

python gradio_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch application!
    echo.
    echo Possible reasons:
    echo - Model files missing (ckpt folder must contain .ckpt files)
    echo - Dependencies missing (Run full setup)
    echo - Python version incompatible (3.10+ required)
    echo.
    pause
    goto MENU
)

:: Application closed
echo.
echo.
echo ========================================================================
echo.
echo Application closed.
echo.
set /p restart="Do you want to restart? (Y/N): "
if /i "%restart%"=="Y" goto RUN_ONLY
goto MENU

:: ============================================================================
:: MODEL DOWNLOAD FUNCTION
:: ============================================================================
:DOWNLOAD_MODELS
echo.
echo ========================================================================
echo                    DOWNLOADING MODEL FILES
echo ========================================================================
echo.
echo Total size: ~4.6 GB
echo This may take 10-30 minutes...
echo.

:: Check curl
curl --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: curl not found!
    echo.
    echo curl is included by default in Windows 10/11.
    echo For older Windows versions, you need to download manually.
    echo.
    pause
    exit /b 1
)

:: Download Model 1
if not exist "%MODEL1%" (
    echo.
    echo [1/2] Downloading Model 1... (~2.3 GB)
    echo File: A2SB_onesplit_0.0_1.0_release.ckpt
    echo.
    
    curl -L --progress-bar -o "%MODEL1%" "%MODEL1_URL%"
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download Model 1!
        echo.
        echo Download manually:
        echo %MODEL1_URL%
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo Model 1 downloaded successfully!
) else (
    echo [1/2] Model 1 already exists
)

:: Download Model 2
if not exist "%MODEL2%" (
    echo.
    echo [2/2] Downloading Model 2... (~2.3 GB)
    echo File: A2SB_twosplit_0.5_1.0_release.ckpt
    echo.
    
    curl -L --progress-bar -o "%MODEL2%" "%MODEL2_URL%"
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download Model 2!
        echo.
        echo Download manually:
        echo %MODEL2_URL%
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo Model 2 downloaded successfully!
) else (
    echo [2/2] Model 2 already exists
)

echo.
echo ========================================================================
echo.
echo ALL MODELS DOWNLOADED SUCCESSFULLY!
echo.
echo Model 1: %MODEL1%
echo Model 2: %MODEL2%
echo.
echo ========================================================================
echo.

exit /b 0

:: ============================================================================
:: EXIT
:: ============================================================================
:EXIT
cls
echo.
echo ========================================================================
echo.
echo Exiting A2SB Audio Restoration...
echo.
echo Thank you!
echo.
echo ========================================================================
echo.
timeout /t 2 >nul
exit /b 0
