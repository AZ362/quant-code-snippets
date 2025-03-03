@echo off
REM Log starting script
echo Starting supervisord batch script...

REM Step 1: Activate the Conda environment
CALL "C:\ProgramData\anaconda3\condabin\conda.bat" activate py3_10
if errorlevel 1 (
    echo Error: Failed to activate Conda environment.
    pause
    exit /b 1
) else (
    echo Conda environment activated successfully.
)

REM Step 2: Change directory to the bot location
cd C:\bot_MRS_4.2
if errorlevel 1 (
    echo Error: Failed to change directory.
    pause
    exit /b 1
) else (
    echo Directory changed successfully.
)

REM Step 3: Start supervisord with the correct configuration file path
supervisord -c C:\Users\Administrator\Desktop\MRS_run\supervisord_MRS.conf
if errorlevel 1 (
    echo Error: Failed to start supervisord.
    pause
    exit /b 1
) else (
    echo supervisord started successfully.
)

echo Script completed successfully.
pause