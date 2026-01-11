@echo off
REM =============================================================================
REM Script de lancement rapide - Training NSCLC
REM Double-cliquez sur ce fichier pour lancer le training
REM =============================================================================

echo.
echo ========================================
echo    NSCLC Multi-Organ Segmentation
echo ========================================
echo.

REM Fix OpenMP
set KMP_DUPLICATE_LIB_OK=TRUE

REM Activer conda (modifiez le chemin si nécessaire)
call conda activate radio_env 2>nul
if errorlevel 1 (
    echo Environnement 'radio_env' non trouve.
    echo Essai avec l'environnement par defaut...
)

REM Lancer le training
echo Lancement du training...
echo.
python incremental_training.py

echo.
echo Training termine ou interrompu.
pause
