@echo off
REM ============================================================================
REM Script d'arrêt du serveur PACS Orthanc
REM ============================================================================

echo.
echo ========================================
echo   ARRET DU SERVEUR PACS ORTHANC
echo ========================================
echo.

docker-compose down

if %errorlevel% equ 0 (
    echo.
    echo [OK] Orthanc arrete avec succes.
    echo     Les donnees sont conservees dans le volume Docker.
) else (
    echo [ERREUR] Probleme lors de l'arret d'Orthanc.
)

pause
