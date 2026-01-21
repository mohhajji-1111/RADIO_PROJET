@echo off
REM ============================================================================
REM Script pour tester la migration DICOM avec serveur PACS
REM ============================================================================

echo.
echo ========================================================================
echo    TEST MIGRATION DICOM - RADIO_PROJET
echo ========================================================================
echo.

REM Activer l'environnement conda
call conda activate .conda
if %errorlevel% neq 0 (
    echo ERREUR: Impossible d'activer l'environnement conda
    pause
    exit /b 1
)

echo.
echo Options:
echo 1. Setup serveur Orthanc PACS
echo 2. Tester la connexion PACS
echo 3. Test complet de migration
echo 4. Test rapide (connexion seulement)
echo 5. Quitter
echo.

set /p choice="Choisissez une option (1-5): "

if "%choice%"=="1" (
    echo.
    echo Lancement du setup Orthanc...
    python setup_orthanc_server.py
) else if "%choice%"=="2" (
    echo.
    echo Test de connexion PACS...
    python test_pacs_connection.py
) else if "%choice%"=="3" (
    echo.
    echo Lancement du test complet de migration...
    python test_dicom_migration.py
) else if "%choice%"=="4" (
    echo.
    echo Test rapide...
    python test_dicom_migration.py --quick
) else if "%choice%"=="5" (
    echo Au revoir!
    exit /b 0
) else (
    echo Option invalide!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Test termine!
echo ========================================================================
echo.

pause
