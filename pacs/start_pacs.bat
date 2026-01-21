@echo off
REM ============================================================================
REM Script de démarrage rapide du serveur PACS Orthanc
REM ============================================================================

echo.
echo ========================================
echo   DEMARRAGE DU SERVEUR PACS ORTHANC
echo ========================================
echo.

REM Vérifier si Docker est installé
docker --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Docker n'est pas installe ou n'est pas dans le PATH.
    echo          Telechargez Docker Desktop depuis: https://www.docker.com
    pause
    exit /b 1
)

REM Vérifier si Docker est en cours d'exécution
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Docker n'est pas demarre.
    echo          Lancez Docker Desktop et reessayez.
    pause
    exit /b 1
)

echo [OK] Docker est installe et demarre.
echo.

REM Créer le dossier d'import si nécessaire
if not exist "dicom-import" mkdir dicom-import

REM Démarrer Orthanc
echo [INFO] Demarrage d'Orthanc...
docker-compose up -d

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   ORTHANC DEMARRE AVEC SUCCES !
    echo ========================================
    echo.
    echo   Interface Web: http://localhost:8042
    echo   Utilisateur:   admin
    echo   Mot de passe:  orthanc123
    echo.
    echo   Port DICOM: 4242
    echo   AET: RADIOPACS
    echo.
    
    REM Ouvrir le navigateur après un délai
    echo [INFO] Ouverture du navigateur dans 5 secondes...
    timeout /t 5 /nobreak > nul
    start http://localhost:8042
) else (
    echo [ERREUR] Echec du demarrage d'Orthanc.
    echo          Verifiez les logs: docker-compose logs
)

pause
