@echo off
setlocal enabledelayedexpansion
rem Dense-Evolution Dashboard (Streamlit) -- uninstaller (Windows). Removes the
rem icons install-dashboard.bat created and the app folder. The dense-evolution
rem Python package is left installed unless you say yes below.

set "INSTALL_DIR=%USERPROFILE%\DenseEvolutionDashboard"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

echo Disinstallazione di Dense-Evolution Dashboard (Streamlit)
echo.

for %%L in (
    "%USERPROFILE%\Desktop\Dense-Evolution Dashboard (Streamlit).lnk"
    "%STARTMENU%\Dense-Evolution Dashboard (Streamlit).lnk"
) do (
    if exist %%L (
        del %%L
        echo Rimossa: %%L
    )
)

rem Termina eventuali processi che potrebbero bloccare i file nella cartella
echo Terminazione eventuali processi collegati...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*DenseEvolutionDashboard*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 2 /nobreak >nul

rem Rimuovi la cartella con tentativi multipli
set "MAX_RETRIES=5"
set /a RETRY=0

:REMOVE_DIR
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%" 2>nul
    if exist "%INSTALL_DIR%" (
        set /a RETRY+=1
        if !RETRY! LSS %MAX_RETRIES% (
            echo Tentativo !RETRY!/%MAX_RETRIES%: cartella ancora presente, attendo...
            timeout /t 2 /nobreak >nul
            goto REMOVE_DIR
        ) else (
            echo Errore: impossibile rimuovere "%INSTALL_DIR%" dopo %MAX_RETRIES% tentativi.
            exit /b 1
        )
    ) else (
        echo Cartella "%INSTALL_DIR%" rimossa.
    )
) else (
    echo Nessuna cartella trovata.
)
echo.

set "REMOVE_PKG=N"
set /p "REMOVE_PKG=Disinstallare anche il pacchetto Python dense-evolution? [s/N] "
if /i "!REMOVE_PKG!"=="s" (
    python -m pip uninstall -y dense-evolution
) else (
    echo Pacchetto dense-evolution lasciato installato.
)

echo.
echo Fatto.
pause
