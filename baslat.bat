@echo off
rem Garaj demo - Windows baslatici (Sanal ortamsiz / Sistem Python)
setlocal
cd /d "%~dp0"

if not exist ".env" (
  echo baslat.bat: .env bulunamadi. Once .env.example dosyasini .env olarak kopyalayip doldurun.
  pause
  exit /b 2
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
  if not "%%B"=="" set "%%A=%%B"
)

echo Sistem Python ile demo baslatiliyor...
python pipeline\run_demo.py serve --insecure-local-http --allow-shutdown --open-browser %*

:: Hata alinsa bile pencerenin hemen kapanmamasi icin pause ekledik:
if errorlevel 1 (
    echo.
    echo Bir hata olustu. Lutfen yukaridaki hata mesajini kontrol edin.
    pause
)

endlocal