@echo off
rem Garaj demo - Windows baslatici.
rem .env dosyasini yukler, sunucuyu yerel modda baslatir ve tarayiciyi acar.
rem Sayfadaki "Oturumu kapat" dugmesi sunucuyu durdurur; bu pencere de kendiliginden kapanir.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
if not exist ".env" (
  echo baslat.bat: .env bulunamadi. Once .env.example dosyasini .env olarak kopyalayip doldurun.
  pause
  exit /b 2
)
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
  if not "%%B"=="" set "%%A=%%B"
)
python pipeline\run_demo.py serve --insecure-local-http --allow-shutdown --open-browser %*
if errorlevel 1 pause
endlocal
