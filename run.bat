@echo off
echo Paleidžiama Random Forest trūkstamų reikšmių užpildymo aplikacija...
echo.
echo Tikrinama Python instaliacija...
python --version
if %errorlevel% neq 0 (
    echo KLAIDA: Python nerastas sistemoje!
    echo Prašome įdiegti Python iš https://python.org
    pause
    exit /b 1
)

echo.
echo Diegiamos priklausomybės...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo KLAIDA: Nepavyko įdiegti priklausomybių!
    pause
    exit /b 1
)

echo.
echo Paleidžiama Flask aplikacija...
echo Aplikacija bus prieinama adresu: http://localhost:5000
echo Norėdami sustabdyti aplikaciją, paspauskite Ctrl+C
echo.
python app.py
pause