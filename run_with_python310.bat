@echo off
echo Vérification de Python 3.10...
python3.10 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.10 n'est pas trouvé dans le PATH.
    echo Veuillez installer Python 3.10 ou mettre à jour le PATH.
    pause
    exit /b 1
)

echo Lancement de l'entraînement avec Python 3.10...
python3.10 -u train_improved.py > training_log.txt 2>&1

if %ERRORLEVEL% EQU 0 (
    echo L'entraînement s'est terminé avec succès.
) else (
    echo Une erreur s'est produite pendant l'entraînement.
)

echo Consultez le fichier training_log.txt pour les détails.
pause
