@echo off
echo Démarrage de l'entraînement...
python -u train_improved.py > training_log.txt 2>&1
echo Fin de l'exécution. Vérifiez le fichier training_log.txt pour les détails.
