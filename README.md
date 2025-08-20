# Chatbot de Détection d'Urgence

Ce projet implémente un système intelligent de classification d'urgence basé sur des modèles de langue avancés (comme CamemBERT) pour analyser des messages et détecter les demandes urgentes.

## 🚀 Fonctionnalités

- Classification binaire des messages (Urgent/Non urgent)
- Interface en ligne de commande simple d'utilisation
- Support de plusieurs modes d'entrée :
  - Texte direct
  - Fichier texte avec un message par ligne
  - Mode interactif
  - Interface web en temps réel
- Modèle CamemBERT finetuné pour la classification de texte
- Interface utilisateur intuitive

## 📦 Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-utilisateur/chatbot_urgency_ethics.git
cd chatbot_urgency_ethics
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 🛠 Configuration

Créez un fichier `config.yaml` à la racine du projet avec la configuration suivante :

```yaml
model:
  model_name: "camembert-base"
  model_path: "./models/urgency_model"
  max_length: 256
  batch_size: 16

inference:
  high_priority_threshold: 0.85
  medium_priority_threshold: 0.65
  low_priority_threshold: 0.5

web:
  host: "0.0.0.0"
  port: 5000
  debug: true
```

## 🚀 Utilisation

### 1. Entraînement du modèle
```bash
python train.py --train_data data/raw/train.csv --val_data data/raw/eval.csv --epochs 5
```

### 2. Prédiction sur un fichier Excel
```bash
python predict_excel.py --input data/input/tickets.xlsx --output data/output/results.xlsx
```

### 3. Démarrer l'interface web
```bash
python app.py
```
Puis ouvrez http://localhost:5000 dans votre navigateur.

## 📁 Structure du projet

```
chatbot_urgency_ethics/
├── app.py                  # Application Flask
├── config.yaml             # Fichier de configuration
├── requirements.txt        # Dépendances
├── README.md               # Ce fichier
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Chargement des données
│   ├── model.py           # Modèle CamemBERT
│   ├── preprocess.py      # Prétraitement du texte
│   └── utils.py           # Fonctions utilitaires
├── static/                # Fichiers statiques (CSS, JS)
├── templates/             # Templates HTML
└── tests/                 # Tests unitaires
```

## 📊 Métriques de performance

Le modèle est évalué sur les métriques suivantes :
- Précision
- Rappel
- F1-score
- Matrice de confusion

## 📝 Licence

Ce projet est sous licence MIT.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.