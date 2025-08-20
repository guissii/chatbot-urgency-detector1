# Chatbot Urgency Classifier

Système de classification d'urgence des conversations en temps réel utilisant le NLP et BERT.

## 🚀 Fonctionnalités

- Analyse sémantique des messages en temps réel
- Détection du niveau d'urgence (faible, moyen, élevé)
- Interface utilisateur intuitive avec Streamlit
- Modèle BERT fine-tuné pour la détection d'urgence

## 📦 Installation

1. Cloner le dépôt :
```bash
git clone [https://github.com/guissii/chatbot-urgency-detector1.git](https://github.com/guissii/chatbot-urgency-detector1.git)
cd chatbot-urgency-detector1
Créer un environnement virtuel :
bash
At mention
python -m venv .venv
.venv\Scripts\activate  # Windows
Installer les dépendances :
bash
At mention
pip install -r requirements.txt
🚀 Utilisation
Lancer l'application :
bash
At mention
streamlit run app.py
Accéder à l'interface :
At mention
http://localhost:8501
Entrer un message pour voir la prédiction d'urgence
🏗️ Structure du projet
At mention
chatbot-urgency-detector1/
├── data/               # Données d'entraînement et de test
├── models/             # Modèles entraînés (non suivi par git)
├── src/                # Code source
│   ├── train.py       # Script d'entraînement
│   └── predict.py     # Script de prédiction
├── app.py             # Application Streamlit
└── requirements.txt   # Dépendances
📝 Notes importantes
Les modèles entraînés ne sont pas inclus dans le dépôt en raison de leur taille
Consultez la documentation pour les instructions d'entraînement
