import os
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement de la configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class UrgencyPredictor:
    """Classe pour effectuer des prédictions avec le modèle entraîné."""
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du dispositif: {self.device}")
        
        # Chargement du tokenizer et du modèle
        self.tokenizer = CamembertTokenizer.from_pretrained(model_path)
        self.model = CamembertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Seuils de priorité
        self.high_threshold = config['inference']['high_priority_threshold']
        self.medium_threshold = config['inference']['medium_priority_threshold']
        self.low_threshold = config['inference']['low_priority_threshold']
    
    def predict_batch(self, texts, batch_size=16):
        """Effectue des prédictions par lots."""
        predictions = []
        confidences = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Prédiction"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenisation
            inputs = self.tokenizer(
                batch_texts,
                max_length=config['model']['max_length'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                batch_confs = torch.max(probs, dim=1)[0].cpu().numpy()
                
                predictions.extend(batch_preds)
                confidences.extend(batch_confs)
        
        return predictions, confidences
    
    def get_priority(self, confidence, prediction):
        """Détermine la priorité en fonction du score de confiance."""
        if prediction == 0:  # Non urgent
            return "Basse"
            
        if confidence >= self.high_threshold:
            return "Haute"
        elif confidence >= self.medium_threshold:
            return "Moyenne"
        else:
            return "Basse"

def process_excel(input_path, output_path):
    """Traite un fichier Excel et génère les prédictions."""
    # Vérification des fichiers
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier d'entrée {input_path} n'existe pas.")
    
    # Création du répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Chargement des données
    logger.info(f"Chargement du fichier {input_path}...")
    df = pd.read_excel(input_path)
    
    # Vérification des colonnes requises
    if 'text' not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne 'text'.")
    
    # Initialisation du prédicteur
    predictor = UrgencyPredictor(config['model']['model_path'])
    
    # Prédictions
    logger.info("Début des prédictions...")
    texts = df['text'].astype(str).tolist()
    predictions, confidences = predictor.predict_batch(texts)
    
    # Ajout des résultats au DataFrame
    df['prediction'] = predictions
    df['confidence'] = confidences
    df['priority'] = [
        predictor.get_priority(conf, pred) 
        for conf, pred in zip(confidences, predictions)
    ]
    
    # Sauvegarde des résultats
    logger.info(f"Sauvegarde des résultats dans {output_path}")
    df.to_excel(output_path, index=False)
    
    # Affichage des statistiques
    stats = df['priority'].value_counts().to_dict()
    logger.info("Statistiques des prédictions:")
    for priority, count in stats.items():
        logger.info(f"- {priority}: {count} tickets")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prédiction d'urgence sur un fichier Excel")
    parser.add_argument('--input', type=str, required=True, 
                       help="Chemin vers le fichier Excel d'entrée")
    parser.add_argument('--output', type=str, required=True,
                       help="Chemin vers le fichier Excel de sortie")
    
    args = parser.parse_args()
    
    try:
        process_excel(args.input, args.output)
        logger.info("Traitement terminé avec succès!")
    except Exception as e:
        logger.error(f"Une erreur s'est produite: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
