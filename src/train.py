import os
import argparse
import pandas as pd
import logging
from pathlib import Path
from models.urgency_classifier import UrgencyClassifier

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Entraînement du modèle de détection d\'urgence')
    parser.add_argument('--data_path', type=str, default='data/raw/train.csv',
                      help='Chemin vers le fichier de données d\'entraînement')
    parser.add_argument('--output_dir', type=str, default='models/urgency_model',
                      help='Dossier de sortie pour sauvegarder le modèle')
    parser.add_argument('--model_name', type=str, default='camembert-base',
                      help='Nom du modèle pré-entraîné à utiliser')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Taille du batch pour l\'entraînement')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Longueur maximale des séquences de texte')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Taux d\'apprentissage')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion des données à utiliser pour le test')
    parser.add_argument('--seed', type=int, default=42,
                      help='Graine aléatoire pour la reproductibilité')
    return parser.parse_args()

def main():
    # Parser les arguments
    args = parse_args()
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initialisation du classifieur...")
    classifier = UrgencyClassifier(model_name=args.model_name)
    
    # Charger et préparer les données
    logger.info(f"Chargement des données depuis {args.data_path}...")
    train_df, val_df = classifier.load_data(
        args.data_path,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    logger.info(f"Taille de l'ensemble d'entraînement: {len(train_df)}")
    logger.info(f"Taille de l'ensemble de validation: {len(val_df)}")
    
    # Créer les data loaders
    logger.info("Préparation des données...")
    train_loader = classifier.create_data_loader(
        train_df,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    val_loader = classifier.create_data_loader(
        val_df,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Entraîner le modèle
    logger.info("Début de l'entraînement...")
    classifier.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Sauvegarder le modèle final
    logger.info(f"Sauvegarde du modèle dans {output_dir}...")
    classifier.save_model(output_dir)
    
    # Évaluer le modèle final
    logger.info("\nÉvaluation sur l'ensemble de validation:")
    metrics = classifier.evaluate(val_loader)
    
    # Sauvegarder les métriques
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1']),
            'loss': float(metrics['loss'])
        }, f, indent=2)
    
    logger.info(f"\nRapport de classification:\n{metrics['report']}")
    logger.info(f"\nPrécision: {metrics['accuracy']:.4f}")
    logger.info(f"F1-score: {metrics['f1']:.4f}")
    logger.info(f"Perte: {metrics['loss']:.4f}")
    logger.info(f"\nModèle sauvegardé dans {output_dir.absolute()}")

if __name__ == "__main__":
    main()
