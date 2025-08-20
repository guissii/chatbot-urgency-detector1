import torch
import argparse
import json
from pathlib import Path
from models.urgency_classifier import UrgencyClassifier

class UrgencyPredictor:
    def __init__(self, model_path):
        """
        Initialise le prédicteur d'urgence.
        
        Args:
            model_path (str): Chemin vers le dossier contenant le modèle sauvegardé
        """
        self.classifier = UrgencyClassifier.load_model(model_path)
        self.class_names = self.classifier.class_names
    
    def predict(self, text):
        """
        Prédit si le texte est urgent ou non.
        
        Args:
            text (str): Le texte à analyser
            
        Returns:
            dict: Un dictionnaire contenant la prédiction et les probabilités
        """
        return self.classifier.predict(text)
    
    def predict_batch(self, texts):
        """
        Effectue des prédictions sur un lot de textes.
        
        Args:
            texts (list): Liste de textes à analyser
            
        Returns:
            list: Liste de dictionnaires contenant les prédictions
        """
        return [self.predict(text) for text in texts]
    
    def get_class_names(self):
        """
        Retourne les noms des classes.
        
        Returns:
            list: Liste des noms des classes
        """
        return self.class_names

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Prédire l\'urgence d\'un message')
    parser.add_argument('--model_dir', type=str, default='models/urgency_model',
                      help='Dossier contenant le modèle entraîné')
    parser.add_argument('--text', type=str, default=None,
                      help='Texte à analyser (optionnel)')
    parser.add_argument('--file', type=str, default=None,
                      help='Fichier texte contenant les messages à analyser (un par ligne)')
    parser.add_argument('--output', type=str, default=None,
                      help='Fichier de sortie pour sauvegarder les résultats (format JSON)')
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Erreur: Le dossier du modèle {model_dir} n'existe pas.")
        return
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_dir}...")
    predictor = UrgencyPredictor(model_dir)
    
    # Charger les textes à analyser
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {e}")
            return
    else:
        # Mode interactif
        print("Entrez vos messages (une ligne par message). Appuyez sur Entrée deux fois pour terminer:")
        while True:
            try:
                line = input("> ")
                if not line:
                    break
                texts.append(line)
            except EOFError:
                break
    
    if not texts:
        print("Aucun texte à analyser.")
        return
    
    # Faire les prédictions
    print("\nAnalyse en cours...\n")
    results = predictor.predict_batch(texts)
    
    # Afficher les résultats
    output_data = []
    
    for i, (text, result) in enumerate(zip(texts, results), 1):
        print(f"{i}. {text}")
        print(f"   Classe: {result['class_name']} (Confiance: {result['confidence']:.1%})")
        print(f"   Probabilités: {dict(zip(predictor.get_class_names(), [f'{p:.1%}' for p in result['probabilities']]))}")
        print()
        
        # Préparer les données pour la sortie JSON si nécessaire
        output_data.append({
            'text': text,
            'prediction': result['class_name'],
            'confidence': result['confidence'],
            'probabilities': dict(zip(predictor.get_class_names(), result['probabilities']))
        })
    
    # Sauvegarder les résultats si demandé
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nRésultats sauvegardés dans {args.output}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des résultats: {e}")

if __name__ == "__main__":
    main()
