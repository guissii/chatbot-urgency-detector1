"""
Module principal pour le modèle de classification d'urgence.
"""
import os
import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertConfig, CamembertForSequenceClassification
from typing import Dict, List, Tuple, Optional, Union
import logging

from src.utils import get_device

logger = logging.getLogger(__name__)

class UrgencyClassifier(nn.Module):
    """
    Modèle de classification d'urgence basé sur CamemBERT.
    """
    
    def __init__(self, model_name: str = 'camembert-base', num_labels: int = 2, 
                 dropout: float = 0.1):
        """
        Initialise le classifieur d'urgence.
        
        Args:
            model_name: Nom du modèle CamemBERT pré-entraîné.
            num_labels: Nombre de classes de sortie.
            dropout: Taux de dropout à appliquer.
        """
        super(UrgencyClassifier, self).__init__()
        self.device = get_device()
        self.num_labels = num_labels
        
        # Chargement de la configuration de base de CamemBERT
        config = CamembertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=False
        )
        
        # Initialisation du modèle CamemBERT
        self.camembert = CamembertModel.from_pretrained(
            model_name,
            config=config
        )
        
        # Couches supplémentaires pour la classification
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.camembert.config.hidden_size,
            num_labels
        )
        
        # Initialisation des poids du classifieur
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.zero_()
        
        self.to(self.device)
    
    def forward(self, input_ids: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Passe avant du modèle.
        
        Args:
            input_ids: Identifiants des tokens d'entrée.
            attention_mask: Masque d'attention.
            labels: Labels pour le calcul de la perte.
            
        Returns:
            Un dictionnaire contenant les logits et éventuellement la perte.
        """
        outputs = self.camembert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Récupération du vecteur [CLS] pour la classification
        pooled_output = outputs[1]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        outputs = (logits,) + outputs[2:]  # Ajout des hidden_states si nécessaire
        
        # Calcul de la perte si des labels sont fournis
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs  # (loss), logits, (hidden_states)
    
    def predict_proba(self, input_ids: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Prédit les probabilités pour chaque classe.
        
        Args:
            input_ids: Identifiants des tokens d'entrée.
            attention_mask: Masque d'attention.
            
        Returns:
            Un tenseur de probabilités de forme [batch_size, num_labels].
        """
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask=attention_mask)
            logits = outputs[0]  # [batch_size, num_labels]
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def save_pretrained(self, output_dir: str) -> None:
        """
        Sauvegarde le modèle et le tokenizer.
        
        Args:
            output_dir: Répertoire de sortie.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde de la configuration du modèle
        model_config = {
            'model_name': 'camembert-base',
            'num_labels': self.num_labels,
            'dropout': self.dropout.p
        }
        
        # Sauvegarde des poids du modèle
        model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        # Sauvegarde de la configuration
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(model_config, f)
        
        logger.info(f"Modèle sauvegardé dans {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'UrgencyClassifier':
        """
        Charge un modèle pré-entraîné.
        
        Args:
            model_path: Chemin vers le modèle sauvegardé.
            
        Returns:
            Une instance de UrgencyClassifier chargée avec les poids sauvegardés.
        """
        # Vérification de l'existence des fichiers nécessaires
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le répertoire du modèle {model_path} n'existe pas.")
        
        config_path = os.path.join(model_path, 'config.json')
        model_weights_path = os.path.join(model_path, 'pytorch_model.bin')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
        
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Poids du modèle introuvables: {model_weights_path}")
        
        # Chargement de la configuration
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        
        # Initialisation du modèle
        model = cls(
            model_name=config.get('model_name', 'camembert-base'),
            num_labels=config.get('num_labels', 2),
            dropout=config.get('dropout', 0.1)
        )
        
        # Chargement des poids
        model.load_state_dict(torch.load(model_weights_path, map_location=model.device))
        model.to(model.device)
        
        logger.info(f"Modèle chargé depuis {model_path}")
        return model


class UrgencyPredictor:
    """
    Classe utilitaire pour effectuer des prédictions avec le modèle de classification d'urgence.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 model: Optional[UrgencyClassifier] = None,
                 tokenizer = None):
        """
        Initialise le prédicteur.
        
        Args:
            model_path: Chemin vers le modèle sauvegardé.
            model: Instance de UrgencyClassifier (optionnel).
            tokenizer: Tokenizer à utiliser (optionnel).
        """
        self.device = get_device()
        
        # Chargement du modèle
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = UrgencyClassifier.from_pretrained(model_path)
        else:
            raise ValueError("Soit model_path, soit model doit être spécifié.")
        
        # Initialisation du tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import CamembertTokenizer
            self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    def predict(self, text: str, return_prob: bool = False) -> Union[int, Tuple[int, float]]:
        """
        Prédit la classe d'un texte donné.
        
        Args:
            text: Texte à classifier.
            return_prob: Si True, retourne également la probabilité de la classe prédite.
            
        Returns:
            La classe prédite (0 ou 1) et éventuellement sa probabilité.
        """
        self.model.eval()
        
        # Tokenisation
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Déplacement sur le bon dispositif
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs[0]
            probs = torch.softmax(logits, dim=1)
            prob, pred = torch.max(probs, dim=1)
        
        if return_prob:
            return pred.item(), prob.item()
        return pred.item()
    
    def predict_batch(self, texts: List[str], batch_size: int = 16, 
                     show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit les classes pour un lot de textes.
        
        Args:
            texts: Liste de textes à classifier.
            batch_size: Taille des lots pour l'inférence.
            show_progress: Afficher une barre de progression.
            
        Returns:
            Un tuple (predictions, probabilities) contenant les prédictions et les probabilités.
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        # Création d'un itérateur avec barre de progression
        iter_texts = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iter_texts = tqdm(iter_texts, desc="Prédiction des lots")
        
        for i in iter_texts:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenisation par lots
            encoding = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Déplacement sur le bon dispositif
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Prédiction par lots
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs[0]
                probs = torch.softmax(logits, dim=1)
                batch_probs, batch_preds = torch.max(probs, dim=1)
            
            predictions.extend(batch_preds.cpu().numpy())
            probabilities.extend(batch_probs.cpu().numpy())
        
        return torch.tensor(predictions), torch.tensor(probabilities)
