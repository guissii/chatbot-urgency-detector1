"""
Module contenant des fonctions utilitaires pour le projet.
"""
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import torch

# Chargement de la configuration
def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Charge la configuration à partir d'un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration.
        
    Returns:
        Un dictionnaire contenant la configuration.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Le fichier de configuration {config_path} n'a pas été trouvé.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> None:
    """
    Configure le système de logging.
    
    Args:
        log_dir: Répertoire pour les fichiers de log.
        log_level: Niveau de log.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Format des logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configuration du fichier de log
    log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configuration de base
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_device() -> torch.device:
    """
    Retourne le dispositif à utiliser (GPU si disponible, sinon CPU).
    
    Returns:
        Un objet torch.device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_confidence(confidence: float) -> str:
    """
    Formate un score de confiance en pourcentage.
    
    Args:
        confidence: Score de confiance entre 0 et 1.
        
    Returns:
        Une chaîne formatée en pourcentage avec 2 décimales.
    """
    return f"{confidence * 100:.2f}%"

def ensure_dir(directory: str) -> None:
    """
    Crée un répertoire s'il n'existe pas déjà.
    
    Args:
        directory: Chemin du répertoire à créer.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_timestamp() -> str:
    """
    Retourne un horodatage formaté.
    
    Returns:
        Une chaîne représentant la date et l'heure actuelles.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
