"""
Module de prétraitement des données pour la classification d'urgence.
"""
import re
import string
from typing import List, Tuple, Optional
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """Classe pour le prétraitement des textes."""
    
    def __init__(self, language: str = 'french'):
        """
        Initialise le prétraitement.
        
        Args:
            language: Langue pour le prétraitement (par défaut: 'french').
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)
        
        # Expressions régulières pour le nettoyage
        self.url_regex = r'http\S+|www\.\S+'
        self.email_regex = r'\S+@\S+\.[a-zA-Z]+'
        self.punctuation = string.punctuation + '«»“”‘’–—'
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte en supprimant les caractères indésirables.
        
        Args:
            text: Texte à nettoyer.
            
        Returns:
            Le texte nettoyé.
        """
        if not isinstance(text, str):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des URLs
        text = re.sub(self.url_regex, '', text)
        
        # Suppression des emails
        text = re.sub(self.email_regex, '', text)
        
        # Suppression de la ponctuation
        text = text.translate(str.maketrans('', '', self.punctuation))
        
        # Suppression des chiffres
        text = re.sub(r'\d+', '', text)
        
        # Suppression des espaces multiples
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenise le texte.
        
        Args:
            text: Texte à tokeniser.
            
        Returns:
            Liste de tokens.
        """
        return word_tokenize(text, language=self.language)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Supprime les mots vides (stop words) des tokens.
        
        Args:
            tokens: Liste de tokens.
            
        Returns:
            Liste de tokens sans les mots vides.
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Applique le stemming sur les tokens.
        
        Args:
            tokens: Liste de tokens.
            
        Returns:
            Liste de tokens avec stemming appliqué.
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       apply_stemming: bool = False) -> str:
        """
        Prétraite un texte complet.
        
        Args:
            text: Texte à prétraiter.
            remove_stopwords: Si True, supprime les mots vides.
            apply_stemming: Si True, applique le stemming.
            
        Returns:
            Le texte prétraité.
        """
        # Nettoyage de base
        text = self.clean_text(text)
        
        # Tokenisation
        tokens = self.tokenize(text)
        
        # Suppression des mots vides si demandé
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Application du stemming si demandé
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        
        return ' '.join(tokens)

def load_and_preprocess_data(file_path: str, text_column: str = 'text', 
                           label_column: str = 'label', 
                           preprocessor: Optional[TextPreprocessor] = None) -> Tuple[List[str], List[int]]:
    """
    Charge et prétraite les données d'un fichier CSV ou Excel.
    
    Args:
        file_path: Chemin vers le fichier de données.
        text_column: Nom de la colonne contenant le texte.
        label_column: Nom de la colonne contenant les labels.
        preprocessor: Instance de TextPreprocessor à utiliser.
        
    Returns:
        Un tuple (texts, labels) contenant les textes prétraités et les labels.
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    # Chargement des données
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    # Vérification des colonnes
    if text_column not in df.columns:
        raise ValueError(f"La colonne '{text_column}' est introuvable dans le fichier.")
    
    if label_column not in df.columns:
        raise ValueError(f"La colonne '{label_column}' est introuvable dans le fichier.")
    
    # Prétraitement des textes
    texts = df[text_column].astype(str).tolist()
    texts = [preprocessor.preprocess_text(text) for text in texts]
    
    # Extraction des labels
    labels = df[label_column].astype(int).tolist()
    
    return texts, labels
