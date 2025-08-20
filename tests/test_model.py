import unittest
import torch
import numpy as np
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from src.model import UrgencyPredictor

class TestUrgencyPredictor(unittest.TestCase):
    """Tests pour la classe UrgencyPredictor."""
    
    @classmethod
    def setUpClass(cls):
        """Initialisation avant tous les tests."""
        # Utilisation d'un petit modèle pour les tests
        cls.model_name = "camembert-base"
        cls.tokenizer = CamembertTokenizer.from_pretrained(cls.model_name)
        
        # Création d'un modèle factice pour les tests
        cls.model = CamembertForSequenceClassification.from_pretrained(
            cls.model_name, 
            num_labels=2
        )
        
        # Sauvegarder le modèle factice
        import tempfile
        import os
        cls.temp_dir = tempfile.mkdtemp()
        cls.model.save_pretrained(cls.temp_dir)
        cls.tokenizer.save_pretrained(cls.temp_dir)
        
        # Initialisation du prédicteur
        cls.predictor = UrgencyPredictor(cls.temp_dir)
    
    def test_predict_single_text(self):
        """Test de prédiction sur un seul texte."""
        text = "Ceci est un test d'urgence"
        prediction, confidence = self.predictor.predict(text)
        
        # Vérifier les types de retour
        self.assertIsInstance(prediction, int)
        self.assertIsInstance(confidence, float)
        
        # Vérifier les valeurs de sortie
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_predict_batch(self):
        """Test de prédiction par lots."""
        texts = [
            "Ceci est un test d'urgence",
            "Ceci est un autre test",
            "Urgence critique"
        ]
        
        predictions, confidences = self.predictor.predict_batch(texts)
        
        # Vérifier les types et tailles
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(confidences, list)
        self.assertEqual(len(predictions), len(texts))
        self.assertEqual(len(confidences), len(texts))
        
        # Vérifier chaque prédiction
        for pred, conf in zip(predictions, confidences):
            self.assertIn(pred, [0, 1])
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
    
    def test_get_priority(self):
        """Test de la méthode get_priority."""
        # Test avec une prédiction non urgente
        priority = self.predictor.get_priority(0.9, 0)
        self.assertEqual(priority, "Basse")
        
        # Test avec une priorité haute
        priority = self.predictor.get_priority(0.9, 1)
        self.assertEqual(priority, "Haute")
        
        # Test avec une priorité moyenne
        priority = self.predictor.get_priority(0.7, 1)
        self.assertEqual(priority, "Moyenne")
        
        # Test avec une priorité basse
        priority = self.predictor.get_priority(0.6, 1)
        self.assertEqual(priority, "Basse")
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        import shutil
        shutil.rmtree(cls.temp_dir)

if __name__ == '__main__':
    unittest.main()
