import unittest
import json
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

class TestUrgencyAPI(unittest.TestCase):
    """Tests pour l'API de prédiction d'urgence."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour les tests."""
        # Créer un répertoire temporaire pour les tests
        cls.temp_dir = tempfile.mkdtemp()
        
        # Créer un modèle factice
        os.makedirs(os.path.join(cls.temp_dir, 'model'), exist_ok=True)
        
        # Sauvegarder une configuration de test
        cls.config = {
            'model': {
                'model_name': 'camembert-base',
                'model_path': os.path.join(cls.temp_dir, 'model'),
                'max_length': 128,
                'batch_size': 16,
                'num_labels': 2
            },
            'inference': {
                'high_priority_threshold': 0.85,
                'medium_priority_threshold': 0.65,
                'low_priority_threshold': 0.5
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': True,
                'secret_key': 'test_secret_key'
            },
            'data': {
                'output_dir': os.path.join(cls.temp_dir, 'output')
            }
        }
        
        # Sauvegarder la configuration
        import yaml
        with open(os.path.join(cls.temp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cls.config, f)
        
        # Importer l'application après avoir défini la configuration
        os.environ['CONFIG_PATH'] = os.path.join(cls.temp_dir, 'config.yaml')
        
        # Mock du modèle
        with patch('transformers.CamembertForSequenceClassification.from_pretrained') as mock_model:
            mock_model.return_value = MagicMock()
            from app import app as test_app
            cls.app = test_app.test_client()
            cls.app.testing = True
    
    def test_predict_endpoint(self):
        """Test du point de terminaison de prédiction."""
        # Données de test
        test_data = {'text': 'Ceci est un test d\'urgence'}
        
        # Mock de la prédiction
        with patch('app.predict_urgency') as mock_predict:
            mock_predict.return_value = {
                'prediction': 1,
                'confidence': 0.95,
                'priority': 'Haute',
                'error': None
            }
            
            # Effectuer la requête
            response = self.app.post(
                '/predict',
                data=json.dumps(test_data),
                content_type='application/json'
            )
            
            # Vérifier la réponse
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertEqual(data['result']['is_urgent'], True)
            self.assertEqual(data['result']['priority'], 'Haute')
    
    def test_empty_text(self):
        """Test avec un texte vide."""
        response = self.app.post(
            '/predict',
            data=json.dumps({'text': ''}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Aucun texte fourni', data['error'])
    
    @patch('pandas.read_excel')
    def test_batch_predict(self, mock_read_excel):
        """Test du traitement par lots."""
        # Données factices
        mock_df = MagicMock()
        mock_df.columns = ['text']
        mock_df.__getitem__.return_value = ['Test 1', 'Test 2']
        mock_read_excel.return_value = mock_df
        
        # Créer un fichier de test
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            test_file_path = tmp.name
        
        try:
            # Mock de la prédiction
            with patch('app.predict_urgency') as mock_predict:
                mock_predict.return_value = {
                    'prediction': 1,
                    'confidence': 0.9,
                    'priority': 'Haute',
                    'error': None
                }
                
                # Effectuer la requête
                with open(test_file_path, 'rb') as f:
                    response = self.app.post(
                        '/batch_predict',
                        data={'file': (f, 'test.xlsx')},
                        content_type='multipart/form-data'
                    )
                
                # Vérifier la réponse
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                self.assertTrue(data['success'])
                self.assertEqual(data['stats']['total'], 2)
                self.assertEqual(len(data['sample_results']), 2)
                
        finally:
            # Nettoyage
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        shutil.rmtree(cls.temp_dir)

if __name__ == '__main__':
    unittest.main()
