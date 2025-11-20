from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch, MagicMock
import threading
import numpy as np

# Importar funciones de views.py
from .views import analyze_emotion, average_emotions

class UnitTests(TestCase):
    def setUp(self):
        # Reiniciar variables globales para pruebas
        from .views import emotion_history, emotion_lock, emotion_detected, all_emotions
        with emotion_lock:
            emotion_history.clear()
            emotion_detected = "Analizando..."
            all_emotions = {}

    @patch('core.views.DeepFace.analyze')
    def test_analyze_emotion_success(self, mock_analyze):
        # Mock de DeepFace.analyze
        mock_analyze.return_value = {
            'emotion': {'happy': 80.0, 'sad': 10.0, 'neutral': 10.0},
            'dominant_emotion': 'happy'
        }

        # Crear un frame falso (imagen dummy)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Llamar a la función
        analyze_emotion(frame)

        # Verificar que se actualizó emotion_detected
        from .views import emotion_detected, all_emotions
        self.assertEqual(emotion_detected, 'Feliz')
        self.assertIn('Feliz', all_emotions)
        self.assertGreater(all_emotions['Feliz'], 0)

    @patch('core.views.DeepFace.analyze')
    def test_analyze_emotion_no_face(self, mock_analyze):
        # Mock para excepción
        mock_analyze.side_effect = Exception("No face detected")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        analyze_emotion(frame)

        from .views import emotion_detected, all_emotions
        self.assertEqual(emotion_detected, "No detectado")
        self.assertEqual(all_emotions, {})

    def test_average_emotions(self):
        from .views import emotion_history, emotion_lock
        # Agregar datos de prueba al historial
        with emotion_lock:
            emotion_history.append({'Feliz': 70.0, 'Triste': 20.0})
            emotion_history.append({'Feliz': 90.0, 'Triste': 10.0})

        avg = average_emotions()
        self.assertAlmostEqual(avg['Feliz'], 80.0)
        self.assertAlmostEqual(avg['Triste'], 15.0)

class IntegrationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_view(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/index.html')

    def test_video_feed_cam_view(self):
        # Nota: Esta vista retorna StreamingHttpResponse, pero podemos verificar el status
        response = self.client.get(reverse('video_feed_cam'))
        self.assertEqual(response.status_code, 200)
        # Verificar que es streaming
        self.assertEqual(response['Content-Type'], 'multipart/x-mixed-replace; boundary=frame')
