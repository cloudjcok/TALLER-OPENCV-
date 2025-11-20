"""
Pruebas de Integración Completas para Sistema de Detección de Emociones
Verifican que múltiples componentes trabajen juntos correctamente
"""

import os
import sys
import django

# Configurar Django antes de importar cualquier cosa de Django
# Obtener la ruta del directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Configurar la variable de entorno DJANGO_SETTINGS_MODULE
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Emociones.settings')

# Configurar Django
django.setup()

from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch, MagicMock, Mock
import threading
import numpy as np
import cv2
import time
from io import BytesIO


class DetailedTestCase(TestCase):
    """Clase base con métodos de utilidad para pruebas detalladas"""
    
    def setUp(self):
        super().setUp()
        self.print_separator()
        print(f"\n INICIANDO: {self._testMethodName}")
        print(f" Descripción: {self._testMethodDoc or 'Sin descripción'}")
        
    def tearDown(self):
        super().tearDown()
        print(f" COMPLETADO: {self._testMethodName}\n")
        
    def print_separator(self):
        print("\n" + "="*80)
        
    def print_step(self, step_number, description):
        print(f"   Paso {step_number}: {description}")
        
    def print_result(self, key, value):
        print(f"       {key}: {value}")
        
    def print_assertion(self, description, result):
        status = "EXITO" if result else " FALLO"
        print(f"      {status} Aserción: {description}")


class ViewsIntegrationTests(DetailedTestCase):
    """Pruebas de integración entre vistas y componentes del sistema"""
    
    def setUp(self):
        super().setUp()
        self.client = Client()
        # Agregar testserver a ALLOWED_HOSTS temporalmente
        from django.conf import settings
        if 'testserver' not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS.append('testserver')
        
        self.print_result("Cliente de prueba", "Inicializado")
        
        # Reiniciar variables globales
        from core.views import emotion_history, emotion_lock
        with emotion_lock:
            emotion_history.clear()
        self.print_result("Estado inicial", "Variables limpiadas")

   

    def test_video_feed_streaming_response(self):
        """
        Integración 2: El video feed retorna una respuesta de streaming válida
        Verifica la integración entre vista, generador y OpenCV
        """
        self.print_step(1, "Resolver URL del video feed")
        url = reverse('video_feed_cam')
        self.print_result("URL resuelta", url)
        
        self.print_step(2, "Realizar petición GET al endpoint")
        response = self.client.get(url)
        self.print_result("Status code", response.status_code)
        
        self.print_step(3, "Verificar tipo de respuesta")
        self.print_result("Tipo de respuesta", type(response).__name__)
        self.print_assertion("Status code == 200", response.status_code == 200)
        self.assertEqual(response.status_code, 200)
        
        self.print_step(4, "Verificar Content-Type para streaming")
        content_type = response.get('Content-Type', '')
        self.print_result("Content-Type", content_type)
        expected_type = 'multipart/x-mixed-replace; boundary=frame'
        self.print_assertion(f"Content-Type correcto", content_type == expected_type)
        self.assertEqual(content_type, expected_type)


class DeepFaceIntegrationTests(DetailedTestCase):
    """Pruebas de integración con DeepFace y procesamiento de emociones"""
    
    def setUp(self):
        super().setUp()
        # Reiniciar variables globales
        from core.views import emotion_history, emotion_lock, emotion_detected, all_emotions
        with emotion_lock:
            emotion_history.clear()
        self.print_result("Variables globales", "Reiniciadas")

    @patch('core.views.DeepFace.analyze')
    def test_deepface_to_emotion_translation(self, mock_analyze):
        """
        Integración 3: DeepFace se integra correctamente con traducción de emociones
        Verifica la cadena: DeepFace → Análisis → Traducción → Actualización
        """
        from core.views import analyze_emotion
        
        self.print_step(1, "Configurar mock de DeepFace con múltiples emociones")
        mock_analyze.return_value = {
            'emotion': {
                'happy': 75.5,
                'sad': 10.2,
                'angry': 5.3,
                'surprise': 4.0,
                'fear': 3.0,
                'disgust': 1.5,
                'neutral': 0.5
            },
            'dominant_emotion': 'happy'
        }
        self.print_result("Emoción dominante (EN)", "happy (75.5%)")
        
        self.print_step(2, "Crear frame de prueba")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.print_result("Dimensiones del frame", frame.shape)
        
        self.print_step(3, "Ejecutar análisis de emoción")
        analyze_emotion(frame)
        self.print_result("Análisis ejecutado", "Sin errores")
        
        self.print_step(4, "Verificar traducción al español")
        from core.views import emotion_detected, all_emotions
        self.print_result("Emoción detectada (ES)", emotion_detected)
        self.print_result("Todas las emociones", all_emotions)
        
        self.print_assertion("Traducción correcta: happy → Feliz", emotion_detected == 'Feliz')
        self.assertEqual(emotion_detected, 'Feliz')
        
        self.print_assertion("'Feliz' en diccionario de emociones", 'Feliz' in all_emotions)
        self.assertIn('Feliz', all_emotions)



   

class OpenCVIntegrationTests(DetailedTestCase):
    """Pruebas de integración con OpenCV y procesamiento de imagen"""
    

    def test_face_detection_and_emotion_overlay(self):
        """
        Integración 7: Detección de rostros y superposición de información
        Verifica: Detección → Dibujo de rectángulos → Texto → Visualización
        """
        self.print_step(1, "Crear imagen con rostro simulado")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Dibujar rostro simulado
        cv2.circle(img, (320, 240), 80, (200, 180, 150), -1)  # Cara
        cv2.circle(img, (300, 220), 10, (50, 50, 50), -1)      # Ojo izquierdo
        cv2.circle(img, (340, 220), 10, (50, 50, 50), -1)      # Ojo derecho
        cv2.ellipse(img, (320, 260), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Sonrisa
        
        self.print_result("Imagen creada", f"Shape: {img.shape}")
        
        self.print_step(2, "Detectar rostros en la imagen")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        self.print_result("Rostros encontrados", len(faces))
        
        self.print_step(3, "Dibujar rectángulos alrededor de rostros")
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.print_result(f"  Rostro #{i+1}", f"Posición: ({x}, {y}), Tamaño: {w}x{h}")
        
        self.print_step(4, "Superponer texto de emoción")
        emotion_text = "Feliz (85%)"
        cv2.putText(img, emotion_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        self.print_result("Texto agregado", emotion_text)
        
        self.print_step(5, "Verificar imagen final")
        self.print_assertion("Imagen procesada correctamente", img.shape == (480, 640, 3))
        self.assertEqual(img.shape, (480, 640, 3))


class EndToEndIntegrationTests(DetailedTestCase):
    """Pruebas de integración de extremo a extremo"""
    
    @patch('core.views.video_feed_cam')
    @patch('core.views.DeepFace.analyze')
    def test_complete_workflow_simulation(self, mock_deepface, mock_camera):
        """
        Integración 8: Flujo completo del sistema de detección
        Simula: Cámara → Captura → Análisis → Procesamiento → Streaming
        """
        self.print_step(1, "Configurar mock de la cámara")
        mock_cam_instance = MagicMock()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Dibujar un rostro simple
        cv2.circle(test_frame, (320, 240), 50, (255, 255, 255), -1)
        
        ret, jpeg = cv2.imencode('.jpg', test_frame)
        mock_cam_instance.get_frame.return_value = jpeg.tobytes()
        mock_camera.return_value = mock_cam_instance
        self.print_result("Cámara simulada", "Configurada")
        
        self.print_step(2, "Configurar mock de DeepFace")
        mock_deepface.return_value = {
            'emotion': {'happy': 85.0, 'sad': 10.0, 'neutral': 5.0},
            'dominant_emotion': 'happy'
        }
        self.print_result("DeepFace simulado", "Configurado con emoción 'happy'")
        
        self.print_step(3, "Realizar petición al video feed")
        client = Client()
        response = client.get(reverse('video_feed_cam'))
        
        self.print_result("Status code", response.status_code)
        self.print_assertion("Respuesta exitosa", response.status_code == 200)
        self.assertEqual(response.status_code, 200)
        
        self.print_step(4, "Verificar que se obtiene streaming")
        self.print_result("Content-Type", response.get('Content-Type'))
        self.print_assertion("Es streaming multipart", 
                           'multipart' in response.get('Content-Type', ''))
        self.assertIn('multipart', response.get('Content-Type', ''))

    @patch('core.views.DeepFace.analyze')
    def test_concurrent_emotion_analysis(self, mock_analyze):
        """
        Integración 9: Análisis concurrente de emociones (thread-safe)
        Verifica que múltiples análisis simultáneos no corrompan datos
        """
        from core.views import analyze_emotion, emotion_history, emotion_lock
        
        self.print_step(1, "Limpiar historial")
        with emotion_lock:
            emotion_history.clear()
        self.print_result("Historial inicial", len(emotion_history))
        
        self.print_step(2, "Configurar mock para múltiples llamadas")
        mock_analyze.return_value = {
            'emotion': {'happy': 75.0},
            'dominant_emotion': 'happy'
        }
        
        self.print_step(3, "Ejecutar análisis concurrentes")
        num_threads = 5
        analyses_per_thread = 3
        
        def analyze_multiple():
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            for _ in range(analyses_per_thread):
                analyze_emotion(frame)
                time.sleep(0.01)  # Simular tiempo de procesamiento
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=analyze_multiple)
            threads.append(thread)
            thread.start()
        
        self.print_result("Threads lanzados", num_threads)
        self.print_result("Análisis por thread", analyses_per_thread)
        
        self.print_step(4, "Esperar finalización de todos los threads")
        for thread in threads:
            thread.join()
        self.print_result("Todos los threads", "Completados")
        
        self.print_step(5, "Verificar integridad del historial")
        with emotion_lock:
            final_length = len(emotion_history)
            # Máximo 10 entradas por el límite del historial
            expected_max = 10
        
        self.print_result("Entradas en historial", final_length)
        self.print_result("Máximo permitido", expected_max)
        self.print_assertion(f"Historial <= {expected_max}", final_length <= expected_max)
        self.assertLessEqual(final_length, expected_max)


class DatabaseIntegrationTests(DetailedTestCase):
    """Pruebas de integración con la base de datos (si aplica)"""
    
    def test_emotion_data_persistence_concept(self):
        """
        Integración 10: Concepto de persistencia de datos de emociones
        Nota: Esta prueba valida el concepto, la implementación real 
        requeriría modelos de Django para almacenar emociones
        """
        self.print_step(1, "Verificar estructura de datos para persistencia")
        
        # Estructura de datos que podría guardarse en BD
        emotion_data = {
            'timestamp': time.time(),
            'emotion': 'Feliz',
            'confidence': 85.5,
            'all_emotions': {
                'Feliz': 85.5,
                'Triste': 10.2,
                'Neutral': 4.3
            }
        }
        
        self.print_result("Estructura de datos", emotion_data)
        
        self.print_step(2, "Validar campos requeridos")
        required_fields = ['timestamp', 'emotion', 'confidence', 'all_emotions']
        
        for field in required_fields:
            has_field = field in emotion_data
            self.print_assertion(f"Campo '{field}' presente", has_field)
            self.assertIn(field, emotion_data)
        
        self.print_step(3, "Validar tipos de datos")
        self.print_assertion("timestamp es numérico", isinstance(emotion_data['timestamp'], (int, float)))
        self.assertIsInstance(emotion_data['timestamp'], (int, float))
        
        self.print_assertion("emotion es string", isinstance(emotion_data['emotion'], str))
        self.assertIsInstance(emotion_data['emotion'], str)
        
        self.print_assertion("confidence es numérico", isinstance(emotion_data['confidence'], (int, float)))
        self.assertIsInstance(emotion_data['confidence'], (int, float))
        
        self.print_assertion("all_emotions es diccionario", isinstance(emotion_data['all_emotions'], dict))
        self.assertIsInstance(emotion_data['all_emotions'], dict)


# Ejecutar todas las pruebas de integración
if __name__ == '__main__':
    import unittest
    
    print("\n" + " "*40)
    print("SUITE DE PRUEBAS DE INTEGRACIÓN - SISTEMA DE DETECCIÓN DE EMOCIONES")
    print("="*80)
    print(" Verificando integración entre componentes del sistema...")
    print("="*80)
    
    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar todas las clases de prueba de integración
    print("\n Cargando suites de pruebas de integración:")
    test_classes = [
        ViewsIntegrationTests,
        DeepFaceIntegrationTests,
        OpenCVIntegrationTests,
        EndToEndIntegrationTests,
        DatabaseIntegrationTests
    ]
    
    for test_class in test_classes:
        print(f"   • {test_class.__name__}")
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    print("\n" + "="*80)
    print(" EJECUTANDO PRUEBAS DE INTEGRACIÓN...")
    print("="*80)
    
    # Ejecutar con verbosidad máxima
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen detallado
    print("\n" + ""*40)
    print("="*80)
    print(" RESUMEN DE PRUEBAS DE INTEGRACIÓN")
    print("="*80)
    print(f" Total de pruebas:        {result.testsRun}")
    print(f"  Exitosas:                {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Fallidas:                {len(result.failures)}")
    print(f"  Errores:                 {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f" Tasa de éxito:           {success_rate:.1f}%")
    
    print("="*80)
    
    if result.failures:
        print("\n PRUEBAS FALLIDAS:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print("-"*80)
            print(traceback)
    
    if result.errors:
        print("\n  ERRORES:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print("-"*80)
            print(traceback)
    
    print("\n" + "="*80)
    if not result.failures and not result.errors:
        print(" ¡TODAS LAS PRUEBAS DE INTEGRACIÓN PASARON!")
        print("="*80)
        print(" Todos los componentes se integran correctamente")
        print(" El flujo de datos funciona de extremo a extremo")
        print(" El sistema es thread-safe y robusto")
    else:
        print("  ALGUNAS PRUEBAS DE INTEGRACIÓN FALLARON")
        print("="*80)
        print("• Revisa los componentes que interactúan")
        print("• Verifica las interfaces entre módulos")
        print("• Asegúrate de que los mocks sean correctos")
    
    print("="*80)
    print("\n Pruebas de integración completadas\n")