"""
Pruebas Unitarias e Integración para Sistema de Detección de Emociones
Con salida detallada en consola
"""

from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch, MagicMock
import threading
import numpy as np
import sys

# Importar funciones de views.py
from .views import analyze_emotion, average_emotions


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
        print(f"    Paso {step_number}: {description}")
        
    def print_result(self, key, value):
        print(f"      ➤ {key}: {value}")
        
    def print_assertion(self, description, result):
        status = "✓" if result else "✗"
        print(f"      {status} Aserción: {description}")


class UnitTests(DetailedTestCase):
    """Pruebas unitarias para funciones individuales"""
    
    def setUp(self):
        super().setUp()
        # Reiniciar variables globales para pruebas
        from .views import emotion_history, emotion_lock, emotion_detected, all_emotions
        with emotion_lock:
            emotion_history.clear()
            emotion_detected = "Analizando..."
            all_emotions = {}
        self.print_result("Estado inicial", "Variables globales reiniciadas")

    @patch('core.views.DeepFace.analyze')
    def test_analyze_emotion_success(self, mock_analyze):
        """
        Prueba: Análisis exitoso de emoción cuando se detecta un rostro feliz
        Verifica que DeepFace procese correctamente y actualice las variables globales
        """
        self.print_step(1, "Configurar mock de DeepFace.analyze")
        mock_analyze.return_value = {
            'emotion': {'happy': 80.0, 'sad': 10.0, 'neutral': 10.0},
            'dominant_emotion': 'happy'
        }
        self.print_result("Mock configurado", "Emoción dominante = 'happy' (80%)")

        self.print_step(2, "Crear frame de prueba (imagen dummy)")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.print_result("Dimensiones del frame", f"{frame.shape}")

        self.print_step(3, "Ejecutar analyze_emotion()")
        analyze_emotion(frame)
        self.print_result("Función ejecutada", "Sin excepciones")

        self.print_step(4, "Verificar resultados")
        from .views import emotion_detected, all_emotions
        self.print_result("emotion_detected", f"'{emotion_detected}'")
        self.print_result("all_emotions", f"{all_emotions}")
        
        # Aserciones con feedback
        self.print_assertion("emotion_detected == 'Feliz'", emotion_detected == 'Feliz')
        self.assertEqual(emotion_detected, 'Feliz')
        
        self.print_assertion("'Feliz' in all_emotions", 'Feliz' in all_emotions)
        self.assertIn('Feliz', all_emotions)
        
        self.print_assertion("all_emotions['Feliz'] > 0", all_emotions.get('Feliz', 0) > 0)
        self.assertGreater(all_emotions['Feliz'], 0)

    @patch('core.views.DeepFace.analyze')
    def test_analyze_emotion_no_face(self, mock_analyze):
        """
        Prueba: Manejo de excepción cuando no se detecta ningún rostro
        Verifica que el sistema maneje errores correctamente
        """
        self.print_step(1, "Configurar mock para lanzar excepción")
        mock_analyze.side_effect = Exception("No face detected")
        self.print_result("Mock configurado", "Excepción programada")

        self.print_step(2, "Crear frame de prueba")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.print_result("Frame creado", f"Shape: {frame.shape}")

        self.print_step(3, "Ejecutar analyze_emotion() con frame sin rostro")
        analyze_emotion(frame)
        self.print_result("Función ejecutada", "Excepción manejada correctamente")

        self.print_step(4, "Verificar estado después de la excepción")
        from .views import emotion_detected, all_emotions
        self.print_result("emotion_detected", f"'{emotion_detected}'")
        self.print_result("all_emotions", f"{all_emotions}")
        
        self.print_assertion("emotion_detected == 'No detectado'", emotion_detected == "No detectado")
        self.assertEqual(emotion_detected, "No detectado")
        
        self.print_assertion("all_emotions == {}", all_emotions == {})
        self.assertEqual(all_emotions, {})

    def test_average_emotions(self):
        """
        Prueba: Cálculo del promedio de emociones desde el historial
        Verifica que los promedios se calculen correctamente
        """
        from .views import emotion_history, emotion_lock
        
        self.print_step(1, "Agregar datos al historial de emociones")
        with emotion_lock:
            emotion_history.append({'Feliz': 70.0, 'Triste': 20.0})
            emotion_history.append({'Feliz': 90.0, 'Triste': 10.0})
        self.print_result("Historial agregado", f"{len(emotion_history)} entradas")
        for i, entry in enumerate(emotion_history, 1):
            self.print_result(f"  Entrada {i}", entry)

        self.print_step(2, "Calcular promedio de emociones")
        avg = average_emotions()
        self.print_result("Promedio calculado", avg)
        
        self.print_step(3, "Verificar cálculos")
        self.print_result("Promedio esperado Feliz", "80.0")
        self.print_result("Promedio calculado Feliz", avg.get('Feliz', 0))
        self.print_assertion("avg['Feliz'] ≈ 80.0", abs(avg.get('Feliz', 0) - 80.0) < 0.1)
        self.assertAlmostEqual(avg['Feliz'], 80.0)
        
        self.print_result("Promedio esperado Triste", "15.0")
        self.print_result("Promedio calculado Triste", avg.get('Triste', 0))
        self.print_assertion("avg['Triste'] ≈ 15.0", abs(avg.get('Triste', 0) - 15.0) < 0.1)
        self.assertAlmostEqual(avg['Triste'], 15.0)

    @patch('core.views.DeepFace.analyze')
    def test_analyze_emotion_multiple_calls(self, mock_analyze):
        """
        Prueba: Múltiples análisis secuenciales de emociones
        Verifica que el historial se actualice correctamente con múltiples llamadas
        """
        self.print_step(1, "Configurar mock para múltiples emociones")
        emotions_sequence = [
            {'emotion': {'happy': 90.0, 'sad': 5.0, 'neutral': 5.0}, 'dominant_emotion': 'happy'},
            {'emotion': {'sad': 85.0, 'happy': 10.0, 'neutral': 5.0}, 'dominant_emotion': 'sad'},
            {'emotion': {'neutral': 80.0, 'happy': 15.0, 'sad': 5.0}, 'dominant_emotion': 'neutral'}
        ]
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        for i, emotion_data in enumerate(emotions_sequence, 1):
            self.print_step(i+1, f"Análisis {i}/{len(emotions_sequence)}")
            mock_analyze.return_value = emotion_data
            analyze_emotion(frame)
            
            from .views import emotion_detected
            self.print_result(f"Emoción detectada #{i}", emotion_detected)
        
        self.print_step(len(emotions_sequence)+2, "Verificar historial final")
        from .views import emotion_history
        self.print_result("Entradas en historial", len(emotion_history))
        self.print_assertion(f"len(emotion_history) <= 10", len(emotion_history) <= 10)
        self.assertLessEqual(len(emotion_history), 10)


class IntegrationTests(DetailedTestCase):
    """Pruebas de integración para vistas y endpoints"""
    
    def setUp(self):
        super().setUp()
        self.client = Client()
        self.print_result("Cliente de prueba", "Inicializado")

    def test_index_view(self):
        """
        Prueba: Vista principal (index) carga correctamente
        Verifica que la página principal sea accesible y use el template correcto
        """
        self.print_step(1, "Obtener URL de la vista 'index'")
        url = reverse('index')
        self.print_result("URL", url)

        self.print_step(2, "Realizar petición GET")
        response = self.client.get(url)
        self.print_result("Status code", response.status_code)
        self.print_result("Template usado", response.templates[0].name if response.templates else "Ninguno")

        self.print_step(3, "Verificar respuesta")
        self.print_assertion("status_code == 200", response.status_code == 200)
        self.assertEqual(response.status_code, 200)
        
        self.print_assertion("Template correcto usado", 'core/index.html' in [t.name for t in response.templates])
        self.assertTemplateUsed(response, 'core/index.html')

    def test_video_feed_cam_view(self):
        """
        Prueba: Vista de streaming de video funciona correctamente
        Verifica que el endpoint de video retorne streaming multipart
        """
        self.print_step(1, "Obtener URL de la vista 'video_feed_cam'")
        url = reverse('video_feed_cam')
        self.print_result("URL", url)

        self.print_step(2, "Realizar petición GET al endpoint de video")
        response = self.client.get(url)
        self.print_result("Status code", response.status_code)
        self.print_result("Content-Type", response.get('Content-Type', 'No definido'))

        self.print_step(3, "Verificar respuesta de streaming")
        self.print_assertion("status_code == 200", response.status_code == 200)
        self.assertEqual(response.status_code, 200)
        
        expected_content_type = 'multipart/x-mixed-replace; boundary=frame'
        self.print_assertion(f"Content-Type == '{expected_content_type}'", 
                           response['Content-Type'] == expected_content_type)
        self.assertEqual(response['Content-Type'], expected_content_type)

    def test_index_view_context(self):
        """
        Prueba: Contexto de la vista index contiene datos esperados
        Verifica que se pasen las variables de contexto correctas al template
        """
        self.print_step(1, "Realizar petición a la vista index")
        response = self.client.get(reverse('index'))
        
        self.print_step(2, "Examinar contexto de la respuesta")
        # RequestContext necesita ser convertido a dict para obtener las claves
        if response.context:
            # Convertir RequestContext a diccionario
            context_dict = {}
            for context in response.context:
                context_dict.update(dict(context))
            context_keys = list(context_dict.keys())
            self.print_result("Claves en contexto", context_keys)
            self.print_result("Número de variables", len(context_keys))
            
            # Mostrar algunas variables comunes de Django
            common_vars = ['view', 'request', 'user', 'perms', 'csrf_token']
            found_vars = [var for var in common_vars if var in context_keys]
            if found_vars:
                self.print_result("Variables comunes encontradas", found_vars)
        else:
            context_keys = []
            self.print_result("Claves en contexto", "Contexto vacío")
        
        self.print_step(3, "Verificar estructura del contexto")
        self.print_assertion("Contexto no está vacío", response.context is not None)
        self.assertIsNotNone(response.context)


class ThreadSafetyTests(DetailedTestCase):
    """Pruebas de seguridad en hilos múltiples"""
    
    def test_emotion_lock_thread_safety(self):
        """
        Prueba: El lock de emociones previene condiciones de carrera
        Verifica que múltiples hilos puedan acceder de forma segura al historial
        """
        from .views import emotion_history, emotion_lock
        
        self.print_step(1, "Limpiar historial de emociones")
        with emotion_lock:
            emotion_history.clear()
        self.print_result("Historial limpiado", f"Tamaño: {len(emotion_history)}")
        
        self.print_step(2, "Definir función para agregar datos desde hilos")
        def add_emotion(emotion_data, iterations):
            for i in range(iterations):
                with emotion_lock:
                    emotion_history.append(emotion_data)
        
        self.print_step(3, "Crear y lanzar múltiples hilos")
        threads = []
        num_threads = 5
        iterations_per_thread = 10
        
        for i in range(num_threads):
            emotion_data = {'test': i}
            thread = threading.Thread(target=add_emotion, args=(emotion_data, iterations_per_thread))
            threads.append(thread)
            thread.start()
            
        self.print_result("Hilos creados", num_threads)
        self.print_result("Iteraciones por hilo", iterations_per_thread)
        
        self.print_step(4, "Esperar a que todos los hilos terminen")
        for thread in threads:
            thread.join()
        self.print_result("Hilos completados", "Todos")
        
        self.print_step(5, "Verificar integridad del historial")
        expected_total = num_threads * iterations_per_thread
        actual_total = len(emotion_history)
        
        self.print_result("Total esperado", expected_total)
        self.print_result("Total actual", actual_total)
        
        self.print_assertion(f"len(emotion_history) == {expected_total}", actual_total == expected_total)
        self.assertEqual(actual_total, expected_total)


# Ejecutar pruebas con salida detallada
if __name__ == '__main__':
    import unittest
    
    print("\n" + "="*80)
    print("🚀 SISTEMA DE PRUEBAS - DETECCIÓN DE EMOCIONES")
    print("="*80)
    print(f"📅 Ejecutando pruebas unitarias e integración...")
    print("="*80 + "\n")
    
    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar todas las clases de prueba
    suite.addTests(loader.loadTestsFromTestCase(UnitTests))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    suite.addTests(loader.loadTestsFromTestCase(ThreadSafetyTests))
    
    # Ejecutar con verbosidad máxima
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen final detallado
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL DE EJECUCIÓN")
    print("="*80)
    print(f" Pruebas ejecutadas:     {result.testsRun}")
    print(f"✓  Exitosas:               {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗  Fallidas:               {len(result.failures)}")
    print(f"⚠  Errores:                {len(result.errors)}")
    print(f"⏱  Tiempo total:           {result.timeTaken:.2f}s" if hasattr(result, 'timeTaken') else "")
    print("="*80)
    
    if result.failures:
        print("\n❌ PRUEBAS FALLIDAS:")
        for test, traceback in result.failures:
            print(f"\n   • {test}")
            print(f"     {traceback}")
    
    if result.errors:
        print("\n⚠️  ERRORES EN PRUEBAS:")
        for test, traceback in result.errors:
            print(f"\n   • {test}")
            print(f"     {traceback}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
    
    print("\n" + "="*80 + "\n")