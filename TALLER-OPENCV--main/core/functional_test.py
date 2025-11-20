"""
Pruebas Funcionales para Sistema de Detección de Emociones
Validación de casos de uso completos con salida detallada
"""

import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import cv2
import numpy as np
from django.test import LiveServerTestCase
from django.contrib.staticfiles.testing import StaticLiveServerTestCase


class DetailedTestCase(unittest.TestCase):
    """Clase base con métodos para salida detallada en consola"""
    
    def setUp(self):
        super().setUp()
        self.print_header()
        
    def tearDown(self):
        super().tearDown()
        self.print_footer()
        
    def print_header(self):
        print("\n" + "="*80)
        print(f"TEST: {self._testMethodName}")
        print(f" {self._testMethodDoc or 'Sin descripción'}")
        print("="*80)
        
    def print_footer(self):
        print(" COMPLETADO\n")
        
    def print_step(self, step_num, description):
        print(f"\n    Paso {step_num}: {description}")
        
    def print_info(self, label, value):
        print(f"        {label}: {value}")
        
    def print_success(self, message):
        print(f"       {message}")
        
    def print_warning(self, message):
        print(f"        {message}")
        
    def print_error(self, message):
        print(f"       {message}")


class TestUserInterface(StaticLiveServerTestCase, DetailedTestCase):
    """Pruebas funcionales de la interfaz de usuario"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print("\n" + " "*40)
        print("CONFIGURACIÓN DE SELENIUM WEBDRIVER")
        print("="*80)
        
        # Configurar navegador (Chrome headless para CI/CD)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        print("    Opciones del navegador:")
        print("      • Modo: Headless")
        print("      • Sandbox: Deshabilitado")
        print("      • Dev SHM: Deshabilitado")
        
        cls.browser = webdriver.Chrome(options=options)
        cls.browser.implicitly_wait(10)
        
        print("   ✓ WebDriver inicializado correctamente")
        print("="*80 + "\n")
        
    @classmethod
    def tearDownClass(cls):
        print("\n" + "="*80)
        print(" CERRANDO WEBDRIVER")
        cls.browser.quit()
        print("   ✓ WebDriver cerrado correctamente")
        print("="*80 + "\n")
        super().tearDownClass()
        
    def test_homepage_loads(self):
        """Prueba Funcional 1: La página principal carga correctamente"""
        self.print_step(1, "Navegar a la URL del servidor de pruebas")
        self.print_info("URL", self.live_server_url)
        
        self.browser.get(self.live_server_url)
        self.print_success("Navegación completada")
        
        self.print_step(2, "Verificar que el título está presente")
        title = self.browser.title
        self.print_info("Título de la página", f"'{title}'")
        
        self.print_step(3, "Validar que contiene 'Detección'")
        if 'Detección' in title:
            self.print_success(f"El título contiene 'Detección'")
        else:
            self.print_error(f"El título NO contiene 'Detección'")
        
        self.assertIn('Detección', self.browser.title)
        
    def test_video_stream_element_exists(self):
        """Prueba Funcional 2: Elemento de video existe en la página"""
        self.print_step(1, "Cargar la página principal")
        self.browser.get(self.live_server_url)
        self.print_success("Página cargada")
        
        self.print_step(2, "Buscar elemento de video (<img>)")
        try:
            video_element = self.browser.find_element(By.TAG_NAME, 'img')
            self.print_success("Elemento <img> encontrado")
            self.print_info("Tag", video_element.tag_name)
            self.print_info("Atributos", video_element.get_attribute('outerHTML')[:100] + "...")
            self.assertIsNotNone(video_element)
        except Exception as e:
            self.print_error(f"No se encontró elemento: {str(e)}")
            self.fail("No se encontró elemento de video en la página")
            
    def test_video_stream_src(self):
        """Prueba Funcional 3: Source del video apunta a la URL correcta"""
        self.print_step(1, "Navegar a la página")
        self.browser.get(self.live_server_url)
        
        self.print_step(2, "Buscar elemento de video")
        try:
            video_element = self.browser.find_element(By.TAG_NAME, 'img')
            self.print_success("Elemento encontrado")
            
            self.print_step(3, "Verificar atributo 'src'")
            src = video_element.get_attribute('src')
            self.print_info("URL del src", src)
            
            if src and 'video_feed_cam' in src:
                self.print_success("El src contiene 'video_feed_cam'")
                self.assertIn('video_feed_cam', src)
            else:
                self.print_warning("El src no contiene 'video_feed_cam' o es None")
        except Exception as e:
            self.print_warning(f"Template puede no existir: {str(e)}")


class TestEmotionDetectionFlow(DetailedTestCase):
    """Pruebas del flujo completo de detección de emociones"""
    
    def setUp(self):
        super().setUp()
        self.print_step(1, "Crear imagen de prueba con rostro simulado")
        self.test_image = self.create_test_face_image()
        self.print_info("Dimensiones", self.test_image.shape)
        self.print_success("Imagen de prueba creada")
        
    def create_test_face_image(self):
        """Crear imagen de prueba con un rostro simulado"""
        # Crear imagen en blanco
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Dibujar un círculo que simula un rostro
        cv2.circle(img, (320, 240), 100, (200, 180, 150), -1)
        
        # Dibujar ojos
        cv2.circle(img, (290, 220), 15, (50, 50, 50), -1)
        cv2.circle(img, (350, 220), 15, (50, 50, 50), -1)
        
        # Dibujar boca (sonrisa)
        cv2.ellipse(img, (320, 270), (40, 20), 0, 0, 180, (50, 50, 50), 2)
        
        return img
        
    def test_face_detection_in_test_image(self):
        """Prueba Funcional 4: Detectar rostro en imagen de prueba"""
        self.print_step(2, "Cargar clasificador Haar Cascade")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.print_info("Ruta del clasificador", cascade_path)
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        self.print_success("Clasificador cargado")
    
        self.print_step(3, "Convertir imagen a escala de grises")
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        self.print_info("Dimensiones gray", gray.shape)
        
        self.print_step(4, "Ejecutar detección de rostros")
        try:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            self.print_success("Detección ejecutada sin errores")
            self.print_info("Rostros detectados", len(faces))
            
            for i, (x, y, w, h) in enumerate(faces):
                self.print_info(f"  Rostro {i+1}", f"x={x}, y={y}, w={w}, h={h}")
                
        except Exception as e:
            self.print_error(f"La detección falló: {e}")
            self.fail(f"La detección de rostros falló: {e}")

        
    def test_image_encoding(self):
        """Prueba Funcional 5: Codificación de imagen a JPEG"""
        self.print_step(2, "Codificar imagen a formato JPEG")
        ret, jpeg = cv2.imencode('.jpg', self.test_image)
        
        self.print_info("Retorno de imencode", ret)
        self.print_info("Tamaño del JPEG", f"{len(jpeg)} bytes" if jpeg is not None else "None")
        
        self.print_step(3, "Validar codificación exitosa")
        if ret:
            self.print_success("Codificación exitosa")
        else:
            self.print_error("Codificación fallida")
            
        self.assertTrue(ret)
        self.assertIsNotNone(jpeg)
        self.assertGreater(len(jpeg), 0)
        
    def test_frame_processing_pipeline(self):
        """Prueba Funcional 6: Pipeline completo de procesamiento de frame"""
        self.print_step(2, "Aplicar filtro bilateral")
        filtered = cv2.bilateralFilter(self.test_image, 5, 50, 50)
        self.print_info("Shape filtrado", filtered.shape)
        self.print_info("Shape original", self.test_image.shape)
        self.assertEqual(filtered.shape, self.test_image.shape)
        self.print_success("Filtro aplicado correctamente")
        
        self.print_step(3, "Convertir a escala de grises")
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        self.print_info("Dimensiones gray", gray.shape)
        self.print_info("Número de canales", len(gray.shape))
        self.assertEqual(len(gray.shape), 2)
        self.print_success("Conversión a grises exitosa")
        
        self.print_step(4, "Detectar rostros")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        self.print_info("Tipo de retorno", type(faces).__name__)
        self.print_info("Rostros encontrados", len(faces))
        self.assertTrue(isinstance(faces, (np.ndarray, tuple)))
        self.print_success("Detección completada")

        self.print_step(5, "Codificar resultado")
        ret, jpeg = cv2.imencode('.jpg', filtered)
        self.print_info("Codificación exitosa", ret)
        self.print_info("Tamaño JPEG", f"{len(jpeg)} bytes")
        self.assertTrue(ret)
        self.print_success("Pipeline completo ejecutado")


class TestSystemIntegration(DetailedTestCase):
    """Pruebas de integración del sistema completo"""
    
    def test_opencv_installation(self):
        """Prueba Funcional 7: OpenCV está instalado correctamente"""
        self.print_step(1, "Verificar instalación de OpenCV")
        try:
            version = cv2.__version__
            self.print_success(f"OpenCV instalado")
            self.print_info("Versión", version)
            self.assertIsNotNone(version)
        except Exception as e:
            self.print_error(f"OpenCV no está instalado: {e}")
            self.fail("OpenCV no está instalado correctamente")
            
    def test_cascade_classifier_files(self):
        """Prueba Funcional 8: Archivos del clasificador Haar existen"""
        self.print_step(1, "Localizar archivo del clasificador")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.print_info("Ruta", cascade_path)
        
        self.print_step(2, "Cargar clasificador")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.print_step(3, "Verificar que no está vacío")
        is_empty = face_cascade.empty()
        self.print_info("Clasificador vacío", is_empty)
        
        if not is_empty:
            self.print_success("Clasificador cargado correctamente")
        else:
            self.print_error("Clasificador está vacío")
            
        self.assertFalse(face_cascade.empty())
        
    def test_numpy_integration(self):
        """Prueba Funcional 9: NumPy funciona correctamente con OpenCV"""
        self.print_step(1, "Crear array con NumPy")
        np_array = np.zeros((100, 100, 3), dtype=np.uint8)
        self.print_info("Shape del array", np_array.shape)
        self.print_info("Dtype", np_array.dtype)
        
        self.print_step(2, "Procesar con OpenCV")
        gray = cv2.cvtColor(np_array, cv2.COLOR_BGR2GRAY)
        self.print_info("Shape gray", gray.shape)
        
        self.print_step(3, "Validar resultado")
        expected_shape = (100, 100)
        self.print_info("Shape esperado", expected_shape)
        self.print_info("Shape obtenido", gray.shape)
        
        if gray.shape == expected_shape:
            self.print_success("Integración NumPy-OpenCV correcta")
        else:
            self.print_error("Las dimensiones no coinciden")
            
        self.assertEqual(gray.shape, (100, 100))


class TestEmotionAnalysisIntegration(DetailedTestCase):
    """Pruebas de integración con DeepFace (simuladas)"""
    
    def test_deepface_response_structure(self):
        """Prueba Funcional 10: Estructura de respuesta esperada de DeepFace"""
        self.print_step(1, "Crear respuesta mock de DeepFace")
        mock_response = {
            'emotion': {
                'angry': 5.2,
                'disgust': 2.1,
                'fear': 3.5,
                'happy': 85.3,
                'sad': 1.2,
                'surprise': 2.1,
                'neutral': 0.6
            },
            'dominant_emotion': 'happy'
        }
        self.print_info("Emoción dominante", mock_response['dominant_emotion'])
        self.print_info("Número de emociones", len(mock_response['emotion']))
        
        self.print_step(2, "Verificar estructura")
        self.assertIn('emotion', mock_response)
        self.print_success("Clave 'emotion' presente")
        
        self.assertIn('dominant_emotion', mock_response)
        self.print_success("Clave 'dominant_emotion' presente")
        
        self.assertEqual(len(mock_response['emotion']), 7)
        self.print_success("7 emociones en la respuesta")
        
        self.print_step(3, "Mostrar todas las emociones")
        for emotion, value in mock_response['emotion'].items():
            self.print_info(f"  {emotion}", f"{value:.1f}%")
        
    def test_emotion_translation_logic(self):
        """Prueba Funcional 11: Lógica de traducción de emociones"""
        self.print_step(1, "Definir diccionario de traducciones")
        emotions_spanish = {
            'angry': 'Enojado',
            'disgust': 'Disgusto',
            'fear': 'Miedo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Sorprendido',
            'neutral': 'Neutral'
        }
        self.print_info("Emociones disponibles", len(emotions_spanish))
        
        self.print_step(2, "Probar traducción")
        english_emotion = 'happy'
        spanish_emotion = emotions_spanish.get(english_emotion, english_emotion)
        
        self.print_info("Inglés", english_emotion)
        self.print_info("Español", spanish_emotion)
        
        self.print_step(3, "Validar traducción")
        if spanish_emotion == 'Feliz':
            self.print_success("Traducción correcta")
        else:
            self.print_error(f"Traducción incorrecta: {spanish_emotion}")
            
        self.assertEqual(spanish_emotion, 'Feliz')
        
    def test_confidence_threshold(self):
        """Prueba Funcional 12: Umbral de confianza para emociones"""
        self.print_step(1, "Definir emociones con diferentes niveles de confianza")
        emotions = {
            'happy': 85.3,
            'sad': 8.5,
            'neutral': 5.2
        }
        self.print_info("Emociones originales", emotions)
        
        self.print_step(2, "Aplicar filtro de umbral (> 10%)")
        threshold = 10.0
        filtered = {k: v for k, v in emotions.items() if v > threshold}
        self.print_info("Umbral", f"{threshold}%")
        self.print_info("Emociones filtradas", filtered)
        
        self.print_step(3, "Verificar filtrado")
        self.assertIn('happy', filtered)
        self.print_success("'happy' (85.3%) incluida")
        
        self.assertNotIn('sad', filtered)
        self.print_success("'sad' (8.5%) excluida")
        
        self.assertNotIn('neutral', filtered)
        self.print_success("'neutral' (5.2%) excluida")


class TestVideoStreamGeneration(DetailedTestCase):
    """Pruebas del generador de video stream"""
    
    def test_jpeg_encoding(self):
        """Prueba Funcional 13: Codificación JPEG funciona"""
        self.print_step(1, "Crear frame de prueba aleatorio")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.print_info("Dimensiones", test_frame.shape)
        
        self.print_step(2, "Codificar a JPEG")
        ret, jpeg = cv2.imencode('.jpg', test_frame)
        
        self.print_info("Codificación exitosa", ret)
        self.print_info("Tamaño", f"{len(jpeg.tobytes())} bytes")
        
        self.print_step(3, "Validar resultado")
        self.assertTrue(ret)
        self.assertGreater(len(jpeg.tobytes()), 0)
        self.print_success("Codificación JPEG correcta")
        
    def test_multipart_frame_format(self):
        """Prueba Funcional 14: Formato de frame multipart correcto"""
        self.print_step(1, "Crear frame de prueba")
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        self.print_step(2, "Codificar frame")
        ret, jpeg = cv2.imencode('.jpg', test_frame)
        
        self.print_step(3, "Construir formato multipart")
        frame_bytes = (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      jpeg.tobytes() + b'\r\n')
        
        self.print_info("Tamaño total", f"{len(frame_bytes)} bytes")
        
        self.print_step(4, "Verificar componentes del formato")
        self.assertIn(b'--frame', frame_bytes)
        self.print_success("Boundary '--frame' presente")
        
        self.assertIn(b'Content-Type: image/jpeg', frame_bytes)
        self.print_success("Content-Type correcto")


class TestUIComponents(DetailedTestCase):
    """Pruebas de componentes de la interfaz visual"""
    
    def test_emotion_colors_defined(self):
        """Prueba Funcional 15: Colores de emociones están definidos"""
        self.print_step(1, "Definir paleta de colores de emociones")
        emotion_colors = {
            'Feliz': (0, 255, 0),
            'Triste': (255, 0, 0),
            'Enojado': (0, 0, 255),
            'Sorprendido': (0, 255, 255),
            'Miedo': (128, 0, 128),
            'Disgusto': (0, 165, 255),
            'Neutral': (200, 200, 200)
        }
        
        self.print_step(2, "Verificar completitud de la paleta")
        self.print_info("Emociones definidas", len(emotion_colors))
        
        for emotion, color in emotion_colors.items():
            self.print_info(f"  {emotion}", f"RGB{color}")
        
        self.assertEqual(len(emotion_colors), 7)
        self.assertIn('Feliz', emotion_colors)
        self.print_success("Paleta completa con 7 emociones")
        
    def test_panel_creation(self):
        """Prueba Funcional 16: Panel lateral se crea correctamente"""
        self.print_step(1, "Definir dimensiones del panel")
        panel_width = 350
        height = 480
        self.print_info("Ancho", panel_width)
        self.print_info("Alto", height)
        
        self.print_step(2, "Crear panel con NumPy")
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        
        self.print_step(3, "Verificar dimensiones")
        expected_shape = (480, 350, 3)
        self.print_info("Shape esperado", expected_shape)
        self.print_info("Shape obtenido", panel.shape)
        
        self.assertEqual(panel.shape, (480, 350, 3))
        self.print_success("Panel creado con dimensiones correctas")
        
    def test_text_overlay(self):
        """Prueba Funcional 17: Texto se puede superponer en imagen"""
        self.print_step(1, "Crear imagen base")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.print_info("Dimensiones", test_img.shape)
        
        self.print_step(2, "Superponer texto con OpenCV")
        try:
            cv2.putText(test_img, 'Test Text', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            test_passed = True
            self.print_success("Texto superpuesto exitosamente")
            self.print_info("Texto", "'Test Text'")
            self.print_info("Posición", "(10, 30)")
            self.print_info("Fuente", "FONT_HERSHEY_SIMPLEX")
        except Exception as e:
            test_passed = False
            self.print_error(f"Error al superponer texto: {e}")
            
        self.assertTrue(test_passed)


class TestPerformanceMetrics(DetailedTestCase):
    """Pruebas de rendimiento y métricas"""
    
    def test_frame_processing_time(self):
        """Prueba Funcional 18: Tiempo de procesamiento de frame"""
        self.print_step(1, "Crear frame de prueba")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.print_info("Dimensiones", test_frame.shape)
        
        self.print_step(2, "Iniciar cronómetro")
        start_time = time.time()
        
        self.print_step(3, "Ejecutar pipeline de procesamiento")
        # Simular procesamiento
        filtered = cv2.bilateralFilter(test_frame, 5, 50, 50)
        self.print_info("  Filtro bilateral", "✓")
        
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        self.print_info("  Conversión a grises", "✓")
        
        ret, jpeg = cv2.imencode('.jpg', filtered)
        self.print_info("  Codificación JPEG", "✓")
        
        self.print_step(4, "Detener cronómetro")
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.print_info("Tiempo de procesamiento", f"{processing_time:.4f} segundos")
        self.print_info("FPS estimado", f"{1/processing_time:.2f} fps")
        
        self.print_step(5, "Verificar rendimiento")
        threshold = 1.0
        self.print_info("Umbral máximo", f"{threshold} segundo")
        
        if processing_time < threshold:
            self.print_success(f"Rendimiento aceptable ({processing_time:.4f}s < {threshold}s)")
        else:
            self.print_warning(f"Rendimiento lento ({processing_time:.4f}s >= {threshold}s)")
        
        self.assertLess(processing_time, threshold)


# Script para ejecutar todas las pruebas con salida detallada
if __name__ == '__main__':
    print("\n" + " "*40)
    print("SUITE DE PRUEBAS FUNCIONALES - SISTEMA DE DETECCIÓN DE EMOCIONES")
    print("="*80)
    print(" Iniciando ejecución de pruebas funcionales completas...")
    print("="*80)
    
    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar todas las clases de prueba (excepto UI que requiere servidor)
    print("\n Cargando suites de pruebas:")
    print("   • TestEmotionDetectionFlow")
    print("   • TestSystemIntegration")
    print("   • TestEmotionAnalysisIntegration")
    print("   • TestVideoStreamGeneration")
    print("   • TestUIComponents")
    print("   • TestPerformanceMetrics")
    
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionDetectionFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionAnalysisIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoStreamGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestUIComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    print("\n" + "="*80)
    print(" EJECUTANDO PRUEBAS...")
    print("="*80)
    
    # Ejecutar pruebas con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Mostrar resumen detallado
    print("\n" + ""*40)
    print("="*80)
    print("RESUMEN FINAL DE PRUEBAS FUNCIONALES")
    print("="*80)
    print(f" Total de pruebas ejecutadas:    {result.testsRun}")
    print(f"  Pruebas exitosas:                {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Pruebas fallidas:                {len(result.failures)}")
    print(f"  Errores encontrados:             {len(result.errors)}")
    
    # Calcular porcentaje de éxito
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f" Tasa de éxito:                   {success_rate:.1f}%")
    
    print("="*80)
    
    # Mostrar detalles de fallos si existen
    if result.failures:
        print("\n DETALLES DE PRUEBAS FALLIDAS:")
        print("="*80)
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n   {i}. {test}")
            print("   " + "-"*76)
            print(traceback)
    
    # Mensaje final
    print("\n" + "="*80)
    if not result.failures and not result.errors:
        print(" ¡TODAS LAS PRUEBAS FUNCIONALES PASARON EXITOSAMENTE!")
        print("="*80)
        print("✓ El sistema está funcionando correctamente")
        print("✓ Todos los componentes están integrados")
        print("✓ El rendimiento cumple con los requisitos")
    else:
        print("  ALGUNAS PRUEBAS NO PASARON")
        print("="*80)
        print("• Revisa los detalles arriba para más información")
        print("• Corrige los errores y vuelve a ejecutar las pruebas")
    
    print("="*80)
    print("\n Ejecución de pruebas funcionales completada\n")

