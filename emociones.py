import cv2
from deepface import DeepFace
import threading
import numpy as np

# Variables globales
emotion_detected = "Analizando..."
all_emotions = {}
emotion_lock = threading.Lock()
emotion_history = []  # Para promediar emociones a lo largo de varios frames
history_size = 2  # Reducido para mayor responsividad (antes 5)

def analyze_emotion(frame):
    """Analiza la emoción en un hilo separado con mayor sensibilidad"""
    global emotion_detected, all_emotions
    try:
        # Analizar con modelo más preciso pero más rápido (opencv para velocidad)
        result = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False,  # Menos estricto para detectar más rápido
            detector_backend='opencv',  # Más rápido que mtcnn
            silent=True
        )
        
        # Obtener resultados
        if isinstance(result, list):
            result = result[0]
        
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        
        # Traducir emociones al español
        emotions_spanish = {
            'angry': 'Enojado',
            'disgust': 'Disgusto',
            'fear': 'Miedo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Sorprendido',
            'neutral': 'Neutral'
        }
        
        # Filtrar solo emociones con confianza mayor al 10% (umbral más bajo para sensibilidad)
        filtered_emotions = {k: v for k, v in emotions.items() if v > 10.0}
        
        with emotion_lock:
            # Guardar emoción dominante solo si tiene más del 20% de confianza (umbral más bajo)
            if emotions[dominant_emotion] > 20.0:
                emotion_detected = emotions_spanish.get(dominant_emotion, dominant_emotion)
            else:
                emotion_detected = "Indefinido"
            
            # Guardar todas las emociones traducidas
            all_emotions = {emotions_spanish.get(k, k): v for k, v in filtered_emotions.items()}
            
            # Agregar a historial para promediar (menos frames para más rapidez)
            emotion_history.append(all_emotions)
            if len(emotion_history) > history_size:
                emotion_history.pop(0)
            
    except Exception as e:
        with emotion_lock:
            emotion_detected = "No detectado"
            all_emotions = {}

def average_emotions():
    """Promedia las emociones del historial para mayor estabilidad"""
    if not emotion_history:
        return {}
    
    # Promediar porcentajes
    avg_emotions = {}
    for emotion in emotion_history[0].keys():
        values = [hist.get(emotion, 0) for hist in emotion_history]
        avg_emotions[emotion] = sum(values) / len(values)
    
    return avg_emotions

# Inicializar cámara con mejor resolución
print("Iniciando cámara...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Cargar clasificador de rostros con detección más sensible
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n" + "="*60)
print("DETECTOR DE EMOCIONES DE ALTA SENSIBILIDAD - DeepFace")
print("="*60)
print("Presiona ESC para salir")
print("="*60 + "\n")

frame_count = 0
analyze_interval = 10  # Analizar más frecuentemente (cada 10 frames, antes 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame")
        break
    
    # Mejorar calidad de imagen (filtro ligero para no ralentizar)
    frame = cv2.bilateralFilter(frame, 5, 50, 50)
    
    # Detectar rostros con parámetros más sensibles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  # Más sensible
        minNeighbors=4,   # Menos vecinos para detectar más rostros
        minSize=(80, 80)  # Tamaño mínimo menor para detectar rostros más pequeños
    )
    
    # Analizar emoción cada cierto número de frames
    if frame_count % analyze_interval == 0 and len(faces) > 0:
        # Extraer región del rostro más grande
        if len(faces) > 0:
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            # Expandir región del rostro un 20% (menos margen para rapidez)
            margin = int(0.2 * w)
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(frame.shape[1], x + w + margin)
            y_end = min(frame.shape[0], y + h + margin)
            face_roi = frame[y_start:y_end, x_start:x_end].copy()
            
            # Crear hilo para análisis
            thread = threading.Thread(target=analyze_emotion, args=(face_roi,))
            thread.daemon = True
            thread.start()
    
    frame_count += 1
    
    # Dibujar rectángulos y mostrar emociones
    for (x, y, w, h) in faces:
        with emotion_lock:
            current_emotion = emotion_detected
            current_all_emotions = average_emotions()  # Usar emociones promediadas
        
        # Color según la emoción
        emotion_colors = {
            'Feliz': (0, 255, 0),      # Verde
            'Triste': (255, 0, 0),     # Azul
            'Enojado': (0, 0, 255),    # Rojo
            'Sorprendido': (0, 255, 255),  # Amarillo
            'Miedo': (128, 0, 128),    # Púrpura
            'Disgusto': (0, 165, 255), # Naranja
            'Neutral': (200, 200, 200) # Gris
        }
        
        color = emotion_colors.get(current_emotion, (0, 255, 0))
        
        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Fondo para el texto principal
        cv2.rectangle(frame, (x, y-50), (x+w, y), color, -1)
        
        # Texto de la emoción dominante
        cv2.putText(frame, current_emotion, (x+5, y-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Mostrar porcentaje de confianza (del promedio)
        if current_all_emotions and current_emotion in current_all_emotions:
            confidence = current_all_emotions[current_emotion]
            cv2.putText(frame, f'{confidence:.1f}%', (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Panel lateral con todas las emociones promediadas
    panel_width = 350
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
    
    # Título del panel
    cv2.putText(panel, 'EMOCIONES DETECTADAS', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar todas las emociones con barras (usando promedio)
    with emotion_lock:
        sorted_emotions = sorted(current_all_emotions.items(), key=lambda x: x[1], reverse=True)
    
    y_offset = 70
    for emotion, percentage in sorted_emotions:
        # Nombre de la emoción
        cv2.putText(panel, emotion, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Barra de progreso
        bar_length = int((percentage / 100) * 250)
        cv2.rectangle(panel, (10, y_offset + 5), (10 + bar_length, y_offset + 20), 
                     emotion_colors.get(emotion, (255, 255, 255)), -1)
        cv2.rectangle(panel, (10, y_offset + 5), (260, y_offset + 20), 
                     (100, 100, 100), 1)
        
        # Porcentaje
        cv2.putText(panel, f'{percentage:.1f}%', (270, y_offset + 17), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 45
    
    # Combinar frame y panel
    combined = cv2.hconcat([frame, panel])
    
    # Mostrar instrucciones
    cv2.putText(combined, 'Presiona ESC para salir', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mostrar frame
    cv2.imshow('Detector de Emociones de Alta Sensibilidad', combined)
    
    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("\n✓ Detector finalizado")
cap.release()
cv2.destroyAllWindows()
