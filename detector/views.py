from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
import threading
import time

# Cargar clasificadores Haar Cascade de OpenCV
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
SMILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)

# Colores para cada emoción (BGR format para OpenCV)
EMOTION_COLORS = {
    'Feliz': (50, 255, 50),       # Verde brillante
    'Triste': (255, 100, 50),     # Azul
    'Enojado': (0, 50, 255),      # Rojo
    'Sorprendido': (0, 200, 255), # Naranja
    'Neutral': (180, 180, 180),   # Gris claro
    'Miedo': (255, 100, 255),     # Magenta
    'Disgusto': (50, 150, 100)    # Verde oscuro
}

# Variable global para la cámara
camera = None
camera_lock = threading.Lock()


def index(request):
    """Vista principal que renderiza el template HTML"""
    return render(request, 'dect.html')


def analizar_emocion_opencv_mejorado(gray_face, face_rect, frame_color):
    """
    Análisis mejorado de emociones usando múltiples técnicas de OpenCV
    """
    x, y, w, h = face_rect
    roi_gray = gray_face[y:y+h, x:x+w]
    roi_color = frame_color[y:y+h, x:x+w]
    
    # Inicializar scores
    emotion_scores = {
        'Feliz': 0.0,
        'Triste': 0.0,
        'Sorprendido': 0.0,
        'Enojado': 0.0,
        'Neutral': 0.2,  # Base neutral
        'Miedo': 0.0,
        'Disgusto': 0.0
    }
    
    # ========== 1. DETECCIÓN DE SONRISA MEJORADA ==========
    # Usar múltiples parámetros para mejor detección
    smiles_strict = smile_cascade.detectMultiScale(
        roi_gray, 
        scaleFactor=1.8,        # Más conservador
        minNeighbors=25,        # Más exigente
        minSize=(int(w*0.35), int(h*0.2))  # Tamaño mínimo aumentado
    )
    
    smiles_relaxed = smile_cascade.detectMultiScale(
        roi_gray, 
        scaleFactor=1.6,        # Más conservador
        minNeighbors=20,        # Más exigente
        minSize=(int(w*0.3), int(h*0.15))  # Tamaño mínimo aumentado
    )
    
    # Analizar zona de la boca (tercio inferior)
    mouth_region = roi_gray[int(h*0.6):, int(w*0.2):int(w*0.8)]
    if mouth_region.size > 0:
        mouth_std = np.std(mouth_region)
        mouth_mean = np.mean(mouth_region)
        
        # Sonrisa fuerte detectada
        if len(smiles_strict) > 0:
            emotion_scores['Feliz'] = 0.7
            emotion_scores['Neutral'] = 0.1
        # Sonrisa moderada
        elif len(smiles_relaxed) > 0 and mouth_std > 40:
            emotion_scores['Feliz'] = 0.5
            emotion_scores['Neutral'] = 0.2
    
    # ========== 2. ANÁLISIS DE OJOS MEJORADO ==========
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(int(w*0.15), int(h*0.1))
    )
    
    if len(eyes) >= 2:
        # Ordenar ojos por posición horizontal
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        eye_left = eyes_sorted[0]
        eye_right = eyes_sorted[-1] if len(eyes_sorted) > 1 else eyes_sorted[0]
        
        # Calcular métricas de ojos
        eye_areas = [ew * eh for (ex, ey, ew, eh) in [eye_left, eye_right]]
        avg_eye_area = np.mean(eye_areas)
        face_area = w * h
        eye_ratio = avg_eye_area / face_area
        
        # Analizar apertura de ojos
        eye_heights = [eh for (ex, ey, ew, eh) in [eye_left, eye_right]]
        avg_eye_height = np.mean(eye_heights)
        eye_widths = [ew for (ex, ey, ew, eh) in [eye_left, eye_right]]
        avg_eye_width = np.mean(eye_widths)
        eye_aspect_ratio = avg_eye_height / avg_eye_width if avg_eye_width > 0 else 0
        
        # Ojos muy abiertos (redondos) - más estricto
        if eye_aspect_ratio > 0.4 and eye_ratio > 0.02:  # Ambas condiciones deben cumplirse
            emotion_scores['Sorprendido'] += 0.4  # Reducido de 0.5
            emotion_scores['Miedo'] += 0.2       # Reducido de 0.3
            emotion_scores['Neutral'] *= 0.5     # Menos reducción de neutral
        # Ojos normales/relajados - rango ampliado
        elif 0.22 < eye_aspect_ratio <= 0.4:
            emotion_scores['Neutral'] += 0.4     # Aumentado
            emotion_scores['Sorprendido'] *= 0.5 # Reducir si estaba activo
        # Ojos entrecerrados
        elif eye_aspect_ratio < 0.22:
            if emotion_scores['Feliz'] < 0.3:    # No es sonrisa
                emotion_scores['Triste'] += 0.4
                emotion_scores['Neutral'] += 0.2  # Añadir algo de neutral
    
    # ========== 3. ANÁLISIS DE CEJAS (ZONA SUPERIOR) ==========
    eyebrow_region = roi_gray[0:int(h*0.35), :]
    if eyebrow_region.size > 0:
        # Detectar bordes para identificar cejas
        eyebrow_edges = cv2.Canny(eyebrow_region, 40, 120)
        edge_density = np.sum(eyebrow_edges > 0) / eyebrow_edges.size
        
        # Analizar posición vertical de los bordes
        rows_with_edges = np.any(eyebrow_edges > 0, axis=1)
        if np.sum(rows_with_edges) > 0:
            first_edge_row = np.argmax(rows_with_edges)
            relative_position = first_edge_row / eyebrow_region.shape[0]
            
            # Cejas levantadas (sorpresa) - más estricto
            if relative_position < 0.25 and edge_density > 0.05:  # Condiciones más estrictas
                emotion_scores['Sorprendido'] += 0.25  # Reducido de 0.3
                emotion_scores['Miedo'] += 0.15       # Reducido de 0.2
            # Cejas fruncidas (enojo) - más sensible
            elif edge_density > 0.07:  # Un poco más sensible
                emotion_scores['Enojado'] += 0.6      # Aumentado de 0.5
                emotion_scores['Disgusto'] += 0.3     # Aumentado de 0.2
                emotion_scores['Neutral'] *= 0.5      # Menos reducción de neutral
    
    # ========== 4. ANÁLISIS DE BOCA (ZONA INFERIOR) ==========
    lower_face = roi_gray[int(h*0.55):, :]
    if lower_face.size > 0:
        lower_edges = cv2.Canny(lower_face, 30, 100)
        lower_edge_density = np.sum(lower_edges > 0) / lower_edges.size
        
        # Buscar apertura de boca (píxeles oscuros)
        _, mouth_thresh = cv2.threshold(lower_face, 50, 255, cv2.THRESH_BINARY_INV)
        dark_pixels = np.sum(mouth_thresh > 0) / mouth_thresh.size
        
        # Boca muy abierta - más estricto
        if dark_pixels > 0.25 and lower_edge_density > 0.15:  # Ambas condiciones más estrictas
            emotion_scores['Sorprendido'] += 0.3              # Reducido de 0.4
            if len(eyes) >= 2 and emotion_scores['Sorprendido'] > 0.3:  # Solo si ya hay indicios
                emotion_scores['Miedo'] += 0.2
        # Boca con tensión (líneas horizontales)
        elif lower_edge_density > 0.08:
            if emotion_scores['Feliz'] < 0.3:
                emotion_scores['Disgusto'] += 0.4             # Aumentado de 0.3
                emotion_scores['Enojado'] += 0.2              # Añadido componente de enojo
    
    # ========== 5. ANÁLISIS DE SIMETRÍA FACIAL ==========
    left_half = roi_gray[:, :w//2]
    right_half = cv2.flip(roi_gray[:, w//2:], 1)
    
    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_width]
    right_half = right_half[:, :min_width]
    
    if left_half.shape == right_half.shape:
        symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        
        # Alta asimetría
        if symmetry_diff > 35:
            emotion_scores['Disgusto'] += 0.25
            emotion_scores['Enojado'] += 0.15
    
    # ========== 6. ANÁLISIS DE TEXTURA Y CONTRASTE ==========
    # Dividir rostro en regiones
    upper_third = roi_gray[0:int(h*0.33), :]
    middle_third = roi_gray[int(h*0.33):int(h*0.66), :]
    lower_third = roi_gray[int(h*0.66):, :]
    
    upper_std = np.std(upper_third)
    middle_std = np.std(middle_third)
    lower_std = np.std(lower_third)
    
    # Contraste alto en zona superior (cejas marcadas)
    if upper_std > 45:
        emotion_scores['Enojado'] += 0.2
    
    # Contraste bajo general (rostro relajado)
    if upper_std < 30 and middle_std < 30 and lower_std < 30:
        emotion_scores['Neutral'] += 0.3
        emotion_scores['Triste'] += 0.2
    
    # ========== 7. AJUSTES Y CORRECCIONES ==========
    # Si hay sonrisa clara, reducir emociones negativas
    if emotion_scores['Feliz'] > 0.5:
        emotion_scores['Triste'] *= 0.2
        emotion_scores['Enojado'] *= 0.2
        emotion_scores['Miedo'] *= 0.3
        emotion_scores['Disgusto'] *= 0.4
    
    # Si hay cejas fruncidas fuertes, reducir felicidad
    if emotion_scores['Enojado'] > 0.4:
        emotion_scores['Feliz'] *= 0.3
        emotion_scores['Neutral'] *= 0.5
    
    # Si todo es débil, aumentar neutral
    max_score = max(emotion_scores.values())
    if max_score < 0.4:  # Umbral aumentado
        emotion_scores['Neutral'] = 0.8  # Más peso a neutral
    
    # Penalizar sorprendido si no hay suficientes indicadores
    if emotion_scores['Sorprendido'] > 0.3:
        indicadores_sorpresa = 0
        # Contar cuántos indicadores de sorpresa tenemos
        if len(eyes) >= 2:  # Solo si detectamos ojos
            eye_heights = [eh for (ex, ey, ew, eh) in [eye_left, eye_right]]
            eye_widths = [ew for (ex, ey, ew, eh) in [eye_left, eye_right]]
            curr_eye_ratio = np.mean(eye_heights) / np.mean(eye_widths) if np.mean(eye_widths) > 0 else 0
            if curr_eye_ratio > 0.4:
                indicadores_sorpresa += 1  # Ojos muy abiertos
        
        if 'dark_pixels' in locals() and dark_pixels > 0.25:
            indicadores_sorpresa += 1      # Boca abierta
        
        if 'relative_position' in locals() and relative_position < 0.25:
            indicadores_sorpresa += 1      # Cejas levantadas
        
        # Si no hay suficientes indicadores, reducir sorpresa
        if indicadores_sorpresa < 2:
            emotion_scores['Sorprendido'] *= 0.3
            emotion_scores['Neutral'] += 0.4
    
    # ========== 8. NORMALIZACIÓN FINAL ==========
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
    else:
        emotion_scores['Neutral'] = 1.0
    
    # Suavizar scores muy bajos
    for emotion in emotion_scores:
        if emotion_scores[emotion] < 0.05:
            emotion_scores[emotion] = 0.0
    
    # Re-normalizar
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
    
    return emotion_scores


def dibujar_recuadro_emocion(img, face_rect, emotion, confidence):
    """
    Dibuja un recuadro alrededor del rostro con la emoción detectada
    """
    x, y, w, h = face_rect
    color = EMOTION_COLORS.get(emotion, (0, 255, 0))
    
    # Grosor de línea
    line_thickness = max(3, int(img.shape[1] / 300))
    
    # 1. RECTÁNGULO PRINCIPAL con efecto glow
    cv2.rectangle(img, (x, y), (x + w, y + h), color, line_thickness)
    
    # 2. ESQUINAS DECORATIVAS
    corner_length = int(w * 0.15)
    corner_thickness = line_thickness + 3
    
    # Dibujar esquinas
    cv2.line(img, (x, y), (x + corner_length, y), color, corner_thickness)
    cv2.line(img, (x, y), (x, y + corner_length), color, corner_thickness)
    cv2.line(img, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
    cv2.line(img, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
    cv2.line(img, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
    
    # 3. ETIQUETA
    percentage = int(confidence * 100)
    label = f"{emotion} {percentage}%"
    
    font_scale = max(0.9, img.shape[1] / 700)
    font_thickness = max(2, int(img.shape[1] / 450))
    font = cv2.FONT_HERSHEY_DUPLEX
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )
    
    padding = 18
    label_x = x
    label_y = y - text_height - padding * 2
    
    if label_y < 10:
        label_y = y + h + text_height + padding * 2
    
    if label_x + text_width + padding * 2 > img.shape[1]:
        label_x = img.shape[1] - text_width - padding * 2 - 5
    if label_x < 5:
        label_x = 5
    
    # Sombra de fondo
    shadow_offset = 4
    cv2.rectangle(
        img,
        (label_x + shadow_offset, label_y - text_height - padding + shadow_offset),
        (label_x + text_width + padding * 2 + shadow_offset, label_y + padding + shadow_offset),
        (0, 0, 0),
        -1
    )
    
    # Fondo de etiqueta
    cv2.rectangle(
        img,
        (label_x, label_y - text_height - padding),
        (label_x + text_width + padding * 2, label_y + padding),
        color,
        -1
    )
    
    # Borde blanco
    cv2.rectangle(
        img,
        (label_x, label_y - text_height - padding),
        (label_x + text_width + padding * 2, label_y + padding),
        (255, 255, 255),
        3
    )
    
    # Texto
    cv2.putText(
        img,
        label,
        (label_x + padding, label_y - padding // 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )
    
    return img


def generar_frames():
    """
    Generador que captura frames de la cámara, procesa emociones y los devuelve
    """
    global camera
    
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        
        # Hacer copia para procesar
        frame_display = frame.copy()
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detectar rostros con parámetros más estables
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # Más conservador
            minNeighbors=8,     # Más exigente para evitar falsos positivos
            minSize=(100, 100), # Tamaño mínimo aumentado
            maxSize=(400, 400)  # Tamaño máximo para evitar detecciones muy grandes
        )
        
        # Procesar rostros detectados
        if len(faces) > 0:
            # Ordenar por tamaño (más grande primero)
            faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            
            for (x, y, w, h) in faces_sorted[:1]:  # Solo el más grande
                # Analizar emoción con algoritmo mejorado
                emotions = analizar_emocion_opencv_mejorado(gray, (x, y, w, h), frame)
                
                # Obtener emoción principal
                top_emotion = max(emotions.items(), key=lambda item: item[1])
                emotion_name = top_emotion[0]
                emotion_confidence = top_emotion[1]
                
                # Dibujar recuadro
                frame_display = dibujar_recuadro_emocion(
                    frame_display,
                    (x, y, w, h),
                    emotion_name,
                    emotion_confidence
                )
        
        # Codificar frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame_display, [cv2.IMWRITE_JPEG_QUALITY, 88])
        
        if not ret:
            continue
        
        # Convertir a bytes
        frame_bytes = buffer.tobytes()
        
        # Yield en formato multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Pausa para mantener FPS estable
        time.sleep(0.033)  # ~30 FPS


def video_feed(request):
    """
    Vista que devuelve el stream de video procesado
    """
    return StreamingHttpResponse(
        generar_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def stop_camera(request):
    """
    Vista para detener la cámara
    """
    global camera
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    
    return JsonResponse({'success': True, 'message': 'Cámara detenida'})