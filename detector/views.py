
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Ruta al modelo preentrenado (debes descargarlo)
MODEL_PATH = 'models/emotion_model.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Diccionario de emociones
EMOTIONS = {
    0: 'Enojado',
    1: 'Disgusto',
    2: 'Miedo',
    3: 'Feliz',
    4: 'Neutral',
    5: 'Triste',
    6: 'Sorprendido'
}

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def index(request):
    """Vista principal que renderiza el template HTML"""
    return render(request, 'detector/index.html')


@csrf_exempt
def detectar_emocion(request):
    """Vista que procesa la imagen y detecta emociones"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Método no permitido'}, status=405)
    
    try:
        # Obtener datos JSON
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'success': False, 'error': 'No se proporcionó imagen'}, status=400)
        
        # Decodificar imagen base64
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convertir a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return JsonResponse({
                'success': False,
                'error': 'No se detectó ningún rostro en la imagen'
            })
        
        # Tomar el primer rostro detectado
        (x, y, w, h) = faces[0]
        
        # Extraer región de interés (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Normalizar y preparar para el modelo
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        # Simular predicción (si no tienes modelo entrenado)
        # En producción, descomentar estas líneas:
        # model = load_model(MODEL_PATH)
        # predictions = model.predict(roi_gray)[0]
        
        # Predicción simulada para demostración
        predictions = np.random.dirichlet(np.ones(7))
        
        # Crear diccionario de emociones con confianzas
        emotions = {EMOTIONS[i]: float(predictions[i]) for i in range(len(predictions))}
        
        # Ordenar por confianza
        emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
        
        return JsonResponse({
            'success': True,
            'emotions': emotions,
            'face_detected': True,
            'face_coordinates': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def entrenar_modelo(request):
    """Vista opcional para entrenar el modelo (usar con cuidado)"""
    # Aquí puedes implementar lógica de entrenamiento si es necesario
    return JsonResponse({'message': 'Función de entrenamiento no implementada'})