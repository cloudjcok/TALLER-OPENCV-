from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2

# Create your views here.

# Vista para la camara con OpenCV
video_cam = None
_camara_started = False


def _lazy_start():
    global video_cam, _camara_started
    if not _camara_started:
        video_cam = cv2.VideoCapture(0)
        _camara_started = True


def gen():
    while True:
        _lazy_start()
        if video_cam is None:
            print("video_cam es None")
            time.sleep(0.1)
            continue
        ret, frame = video_cam.read()
        if not ret:
            print("No se pudo leer frame")
            time.sleep(0.1)
            continue
        print("Frame capturado")
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error al codificar JPEG")
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


# Vista para el streaming de la cámara
def video_feed_cam(request):
    return StreamingHttpResponse(gen(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

# Vista para mostrar el template HTML
def index(request):
    return render(request, 'core/index.html')

    
