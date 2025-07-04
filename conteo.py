from fastapi import FastAPI, Query, Response
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import torch
import threading
import atexit

# Inicializa el servidor FastAPI
app = FastAPI()

# Modelo YOLOv5 cargado una sola vez
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)

# Cámara global
cap = cv2.VideoCapture(0)
lock = threading.Lock()

def detectar_personas(frame):
    results = model(frame)
    df = results.pandas().xyxy[0]
    people = df[df['name'] == 'person']
    return people

@app.get("/count")
def count_people(aulaName: str = Query(...)):
    # Abrimos y liberamos cámara localmente (no usamos la global `cap`)
    temp_cap = cv2.VideoCapture(0)
    ret, frame = temp_cap.read()
    temp_cap.release()

    if not ret:
        return JSONResponse(status_code=500, content={"success": False, "error": "No se pudo capturar la imagen"})

    people = detectar_personas(frame)

    return {
        "success": True,
        "aulaName": aulaName,
        "cantidad": len(people)
    }

def generar_frames():
    while True:
        with lock:
            ret, frame = cap.read()
        if not ret:
            continue

        people = detectar_personas(frame)

        # Dibujar detecciones
        for _, row in people.iterrows():
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),
                          (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)

        cv2.putText(frame, f'Personas: {len(people)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Codificar a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(generar_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# Cierra la cámara al finalizar
@atexit.register
def cleanup():
    cap.release()