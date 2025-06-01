"""

Clase para detección jerárquica de vehículos y placas usando YOLOv8,
procesamiento de imagen y OCR personalizado.

"""

from ultralytics import YOLO
import cv2
import time
from ocr import OCRRecognizer
from procesamiento_imagen import PlacaProcessor


class VehiculoDetector:
    """
    Clase que encapsula la lógica de conteo vehicular, detección de placas y reconocimiento OCR.
    """

    def _init_(self,
                 video_path: str,
                 output_path: str = "video_anotado.mp4",
                 placa_model_path: str = "runs/detect/Localizacion placa colombiana yolov8n/weights/best.pt",
                 ocr_model_path: str = "modelo_chars74k_36clases_ampliado.keras"):
        """
        Inicializa el detector de vehículos y OCR.

        Args:
            video_path (str): Ruta al video de entrada.
            output_path (str): Ruta del video de salida anotado.
            placa_model_path (str): Ruta del modelo de detección de placas.
            ocr_model_path (str): Ruta del modelo OCR.
        """
        self.cap = cv2.VideoCapture(video_path)
        self.model_car = YOLO("yolov8n.pt")
        self.model_plate = YOLO(placa_model_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.line_pt1 = (2000, 400)
        self.line_pt2 = (0, 700)

        self.vehicle_count = 0
        self.seen_vehicles = []  # (x, y, side, counted, timestamp)
        self.resultado = ""

        self.processor = PlacaProcessor(debug=False)
        self.recognizer = OCRRecognizer(ocr_model_path)

    def is_left_of_line(self, pt, a, b):
        """
        Determina si un punto está a la izquierda o derecha de la línea definida por a y b.
        """
        x, y = pt
        x1, y1 = a
        x2, y2 = b
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def find_matching_vehicle(self, cx, cy, max_dist=60):
        """
        Verifica si un vehículo actual ya ha sido visto previamente.

        Args:
            cx (int): Coordenada x del centro.
            cy (int): Coordenada y del centro.
            max_dist (int): Distancia máxima para considerar una coincidencia.

        Returns:
            int: Índice del vehículo encontrado o -1.
        """
        for i, (px, py, _, _, _) in enumerate(self.seen_vehicles):
            if abs(cx - px) < max_dist and abs(cy - py) < max_dist:
                return i
        return -1