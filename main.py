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

    def __init__(self,
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
        
    def procesar_frame(self, frame):
        """
        Procesa un solo frame para detección, conteo y OCR.

        Args:
            frame (np.ndarray): Imagen del frame actual.
        """
        results_car = self.model_car(frame)[0]

        for box in results_car.boxes:
            cls = int(box.cls[0])
            label = self.model_car.names[cls]

            if label in ['car', 'bus', 'truck']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                cx = (x1 + x2) // 2
                cy = y2
                center = (cx, cy)

                side = self.is_left_of_line(center, self.line_pt1, self.line_pt2)
                match_idx = self.find_matching_vehicle(cx, cy)

                if match_idx != -1:
                    prev = self.seen_vehicles[match_idx]
                    prev_side = prev[2]
                    already_counted = prev[3]

                    if side * prev_side < 0 and not already_counted:
                        self.vehicle_count += 1
                        self.seen_vehicles[match_idx] = (cx, cy, side, True, time.time())
                        print(f"🚗 Contado: {self.vehicle_count}")

                        car_region = frame[y1:y2, x1:x2]
                        results_plate = self.model_plate(car_region)[0]

                        if results_plate.boxes:
                            best_plate = max(results_plate.boxes, key=lambda b: b.conf[0])
                            px1, py1, px2, py2 = map(int, best_plate.xyxy[0])
                            abs_x1 = x1 + px1
                            abs_y1 = y1 + py1
                            abs_x2 = x1 + px2
                            abs_y2 = y1 + py2

                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                            label = f"{self.model_plate.names[int(best_plate.cls[0])]} {best_plate.conf[0]:.2f}"
                            cv2.putText(frame, label, (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            plate_img = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                            if plate_img.size > 0:
                                warped, _ = self.processor.detectar_y_corregir_placa(plate_img)
                                if warped is not None:
                                    caracteres = self.processor.segmentar_caracteres(warped)
                                    self.resultado = self.recognizer.reconocer_matricula(caracteres, mostrar_prob=True)

                    else:
                        self.seen_vehicles[match_idx] = (cx, cy, side, already_counted, time.time())
                else:
                    self.seen_vehicles.append((cx, cy, side, False, time.time()))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar línea virtual y anotaciones
        cv2.line(frame, self.line_pt1, self.line_pt2, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehiculos contados: {self.vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, f"Matricula: {self.resultado}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Eliminar vehículos viejos
        now = time.time()
        self.seen_vehicles = [v for v in self.seen_vehicles if now - v[4] < 8]

        self.out.write(frame)
        cv2.imshow("Conteo Vehicular", frame)

    def ejecutar(self):
        """
        Ejecuta el procesamiento completo del video hasta el final.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.procesar_frame(frame)

            if cv2.waitKey(1) == 27:  # ESC
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# === Ejemplo de uso ===
if __name__ == "__main__":
    detector = VehiculoDetector(video_path="20250527_093042.mp4")
    detector.ejecutar()
