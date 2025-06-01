from ultralytics import YOLO
import os

def main():
    # Ruta al archivo data.yaml
    data_path = "Trabajo final vision por computador/data.yaml"

    # Back-bones a utilizar
    #backbones = ["yolov8n.pt", "yolov8s.pt"]
    backbones = ["yolov8n.pt"]

    for backbone in backbones:
        # Cargar modelo preentrenado
        model = YOLO(backbone)
        model_name = os.path.splitext(os.path.basename(backbone))[0]
        print(f"Modelo {model_name} cargado")

        # Entrenamiento
        results = model.train(
            data=data_path,
            epochs=50,
            patience=0,
            imgsz=640,
            batch=16,
            name=f"Localizacion placa colombiana {model_name}",
            project="runs/detect",
            verbose=True,
            device=0  # asegura uso de GPU
        )

if __name__ == '__main__':
    main()
