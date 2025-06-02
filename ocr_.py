"""

Clase OCRRecognizer: realiza reconocimiento de caracteres (OCR) en imágenes individuales
usando un modelo entrenado con Keras basado en AlexNet.


"""

import numpy as np
from tensorflow.keras.models import load_model


class OCRRecognizer:
    """
    Clase encargada de preprocesar caracteres individuales y predecir su valor
    usando un modelo de red neuronal convolucional previamente entrenado.

    Atributos:
        model: Modelo Keras cargado para OCR.
        chars (str): Conjunto de caracteres reconocibles por el modelo.
        img_size (int): Tamaño de entrada de cada imagen de caracter.
    """

    def __init__(self, model_path: str,
                 chars: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                 img_size: int = 227):
        """
        Inicializa el OCR cargando el modelo entrenado.

        Args:
            model_path (str): Ruta al modelo .keras entrenado.
            chars (str): Conjunto de caracteres válidos a reconocer.
            img_size (int): Dimensión de entrada del modelo (cuadrado).
        """
        self.model = load_model(model_path)
        self.chars = chars
        self.img_size = img_size

    def preprocesar(self, char_img: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen de caracter individual para la red neuronal:
        normaliza, agrega canal y dimensión batch.

        Args:
            char_img (np.ndarray): Imagen binaria del caracter (227x227).

        Returns:
            np.ndarray: Tensor listo para ser inferido por el modelo.
        """
        char_input = char_img.astype(np.float32) / 255.0  # Normaliza a [0, 1]
        char_input = np.expand_dims(char_input, axis=-1)  # Agrega canal (H, W, 1)
        char_input = np.expand_dims(char_input, axis=0)   # Agrega batch (1, H, W, 1)
        return char_input

    def reconocer_matricula(self, char_imgs: list[np.ndarray], mostrar_prob: bool = False) -> str:
        """
        Reconoce una secuencia de caracteres a partir de sus imágenes.

        Args:
            char_imgs (list[np.ndarray]): Lista de imágenes de caracteres segmentados.
            mostrar_prob (bool): Si es True, muestra probabilidades por caracter.

        Returns:
            str: Cadena de texto que representa la matrícula reconocida.
        """
        resultado = ""

        for i, char_img in enumerate(char_imgs):
            input_tensor = self.preprocesar(char_img)
            pred = self.model.predict(input_tensor, verbose=0)

            class_index = int(np.argmax(pred))
            predicted_char = self.chars[class_index]
            resultado += predicted_char

            if mostrar_prob:
                print(
                    f"→ Char {i + 1}: {predicted_char} | Confianza: {np.max(pred):.2f} | Probabilidades: {np.round(pred, 2)}"
                )

        print(f"\n🚘 Matrícula reconocida: {resultado}")
        return resultado