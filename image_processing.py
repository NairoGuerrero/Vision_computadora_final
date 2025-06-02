import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt


class PlacaProcessor:
    """
    Clase para procesar imágenes de placas vehiculares. Permite detectar la placa,
    corregir la perspectiva y segmentar los caracteres.

    Atributos:
        img_size (int): Tamaño al que se redimensionarán los caracteres segmentados.
        debug (bool): Indica si se deben mostrar imágenes intermedias para depuración.
    """

    def __init__(self, img_size: int = 227, debug: bool = True):
        """
        Inicializa el procesador de placas.

        Args:
            img_size (int): Tamaño deseado de las imágenes de caracteres segmentados.
            debug (bool): Si es True, se mostrarán imágenes de depuración.
        """
        self.img_size = img_size
        self.debug = debug

    def ordenar_puntos(self, puntos: np.ndarray) -> np.ndarray:
        """
        Ordena los 4 puntos de una figura cuadrilátera en el orden:
        superior izquierda, superior derecha, inferior derecha, inferior izquierda.

        Args:
            puntos (np.ndarray): Array de 4 puntos (x, y).

        Returns:
            np.ndarray: Puntos ordenados como float32.
        """
        puntos = puntos.reshape((4, 2))
        suma = puntos.sum(axis=1)
        resta = np.diff(puntos, axis=1)

        ordenados = np.zeros((4, 2), dtype="float32")
        ordenados[0] = puntos[np.argmin(suma)]     # superior izquierda
        ordenados[2] = puntos[np.argmax(suma)]     # inferior derecha
        ordenados[1] = puntos[np.argmin(resta)]    # superior derecha
        ordenados[3] = puntos[np.argmax(resta)]    # inferior izquierda

        return ordenados

    def mostrar(self, img: np.ndarray, title: str = "", cmap=None) -> None:
        """
        Muestra una imagen utilizando Matplotlib si está habilitado el modo debug.

        Args:
            img (np.ndarray): Imagen a mostrar.
            title (str): Título de la imagen.
            cmap: Mapa de colores a usar (por ejemplo 'gray').
        """
        if self.debug:
            plt.figure(figsize=(10, 4))
            plt.title(title)
            if cmap:
                plt.imshow(img, cmap=cmap)
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()


    def detectar_y_corregir_placa(self, img: np.ndarray,
                                   width: int = 1280, height: int = 641) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Detecta la región de una placa en la imagen y corrige su perspectiva.

        Args:
            img (np.ndarray): Imagen original BGR.
            width (int): Ancho deseado de la imagen corregida.
            height (int): Alto deseado de la imagen corregida.

        Returns:
            tuple: Imagen de la placa corregida y versión en escala de grises.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Umbral para color amarillo típico de placas
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Limpieza morfológica de la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.mostrar(mask_clean, "Máscara Amarilla", cmap='gray')

        # Detección de contornos
        contours = imutils.grab_contours(
            cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            aspect_ratio = w / float(h)
            rectangularidad = area / float(w * h)

            if rectangularidad >= 0.7 and len(approx) == 4 and 1.5 <= aspect_ratio <= 3.0:
                location = approx
                break

        if location is None:
            print("❌ No se encontró una placa.")
            return None, None

        # Transformación de perspectiva
        pts_src = self.ordenar_puntos(location.astype(np.int32))
        pts_dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img, M, (width, height))
        self.mostrar(warped, "Placa Corregida")

        return warped, gray


    def segmentar_caracteres(self, warped: np.ndarray) -> list[np.ndarray]:
        """
        Segmenta los caracteres de una imagen de placa ya corregida.

        Args:
            warped (np.ndarray): Imagen corregida (output de detectar_y_corregir_placa).

        Returns:
            list[np.ndarray]: Lista de imágenes de caracteres redimensionados.
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.mostrar(thresh, "Binarización", cmap='gray')

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        char_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filtro basado en tamaño aproximado de caracteres
            if 100 <= w <= 250 and 300 <= h <= 400:
                char_boxes.append((x, y, w, h))
                cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ordenar de izquierda a derecha
        char_boxes = sorted(char_boxes, key=lambda b: b[0])
        matricula = []

        for i, (x, y, w, h) in enumerate(char_boxes):
            char = thresh[max(0, y-5):y+h+10, max(0, x-5):x+w+10]
            resized = cv2.resize(char, (self.img_size, self.img_size))
            matricula.append(resized)

        # Mostrar resultados si está activado el modo debug
        if self.debug:
            plt.figure(figsize=(12, 3))
            for i, char_img in enumerate(matricula):
                plt.subplot(1, len(matricula), i+1)
                plt.imshow(char_img, cmap='gray')
                plt.title(f"Char {i+1}")
                plt.axis("off")
            plt.suptitle("Caracteres Recortados")
            plt.show()

        return matricula
