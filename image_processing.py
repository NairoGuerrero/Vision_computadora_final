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