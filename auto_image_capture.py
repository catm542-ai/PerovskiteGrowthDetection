# -*- coding: utf-8 -*-
"""

@author: tinajero
"""

import imageio
import time

def tomar_foto(nombre_archivo, camara_id):
    """
    Captures a single frame from the specified camera and saves it as an image.

    Parameters:
        nombre_archivo (str): The name of the output image file.
        camara_id (int): The camera ID (typically 0 or 1, depending on the system).
    """
    print(f"Accessing camera {camara_id}...")

    reader = imageio.get_reader(f"<video{camara_id}>")
    for i, frame in enumerate(reader):
        imageio.imwrite(nombre_archivo, frame)
        print(f"Image saved as {nombre_archivo}")
        break  # Capture only the first available frame

# Image resolution settings (not actively used but documented for system reference)
ancho = 1920
alto = 1080

# Continuous loop for alternating image capture from two cameras
while True:
    # Capture image from camera 0 and wait 2 minutes
    nombre_archivo_c0 = f"foto_0_{int(time.time())}.jpg"
    tomar_foto(nombre_archivo_c0, 0)
    time.sleep(120)

    # Capture image from camera 1 and wait 2 minutes
    nombre_archivo_c1 = f"foto_1_{int(time.time())}.jpg"
    tomar_foto(nombre_archivo_c1, 1)
    time.sleep(120)