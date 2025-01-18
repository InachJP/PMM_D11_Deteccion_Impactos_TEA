import numpy as np
import os
import json
import cv2
import time
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import matplotlib.pyplot as plt

from utils1 import * 

import shutil

# Directorio donde se almacenarán las imágenes

CAPTURE_IMPACTS_DIRECTORY = "./../capturas_impactos"
FRAME_RATE = 30  # Fotogramas por segundo 
USE_EXTEND_KALMAN_FILTER = False
MOTION_DETECTED_THRESHOLD = 20
THRESHOLD_PROXIMITY = 30
IMPACT_DATA_ANNOTATION = dict()
TIME_BETWEEN_IMPACTS = 0.55
right_hand_positions = []
right_hand_times = []

# Eliminar y crear el directorio
if os.path.exists(CAPTURE_IMPACTS_DIRECTORY):
    shutil.rmtree(CAPTURE_IMPACTS_DIRECTORY)  # Eliminar el directorio si existe
os.makedirs(CAPTURE_IMPACTS_DIRECTORY)  # Crear un nuevo directorio

# Principal
if __name__ == "__main__":
    # Inicializar Kinect y configuración
    video_filename = "./../videos/NFOV_2X2/videos NFOV 2X2/boca_NFOV_2x2_luz_artificial.mkv"
    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_filename)
    recording_length = playback.get_recording_length() / 10e5  # Duración del video en segundos
    playback_calibration = playback.get_calibration()
    body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)

    # Variables de inicialización
    impacts = []
    impact_counter = 1
    last_impact_frame = -1
    min_frames_between_impacts = int(TIME_BETWEEN_IMPACTS * FRAME_RATE)  # 300 ms convertidos a fotogramas
    total_frames = int(recording_length * FRAME_RATE)  # Total de fotogramas calculados
    frame_counter = 0  # Contador de fotogramas procesados

    print(f"Duración del video: {recording_length:.2f} segundos")
    print(f"Total de fotogramas estimados: {total_frames}")

    while True:
        ret, capture = playback.update()
        if not ret:
            break

        body_frame = body_tracker.update(capture=capture)
        # Obtener la imagen de color
        ret_color, color_image = capture.get_color_image()

        # Obtener la imagen de profundidad coloreada
        ret_depth, transformed_colored_depth_image = capture.get_transformed_colored_depth_image()

        # Obtener la nube de puntos
        ret_points, pointcloud = capture.get_pointcloud()

        if not ret_color or not ret_depth or pointcloud.size == 0:
            frame_counter += 1
            continue

        # Combinar las imágenes
        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, transformed_colored_depth_image, 0.3, 0)

        for body_id in range(body_frame.get_num_bodies()):
            skeleton_3d = body_frame.get_body(body_id).numpy()
            neck_position = skeleton_3d[pykinect.K4ABT_JOINT_NECK, :3]
            hand_position = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]
            right_hand_positions.append(hand_position)
            right_hand_times.append(float(frame_counter/FRAME_RATE)) #en segundos

            if len(right_hand_positions) > 1:
                distance = np.linalg.norm(right_hand_positions[-2] - right_hand_positions[-1])
                if distance > MOTION_DETECTED_THRESHOLD:

                    face_points = filter_face_points(pointcloud, neck_position)

                    # Detectar impacto
                    for point_3d in face_points:   

                        distance = np.linalg.norm(hand_position - point_3d)  
                        if distance < THRESHOLD_PROXIMITY and (frame_counter - last_impact_frame) > min_frames_between_impacts:
                            
                            
                            vel = calculate_velocity(right_hand_positions, right_hand_times)
                            IMPACT_DATA_ANNOTATION[impact_counter] = {
                                "Velocity [m/s]": vel,
                                "Time [s]": right_hand_times[-1]
                            }
                                
                            # Dibujar puntos faciales con efectos una sola vez
                            processed_image = draw_face_points_with_effects(
                                image=combined_image.copy(),  # Trabaja sobre una copia
                                face_points=face_points,
                                playback_calibration=playback_calibration,
                                gradient=True,
                                size_variation=True,
                                blur=True,
                                colormap='plasma',
                                alpha=0.6
                            )

                            # Transformar puntos de impacto
                            impact_2d = transform_point_3d_to_2d(playback_calibration, point_3d)
                            hand_impact_2d = transform_point_3d_to_2d(playback_calibration, hand_position)

                            # Pintar puntos de impacto en la imagen procesada
                            if impact_2d is not None:
                                cv2.circle(processed_image, impact_2d, 5, (0, 0, 255), -1)  # Rojo para impacto en rostro
                            if hand_impact_2d is not None:
                                cv2.circle(processed_image, hand_impact_2d, 5, (255, 0, 0), -1)  # Azul para impacto en mano

                            # Guardar imagen del impacto
                            save_impact_image(processed_image, impact_counter, CAPTURE_IMPACTS_DIRECTORY)

                            # Actualizar contador y tiempo del último impacto
                            last_impact_frame = frame_counter
                            impact_counter += 1
                            break
                else:
                    right_hand_positions = []
                    right_hand_times = []

        # Incrementar el contador de fotogramas
        frame_counter += 1

        # Salir si se presiona "q"
        if cv2.waitKey(1) == ord("q"):
            break

    # Guardar impactos en archivo JSON
    with open("impact_data.json", "w") as f:
        json.dump(IMPACT_DATA_ANNOTATION, f, indent=4)

    print("Impactos guardados en 'impact_data.json'.")

