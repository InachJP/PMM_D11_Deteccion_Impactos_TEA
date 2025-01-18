import os
import json
import shutil
import numpy as np
import gc
import cv2
import pykinect_azure as pykinect
from utils1 import calculate_velocity, filter_face_points, draw_face_points_with_effects, transform_point_3d_to_2d

CAPTURE_IMPACTS_DIRECTORY = "./capturas_impactos"
FRAME_RATE = 30  # Fotogramas por segundo
USE_EXTEND_KALMAN_FILTER = False
MOTION_DETECTED_THRESHOLD = 20
THRESHOLD_PROXIMITY = 30
TIME_BETWEEN_IMPACTS = 0.55

# Función para guardar las imágenes de impacto
def save_impact_image(image, impact_counter, base_directory, mode_name, video_name):
    filename = f"impact_{impact_counter:03d}.png"
    filepath = os.path.join(base_directory, filename)
    cv2.imwrite(filepath, image)

# Función para procesar un video y generar datos de impactos
def process_video(video_path, mode_name, results):
    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_path)
    recording_length = playback.get_recording_length() / 10e5  # Duración en segundos
    playback_calibration = playback.get_calibration()
    body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)

    impacts = {}
    frame_counter = 0
    impact_counter = 1
    last_impact_frame = -1
    min_frames_between_impacts = int(TIME_BETWEEN_IMPACTS * FRAME_RATE)

    right_hand_positions = []
    right_hand_times = []

    while True:
        ret, capture = playback.update()
        if not ret:
            break

        body_frame = body_tracker.update(capture=capture)
        ret_color, color_image = capture.get_color_image()
        ret_depth, transformed_colored_depth_image = capture.get_transformed_colored_depth_image()
        ret_points, pointcloud = capture.get_pointcloud()

        if not ret_color or not ret_depth or pointcloud.size == 0:
            frame_counter += 1
            continue

        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, transformed_colored_depth_image, 0.3, 0)

        for body_id in range(body_frame.get_num_bodies()):
            skeleton_3d = body_frame.get_body(body_id).numpy()
            neck_position = skeleton_3d[pykinect.K4ABT_JOINT_NECK, :3]
            hand_position = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]
            right_hand_positions.append(hand_position)
            right_hand_times.append(float(frame_counter / FRAME_RATE))

            if len(right_hand_positions) > 1:
                distance = np.linalg.norm(right_hand_positions[-2] - right_hand_positions[-1])
                if distance > MOTION_DETECTED_THRESHOLD:
                    face_points = filter_face_points(pointcloud, neck_position)

                    for point_3d in face_points:
                        distance = np.linalg.norm(hand_position - point_3d)
                        if distance < THRESHOLD_PROXIMITY and (frame_counter - last_impact_frame) > min_frames_between_impacts:
                            vel = calculate_velocity(right_hand_positions, right_hand_times)
                            impacts[impact_counter] = {
                                "Velocity [m/s]": vel,
                                "Time [s]": right_hand_times[-1]
                            }
                            processed_image = draw_face_points_with_effects(
                                image=combined_image.copy(),
                                face_points=face_points,
                                playback_calibration=playback_calibration,
                                gradient=True,
                                size_variation=True,
                                blur=True,
                                colormap='plasma',
                                alpha=0.6
                            )

                            impact_2d = transform_point_3d_to_2d(playback_calibration, point_3d)
                            hand_impact_2d = transform_point_3d_to_2d(playback_calibration, hand_position)

                            if impact_2d is not None:
                                cv2.circle(processed_image, impact_2d, 5, (0, 0, 255), -1)
                            if hand_impact_2d is not None:
                                cv2.circle(processed_image, hand_impact_2d, 5, (255, 0, 0), -1)

                            video_name = os.path.splitext(os.path.basename(video_path))[0]  # Nombre del video sin extensión
                            mode_directory = os.path.join(CAPTURE_IMPACTS_DIRECTORY, mode_name, video_name)
                            os.makedirs(mode_directory, exist_ok=True)

                            # Guardar la captura del impacto
                            save_impact_image(
                                image=processed_image,
                                impact_counter=impact_counter,
                                base_directory=mode_directory,
                                mode_name=mode_name,
                                video_name=video_name
                            )

                            last_impact_frame = frame_counter
                            impact_counter += 1
                            break
                else:
                    right_hand_positions = []
                    right_hand_times = []

        frame_counter += 1
    del playback, body_tracker, capture, color_image, pointcloud
    gc.collect()
    results[mode_name][os.path.basename(video_path)] = impacts

# Principal
if __name__ == "__main__":
    videos_directory = "./../videos"
    results = {}

    # Crear directorio para las capturas
    if os.path.exists(CAPTURE_IMPACTS_DIRECTORY):
        shutil.rmtree(CAPTURE_IMPACTS_DIRECTORY)
    os.makedirs(CAPTURE_IMPACTS_DIRECTORY)

    # Iterar sobre los modos de configuración
    for mode in os.listdir(videos_directory):
        mode_path = os.path.join(videos_directory, mode)
        if not os.path.isdir(mode_path):
            continue

        results[mode] = {}

        for video_file in os.listdir(mode_path):
            if video_file.endswith(".mkv"):
                video_path = os.path.join(mode_path, video_file)
                process_video(video_path, mode, results)
            

    # Guardar los resultados en un archivo JSON
    with open("impact_data.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Datos de impactos guardados en 'impact_data.json'.")

