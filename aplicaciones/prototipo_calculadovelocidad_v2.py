import numpy as np
import json
import cv2
import time
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import matplotlib.pyplot as plt

def draw_face_points_with_effects(image, face_points, playback_calibration, gradient=True, size_variation=True, blur=False, colormap='viridis', alpha=0.6):
    """
    Dibuja puntos faciales proyectados en una imagen con efectos visuales como gradientes, tamaños dinámicos y desenfoque.
    
    Args:
        image (np.ndarray): La imagen 2D sobre la que se dibujarán los puntos.
        face_points (np.ndarray): Array de puntos 3D correspondientes al mesh facial.
        playback_calibration: Calibración de la cámara para transformar 3D a 2D.
        gradient (bool): Si es True, aplica un gradiente de colores basado en la profundidad.
        size_variation (bool): Si es True, ajusta el tamaño del punto según su profundidad.
        blur (bool): Si es True, aplica desenfoque gaussiano a la imagen resultante.
        colormap (str): El colormap usado para el gradiente (usando matplotlib).
        alpha (float): Transparencia para superponer los puntos en la imagen.
    
    Returns:
        np.ndarray: Imagen con puntos faciales dibujados.
    """
    # Copiar la imagen base para dibujar los puntos
    overlay = image.copy()

    # Normalizar la profundidad para gradiente y tamaño dinámico
    if len(face_points) > 0:
        z_min = np.min(face_points[:, 2])
        z_max = np.max(face_points[:, 2])
    else:
        z_min, z_max = 0, 1  # Evitar división por cero si no hay puntos

    cmap = plt.get_cmap(colormap)

    for point_3d in face_points:
        # Transformar punto 3D a 2D
        face_point_2d = transform_point_3d_to_2d(playback_calibration, point_3d)
        if face_point_2d is None:
            continue

        # Normalizar profundidad
        depth_normalized = (point_3d[2] - z_min) / (z_max - z_min) if z_max > z_min else 0

        # Aplicar gradiente de colores
        if gradient:
            color_normalized = cmap(depth_normalized)
            color = (
                int(color_normalized[0] * 255),  # R
                int(color_normalized[1] * 255),  # G
                int(color_normalized[2] * 255),  # B
            )
        else:
            color = (0, 255, 0)  # Verde por defecto

        # Ajustar tamaño según profundidad
        size = max(1, int(5 * (1 - depth_normalized))) if size_variation else 2

        # Dibujar punto en la imagen
        cv2.circle(overlay, face_point_2d, size, color, -1)

    # Superposición transparente
    combined_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Aplicar desenfoque gaussiano si es necesario
    if blur:
        combined_image = cv2.GaussianBlur(combined_image, (5, 5), 0)

    return combined_image

# Funciones auxiliares (transformación de puntos y filtro)
def transform_point_3d_to_2d(calibration, point_3d, source_camera=_k4a.K4A_CALIBRATION_TYPE_DEPTH, target_camera=_k4a.K4A_CALIBRATION_TYPE_COLOR):
    """
    Transforma un punto 3D desde el sistema de coordenadas de la cámara de profundidad
    al espacio 2D de la cámara de color.
    """
    source_point3d = _k4a.k4a_float3_t()
    source_point3d.xyz.x = point_3d[0]
    source_point3d.xyz.y = point_3d[1]
    source_point3d.xyz.z = point_3d[2]
    try:
        target_point2d = calibration.convert_3d_to_2d(
            source_point3d=source_point3d,
            source_camera=source_camera,
            target_camera=target_camera
        )
        return (int(target_point2d.xy.x), int(target_point2d.xy.y))
    except Exception as e:
        print(f"Error al transformar punto 3D a 2D: {e}")
        return None

# Inicialización del EKF
def initialize_ekf():
    state = np.zeros(6)  # [x, y, z, vx, vy, vz]
    P = np.eye(6) * 0.1  # Covarianza inicial
    Q = np.eye(6) * 0.1  # Ruido del proceso
    R = np.eye(3) * 0.05  # Ruido de observación
    return state, P, Q, R

def fx(state, dt):
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F @ state

def hx(state):
    return state[:3]

def jacobian_F(state, dt):
    J = np.eye(6)
    J[0, 3] = dt
    J[1, 4] = dt
    J[2, 5] = dt
    return J

def jacobian_H(state):
    J = np.zeros((3, 6))
    J[0, 0] = 1
    J[1, 1] = 1
    J[2, 2] = 1
    return J

def ekf_predict(state, P, Q, dt):
    F = jacobian_F(state, dt)
    state = fx(state, dt)
    P = F @ P @ F.T + Q
    return state, P

def ekf_update(state, P, z, R):
    H = jacobian_H(state)
    y = z - hx(state)  # Residual
    S = H @ P @ H.T + R  # Covarianza de innovación
    K = P @ H.T @ np.linalg.inv(S)  # Ganancia de Kalman
    state = state + K @ y
    P = P - K @ H @ P
    return state, P

# Filtrar puntos faciales con margen adicional en y
def filter_face_points(points, neck_position, y_margin=50, x_margin=100, z_depth_limit=200):
    """
    Filtra puntos faciales en función de los márgenes dados respecto al cuello.

    Args:
        points: Nube de puntos 3D (np.ndarray).
        neck_position: Coordenadas 3D del cuello (np.array).
        y_margin: Margen adicional hacia arriba en el eje Y (mm).
        x_margin: Margen en el eje X (mm).
        z_depth_limit: Margen de profundidad en el eje Z (mm).

    Returns:
        np.ndarray: Puntos filtrados.
    """
    neck_position = np.array(neck_position)

    # Margen adicional hacia arriba en el eje Y
    above_neck = points[:, 1] < (neck_position[1] - y_margin)

    # Dentro de los márgenes en X y Z
    within_x = (points[:, 0] >= neck_position[0] - x_margin) & (points[:, 0] <= neck_position[0] + x_margin)
    within_z = (points[:, 2] <= neck_position[2]) & (points[:, 2] >= neck_position[2] - z_depth_limit)

    # Retornar puntos que cumplan las condiciones
    return points[above_neck & within_x & within_z]


# Calcular velocidad
def calculate_velocity(positions, times):
    if len(positions) < 2:
        return 0, 0
    pos1 = np.array(positions[-2])
    pos2 = np.array(positions[-1])
    time1, time2 = times[-2], times[-1]
    delta_time = time2 - time1
    if delta_time == 0:
        return 0, 0
    distance = np.linalg.norm(pos2 - pos1)
    velocity = distance / delta_time / 1000  # Convertir a m/s
    return velocity, distance / 1000

# Principal
if __name__ == "__main__":
    # Inicializar Kinect y configuración
    video_filename = "test1_kalman.mkv"
    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_filename)
    playback_calibration = playback.get_calibration()
    body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)

    # Variables de inicialización
    impacts = []
    impact_counter = 1
    last_impact_time = 0
    min_time_between_impacts = 0.3  # 300 ms entre impactos consecutivos
    start_time = time.time()

    while True:
        ret, capture = playback.update()
        if not ret:
            break

        body_frame = body_tracker.update(capture=capture)
        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

		# Get the colored depth
        ret_depth, transformed_colored_depth_image = capture.get_transformed_colored_depth_image()

        # Get the point cloud
        ret_points, pointcloud = capture.get_pointcloud()

        if not ret_color or not ret_depth or pointcloud.size == 0:
            continue

		# Combine both images
        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, transformed_colored_depth_image, 0.3, 0)

        for body_id in range(body_frame.get_num_bodies()):
            skeleton_3d = body_frame.get_body(body_id).numpy()
            neck_position = skeleton_3d[pykinect.K4ABT_JOINT_NECK, :3]
            hand_position = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]

            # Filtrar puntos faciales
            face_points = filter_face_points(pointcloud, neck_position)

            # Detectar impacto
            current_time = time.time()
            for point_3d in face_points:
                distance = np.linalg.norm(hand_position - point_3d)
                if distance < 50 and (current_time - last_impact_time) > min_time_between_impacts:

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
                    impact_filename = f"impact_{impact_counter}.png"
                    cv2.imwrite(impact_filename, processed_image)

                    # Actualizar contador y tiempo del último impacto
                    last_impact_time = current_time
                    impact_counter += 1
                    break

        # Mostrar la visualización con impactos y mesh facial
        #cv2.imshow("Impact Visualization", combined_image)

        if cv2.waitKey(1) == ord("q"):
            break

    # Guardar impactos en archivo JSON
    with open("impact_data.json", "w") as f:
        json.dump(impacts, f, indent=4)

    print("Impactos guardados en 'impact_data.json'.")