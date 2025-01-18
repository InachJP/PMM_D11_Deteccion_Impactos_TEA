import numpy as np
import os
import json
import cv2
import time
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import matplotlib.pyplot as plt

# Limpiar el directorio antes de guardar imágenes
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):  # Verifica que sea un archivo
                os.remove(file_path)
    else:
        os.makedirs(dir_path)  # Crea el directorio si no existe

# utils.py
def save_impact_image(processed_image, impact_counter, directory):
    impact_filename = f"{directory}/impact_{impact_counter}.png"
    cv2.imwrite(impact_filename, processed_image)



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
    P = np.eye(6) * 10  # Covarianza inicial
    Q = np.eye(6) * 1  # Ruido del proceso
    R = np.eye(3) * 1  # Ruido de observación
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

def ekf_update(state, P, z, R): # Convertir coordenadas de milímetros a metros
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
    pos1 = np.array(positions[0])
    pos2 = np.array(positions[-1])
    time1, time2 = times[0], times[-1]
    delta_time = abs(time2 - time1)
    delta_distance = np.linalg.norm(pos2 - pos1)
    velocity = float((delta_distance / 1000)/ delta_time) # Convertir a m/s
    return velocity