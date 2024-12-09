import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pykinect_azure as pykinect
from scipy.spatial import Delaunay

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

def extract_depth_points(capture, calibration):
    """
    Extrae la nube de puntos 3D del mapa de profundidad.
    """
    ret, depth_image = capture.get_depth_image()
    if not ret:
        return None

    point_cloud = capture.convert_depth_to_points(calibration)
    return point_cloud

def euclidean_distance(point, surface):
    """
    Calcula la distancia euclidiana de un punto a una superficie.
    """
    distances = np.linalg.norm(surface - point, axis=1)
    return distances.min()

if __name__ == "__main__":
    # Inicializar PyKinect
    video_filename = "test1_kalman.mkv"
    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_filename)
    playback_calibration = playback.get_calibration()
    bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)

    # Inicialización del EKF
    state, P, Q, R = initialize_ekf()
    dt = 1 / 30.0  # 30 FPS

    # Lista para almacenar datos
    tracking_data = []

    while True:
        ret, capture = playback.update()
        if not ret:
            break

        body_frame = bodyTracker.update(capture=capture)

        for body_id in range(body_frame.get_num_bodies()):
            skeleton_3d = body_frame.get_body(body_id).numpy()

            # Observaciones actuales (coordenadas de la mano derecha)
            z_k = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]

            # EKF: Predicción y corrección
            state, P = ekf_predict(state, P, Q, dt)
            state, P = ekf_update(state, P, z_k, R)

            # Extraer la nube de puntos del mapa de profundidad
            depth_points = extract_depth_points(capture, playback_calibration)

            if depth_points is None:
                continue

            # Calcular la distancia euclidiana a la superficie de la cara
            distance_original = euclidean_distance(z_k, depth_points)
            distance_smoothed = euclidean_distance(state[:3], depth_points)

            # Registrar datos
            tracking_data.append({
                "original_coordinates": z_k.tolist(),
                "smoothed_coordinates": state[:3].tolist(),
                "distance_to_face_original": distance_original,
                "distance_to_face_smoothed": distance_smoothed
            })

        if len(tracking_data) > 300:  # Límite para pruebas
            break

    # Guardar datos en un archivo JSON
    with open("hand_tracking_data_depth.json", "w") as f:
        json.dump(tracking_data, f, indent=4)
    print("Datos guardados en 'hand_tracking_data_depth.json'.")

    # Graficar trayectorias 3D y nube de puntos
    original_positions = np.array([data["original_coordinates"] for data in tracking_data])
    smoothed_positions = np.array([data["smoothed_coordinates"] for data in tracking_data])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Nube de puntos de profundidad
    ax.scatter(
        depth_points[:, 0],
        depth_points[:, 1],
        depth_points[:, 2],
        c="lightgray",
        s=1,
        alpha=0.5,
        label="Nube de puntos (Rostro)"
    )

    # Trayectoria original
    ax.plot(
        original_positions[:, 0],
        original_positions[:, 1],
        original_positions[:, 2],
        label="Original",
        color="red",
        marker="o",
        linestyle="dotted",
    )

    # Trayectoria suavizada
    ax.plot(
        smoothed_positions[:, 0],
        smoothed_positions[:, 1],
        smoothed_positions[:, 2],
        label="Suavizada (EKF)",
        color="blue",
        marker="^",
        linestyle="solid",
    )

    ax.set_title("Trayectoria 3D de la Mano Derecha y Nube de Puntos del Rostro", fontsize=16)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)
    ax.legend()

    plt.show()
