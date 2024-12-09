import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pykinect_azure as pykinect

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

def filter_points_above_neck(points, neck_position, x_margin=100, z_depth_limit=200):
    """
    Filtra los puntos dentro de la nube segmentada del cuerpo basados en:
    - Posición del cuello
    - Rango en X alrededor del cuello (+/- x_margin)
    - Rango en Z relativo al cuello (- z_depth_limit)
    """
    neck_position = np.array(neck_position)

    # Filtrar puntos por encima del cuello en Y
    above_neck = points[:, 1] < neck_position[1]

    # Filtrar puntos dentro del rango X alrededor del cuello
    within_x = (points[:, 0] >= neck_position[0] - x_margin) & (points[:, 0] <= neck_position[0] + x_margin)

    # Filtrar puntos dentro del rango Z relativo al cuello
    within_z = (points[:, 2] <= neck_position[2]) & (points[:, 2] >= neck_position[2] - z_depth_limit)

    # Combinar las condiciones
    face_points = points[above_neck & within_x & within_z]

    print(f"Filtrando puntos. Total antes: {len(points)}, después: {len(face_points)}")
    return face_points

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

            # Coordenadas de la mano derecha
            z_k = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]

            # EKF: Predicción y corrección
            state, P = ekf_predict(state, P, Q, dt)
            state, P = ekf_update(state, P, z_k, R)

            # Coordenadas del cuello
            neck_position = skeleton_3d[pykinect.K4ABT_JOINT_NECK, :3]

            # Obtener la nube de puntos segmentada del cuerpo
            ret_points, body_points = capture.get_pointcloud()
            if ret_points:
                body_points = body_points.reshape(-1, 3)

                # Filtrar puntos para el rostro
                filtered_points = filter_points_above_neck(body_points, neck_position, x_margin=100, z_depth_limit=200)

                # Registrar datos
                tracking_data.append({
                    "original_coordinates": z_k.tolist(),
                    "smoothed_coordinates": state[:3].tolist(),
                    "face_points": filtered_points.tolist()
                })

        if len(tracking_data) > 300:  # Límite para pruebas
            break

        #Guardar datos en JSON solo con coordenadas originales y suavizadas
        with open("tracking_data.json", "w") as f:
            # Crear una nueva lista con solo los campos necesarios
            filtered_tracking_data = [
                {
                    "original_coordinates": data["original_coordinates"],
                    "smoothed_coordinates": data["smoothed_coordinates"],
                }
                for data in tracking_data
            ]
            # Guardar en JSON
            json.dump(filtered_tracking_data, f, indent=4)

    # Visualización de trayectorias y puntos filtrados
    original_positions = np.array([data["original_coordinates"] for data in tracking_data])
    smoothed_positions = np.array([data["smoothed_coordinates"] for data in tracking_data])
    face_points = np.concatenate([np.array(data["face_points"]) for data in tracking_data], axis=0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Seleccionar puntos faciales solo del último frame
    if len(tracking_data) > 0:
        last_face_points = np.array(tracking_data[-1]["face_points"])
    else:
        last_face_points = np.array([])

    # Graficar puntos faciales (solo del último frame)
    if last_face_points.size > 0:
        ax.scatter(
            last_face_points[:, 0],
            last_face_points[:, 1],
            last_face_points[:, 2],
            c="blue",
            s=1,
            label="Puntos faciales (último frame)"
        )

    # Trayectoria original
    ax.plot(original_positions[:, 0], original_positions[:, 1], original_positions[:, 2],
            color="red", marker="o", linestyle="dotted", label="Trayectoria Original")

    # Trayectoria suavizada
    ax.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], smoothed_positions[:, 2],
            color="blue", marker="^", linestyle="solid", label="Trayectoria Suavizada")

    ax.set_title("Trayectorias y Puntos Faciales", fontsize=16)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    plt.show()
