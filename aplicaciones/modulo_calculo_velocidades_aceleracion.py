import numpy as np
import time
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a

# Funciones de cálculo
def calculate_linear_velocity(position1, position2, time1, time2):
    delta_position = np.array(position2) - np.array(position1)
    delta_time = time2 - time1
    if delta_time == 0:
        return 0
    return np.linalg.norm(delta_position) / delta_time  # Velocidad en m/s

def calculate_linear_acceleration(velocity1, velocity2, time1, time2):
    delta_velocity = velocity2 - velocity1
    delta_time = time2 - time1
    if delta_time == 0:
        return 0
    return delta_velocity / delta_time  # Aceleración en m/s²

def calculate_angular_velocity(quaternion1, quaternion2, time1, time2):
    delta_time = time2 - time1
    if delta_time == 0:
        return 0
    # Cuaternión de diferencia
    quaternion_diff = quaternion_multiply(quaternion_inverse(quaternion1), quaternion2)
    angle = 2 * np.arccos(np.clip(quaternion_diff[3], -1.0, 1.0))  # Ángulo asociado a la componente escalar w
    return angle / delta_time  # Velocidad angular en rad/s

def calculate_angular_acceleration(angular_velocity1, angular_velocity2, time1, time2):
    delta_angular_velocity = angular_velocity2 - angular_velocity1
    delta_time = time2 - time1
    if delta_time == 0:
        return 0
    return delta_angular_velocity / delta_time  # Aceleración angular en rad/s²

# Funciones auxiliares para cuaterniones
def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

def quaternion_inverse(q):
    if len(q) != 4:
        raise ValueError(f"Expected a quaternion of length 4, but got {len(q)}: {q}")
    x, y, z, w = q
    return [-x, -y, -z, w]

# Principal
if __name__ == "__main__":
    # Inicializar Kinect y configuración
    video_filename = "videos NFOV 2X2/mejilla_derecha_NFOV_2X2_luz_artificial.mkv"
    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_filename)
    playback_calibration = playback.get_calibration()
    body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)

    # Variables para cálculos
    timestamps = []
    hand_positions = []
    hand_orientations = []
    velocities = []
    accelerations = []

    while True:
        ret, capture = playback.update()
        if not ret:
            break

        body_frame = body_tracker.update(capture=capture)
        current_time = time.time()

        # Obtener datos del esqueleto
        for body_id in range(body_frame.get_num_bodies()):
            skeleton_3d = body_frame.get_body(body_id).numpy()
            hand_position = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]

            # Convertir las posiciones de mm a metros
            hand_position = hand_position / 1000  # Convertir de mm a metros

            hand_orientation = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, 3:]

            # Limitar la orientación a 4 valores
            if len(hand_orientation) > 4:
                hand_orientation = hand_orientation[:4]  # Tomar solo los primeros 4 componentes

            # Guardar datos
            timestamps.append(current_time)
            hand_positions.append(hand_position)
            hand_orientations.append(hand_orientation)

            # Calcular velocidad y aceleración lineal
            if len(hand_positions) > 1:
                velocity = calculate_linear_velocity(hand_positions[-2], hand_positions[-1], timestamps[-2], timestamps[-1])
                velocities.append(velocity)

                if len(velocities) > 1:
                    acceleration = calculate_linear_acceleration(velocities[-2], velocities[-1], timestamps[-2], timestamps[-1])
                    accelerations.append(acceleration)
                else:
                    acceleration = 0

                # Calcular velocidad y aceleración angular
                angular_velocity = calculate_angular_velocity(hand_orientations[-2], hand_orientations[-1], timestamps[-2], timestamps[-1])
                if len(accelerations) > 1:
                    angular_acceleration = calculate_angular_acceleration(accelerations[-2], accelerations[-1], timestamps[-2], timestamps[-1])
                else:
                    angular_acceleration = 0

                # Imprimir resultados
                print(f"Tiempo: {timestamps[-1]:.2f}s")
                print(f"Velocidad lineal: {velocity:.3f} m/s")
                print(f"Aceleración lineal: {acceleration:.3f} m/s²")
                print(f"Velocidad angular: {angular_velocity:.3f} rad/s")
                print(f"Aceleración angular: {angular_acceleration:.3f} rad/s²")

    print("Cálculos completados.")
