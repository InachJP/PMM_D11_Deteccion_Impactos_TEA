import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pykinect_azure as pykinect


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


def plot_3d_points_with_references(points, neck_position, x_margin, z_depth_limit):
    """
    Grafica los puntos 3D con referencias del cuello ajustado, límites en X y rango relativo en Z.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extraer coordenadas X, Y, Z
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, c='b', marker='o', s=1, label="Puntos filtrados")

    # Agregar punto de referencia para el cuello
    ax.scatter(neck_position[0], neck_position[1], neck_position[2], c='r', marker='o', s=50, label="Cuello")

    # Dibujar líneas para los límites en X y Z
    ax.plot([neck_position[0] - x_margin, neck_position[0] - x_margin], [neck_position[1], neck_position[1]],
            [neck_position[2] - z_depth_limit, neck_position[2]], 'g-', label="Límite Min X")
    ax.plot([neck_position[0] + x_margin, neck_position[0] + x_margin], [neck_position[1], neck_position[1]],
            [neck_position[2] - z_depth_limit, neck_position[2]], 'm-', label="Límite Max X")
    ax.plot([neck_position[0], neck_position[0]], [neck_position[1], neck_position[1]],
            [neck_position[2] - z_depth_limit, neck_position[2]], 'k--', label="Límite Z Relativo")

    # Configurar etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Puntos Filtrados con Referencias y Límite Z Relativo")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Inicializar librería
    pykinect.initialize_libraries(track_body=True)

    # Archivo de video para playback
    playback_filename = "test1_kalman.mkv"
    playback = pykinect.start_playback(playback_filename)
    playback_calibration = playback.get_calibration()

    # Iniciar body tracker con la calibración del playback
    body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)

    while True:
        # Leer frame del playback
        ret, capture = playback.update()
        if not ret:
            print("Fin del video.")
            break

        # Obtener frame del body tracker
        body_frame = body_tracker.update(capture)

        # Obtener datos del cuerpo
        if body_frame.get_num_bodies() > 0:
            body = body_frame.get_body(0)  # Tomamos el primer cuerpo detectado
            skeleton_3d = body.numpy()  # Obtener datos del esqueleto como numpy array

            # Coordenadas del cuello
            neck_position = skeleton_3d[pykinect.K4ABT_JOINT_NECK, :3]
            print(f"Posición del cuello: {neck_position}")

            # Obtener la nube de puntos segmentada del cuerpo
            ret_points, body_points = capture.get_pointcloud()
            if ret_points:
                body_points = body_points.reshape(-1, 3)

                # Filtrar puntos para el rostro
                filtered_points = filter_points_above_neck(body_points, neck_position, x_margin=100, z_depth_limit=200)

                # Graficar los puntos filtrados con referencias
                plot_3d_points_with_references(filtered_points, neck_position, x_margin=100, z_depth_limit=200)

    # Liberar recursos
    playback.close()
