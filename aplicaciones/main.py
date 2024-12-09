import plotly.graph_objects as go
import numpy as np
import pykinect_azure as pykinect
import time

def plot_frame_skeleton_with_head_and_feet(joint_positions, title, head_position, left_foot_position, right_foot_position, hand_position=None):
    """
    Graficar un solo frame del esqueleto con la cabeza y los pies resaltados.
    """
    if len(joint_positions) == 0:
        print("No se detectaron posiciones del esqueleto.")
        return

    # Extraer coordenadas del esqueleto
    x_coords = [joint[0] for joint in joint_positions]
    y_coords = [joint[1] for joint in joint_positions]
    z_coords = [joint[2] for joint in joint_positions]

    # Crear el gráfico 3D
    fig = go.Figure()

    # Añadir esqueleto completo como puntos
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        name='Esqueleto Completo',
        marker=dict(size=5, color='blue', opacity=0.8)
    ))

    # Añadir la cabeza
    fig.add_trace(go.Scatter3d(
        x=[head_position[0]], y=[head_position[1]], z=[head_position[2]],
        mode='markers',
        name='Cabeza',
        marker=dict(size=10, color='green', opacity=1)
    ))

    # Añadir los pies izquierdo y derecho
    fig.add_trace(go.Scatter3d(
        x=[left_foot_position[0]], y=[left_foot_position[1]], z=[left_foot_position[2]],
        mode='markers',
        name='Pie Izquierdo',
        marker=dict(size=10, color='purple', opacity=1)
    ))

    fig.add_trace(go.Scatter3d(
        x=[right_foot_position[0]], y=[right_foot_position[1]], z=[right_foot_position[2]],
        mode='markers',
        name='Pie Derecho',
        marker=dict(size=10, color='orange', opacity=1)
    ))

    # Si se proporciona, añadir la posición de la mano
    if hand_position is not None:
        fig.add_trace(go.Scatter3d(
            x=[hand_position[0]], y=[hand_position[1]], z=[hand_position[2]],
            mode='markers',
            name='Mano Derecha',
            marker=dict(size=8, color='red', opacity=1)
        ))

    # Configuración del gráfico
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)')
        ),
        width=1200,  # Ajuste del ancho
        height=900,  # Ajuste de la altura
        template='plotly_white'
    )

    fig.show()

def plot_hand_trajectory_with_euclidean_calculation(hand_positions):
    """
    Graficar la trayectoria de la mano derecha entre dos frames y mostrar el cálculo detallado de la distancia euclidiana como una anotación fija.
    """
    if len(hand_positions) < 2:
        print("No hay suficientes datos para graficar la trayectoria.")
        return

    # Extraer las coordenadas inicial y final de la mano
    start = np.array(hand_positions[0])
    end = np.array(hand_positions[-1])

    # Calcular la distancia euclidiana
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    delta_z = end[2] - start[2]

    distance_squared = delta_x**2 + delta_y**2 + delta_z**2
    delta_distance = np.sqrt(distance_squared)

    # Crear el gráfico 3D
    fig = go.Figure()

    # Añadir trayectoria de la mano derecha
    x_coords = [pos[0] for pos in hand_positions]
    y_coords = [pos[1] for pos in hand_positions]
    z_coords = [pos[2] for pos in hand_positions]

    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        name=f'Trayectoria Mano Derecha',
        line=dict(color='red', width=4),
        marker=dict(size=6, color='red')
    ))

    # Detalles del cálculo como anotación fija en el área visible
    annotation_text = (f"Δd = √((Δx)^2 + (Δy)^2 + (Δz)^2)<br>"
                       f"Δx = {delta_x:.2f} mm, Δy = {delta_y:.2f} mm, Δz = {delta_z:.2f} mm<br>"
                       f"Δd = √({delta_x**2:.2f} + {delta_y**2:.2f} + {delta_z**2:.2f})<br>"
                       f"Δd = √({distance_squared:.2f}) = {delta_distance:.2f} mm")

    # Configuración del gráfico con anotación
    fig.update_layout(
        title=dict(
            text="Trayectoria de la Mano Derecha y Cálculo de Δd",
            x=0.5
        ),
        annotations=[
            dict(
                x=0.05, y=0.95,  # Posición relativa (arriba a la izquierda del gráfico)
                xref='paper', yref='paper',
                text=annotation_text,
                showarrow=False,
                font=dict(size=12, color="black"),
                align="left",
                bgcolor="rgba(255,255,255,0.7)",  # Fondo semitransparente para mejor visibilidad
                bordercolor="black",
                borderwidth=1
            )
        ],
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)')
        ),
        width=1200,  # Ajuste del ancho
        height=900,  # Ajuste de la altura
        template='plotly_white'
    )

    fig.show()




if __name__ == "__main__":
    video_filename = "test5.mkv"

    # Inicializar PyKinect
    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(video_filename)
    playback_calibration = playback.get_calibration()
    bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)

    joint_positions_first_frame = []
    joint_positions_second_frame = []
    joint_positions_hand_right = []

    frame_count = 0

    while True:
        ret, capture = playback.update()
        if not ret:
            break

        body_frame = bodyTracker.update(capture=capture)

        for body_id in range(body_frame.get_num_bodies()):
            skeleton_3d = body_frame.get_body(body_id).numpy()

            # Capturar todas las posiciones del esqueleto
            if frame_count == 0:
                joint_positions_first_frame = skeleton_3d[:, :3]
                head_position_first = skeleton_3d[pykinect.K4ABT_JOINT_HEAD, :3]
                left_foot_position_first = skeleton_3d[pykinect.K4ABT_JOINT_FOOT_LEFT, :3]
                right_foot_position_first = skeleton_3d[pykinect.K4ABT_JOINT_FOOT_RIGHT, :3]
                hand_first_position = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]

            elif frame_count == 1:
                joint_positions_second_frame = skeleton_3d[:, :3]
                head_position_second = skeleton_3d[pykinect.K4ABT_JOINT_HEAD, :3]
                left_foot_position_second = skeleton_3d[pykinect.K4ABT_JOINT_FOOT_LEFT, :3]
                right_foot_position_second = skeleton_3d[pykinect.K4ABT_JOINT_FOOT_RIGHT, :3]
                hand_second_position = skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3]

            # Capturar las posiciones de la mano derecha
            if frame_count >= 0:
                joint_positions_hand_right.append(skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3])

        # Aumentar contador de frames
        frame_count += 1

        # Mostrar gráficos cuando se tengan suficientes datos
        if frame_count == 2:
            # Graficar el primer frame
            plot_frame_skeleton_with_head_and_feet(
                joint_positions_first_frame,
                "Primer Frame",
                head_position_first,
                left_foot_position_first,
                right_foot_position_first,
                hand_first_position
            )

            # Graficar el segundo frame
            plot_frame_skeleton_with_head_and_feet(
                joint_positions_second_frame,
                "Segundo Frame",
                head_position_second,
                left_foot_position_second,
                right_foot_position_second,
                hand_second_position
            )

            # Graficar la trayectoria de la mano derecha con cálculo de distancia
            plot_hand_trajectory_with_euclidean_calculation(joint_positions_hand_right)
            break
