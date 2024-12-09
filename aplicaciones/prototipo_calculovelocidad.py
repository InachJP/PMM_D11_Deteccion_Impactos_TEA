import cv2
import numpy as np
import time
import json

import pykinect_azure as pykinect

DEFAULT_THRESHOLD_M = 15
DEFAULT_THRESHOLD_I = 200
SPEED_THRESHOLD = 0.15

# Contador para los impactos
impact_counter = 1

# Lista para almacenar la información de tiempo y posición
time_intervals = []
joint_positions_hand_right = []
impacts_list = []

def save_impacts_to_json(impacts, filename='impact_data.json'):
    with open(filename, 'w') as f:
        json.dump(impacts, f, indent=4)

def calibrate_threshold(joint_positions):
    """
    Calibrate the threshold for detecting movement based on the standard deviation of coordinates.
    """
    std_devs = np.std(joint_positions, axis=0)
    threshold = 2 * np.mean(std_devs)
    return threshold


def calculate_velocity(joint_positions_hand_right, time_intervals):
    if len(joint_positions_hand_right) < 2:
        return 0

    # Obtener las últimas dos posiciones y tiempos
    pos1 = np.array(joint_positions_hand_right[0][0])
    pos2 = np.array(joint_positions_hand_right[-1][0])
    time1, time2 = time_intervals[0], time_intervals[-1]
    #print(f'Time1: {time1}, Time2: {time2}')
    # Calcular el tiempo transcurrido
    delta_time = time2 - time1
    if delta_time == 0:
        return 0

    # Calcular la distancia entre las dos posiciones
    distance = np.linalg.norm(pos2 - pos1)
    # Calcular la velocidad (distancia / tiempo)
    velocity = distance / delta_time
    #convert to mm to meters
    velocity = velocity / 1000
    #print(f'Velocity: {velocity:.2f} m/s')
    return velocity, distance / 1000

def detect_movement(joint_positions_hand_right, threshold=DEFAULT_THRESHOLD_M):
    """
    Detect movement based on the distance between the last two positions of the right hand.
    """
    if len(joint_positions_hand_right) < 2:
        return False

    # Obtener las últimas dos posiciones de la mano derecha
    pos1 = np.array(joint_positions_hand_right[-2][0])
    pos2 = np.array(joint_positions_hand_right[-1][0])

    # Calcular la distancia entre las dos posiciones
    distance = np.linalg.norm(pos2 - pos1)
    #qprint(f'Distance: {distance:.2f} mm')
    if distance > threshold:
        print("Movement detected.")
        return True

    return False

def detect_head_impact(joint_positions_hand_right, joint_positions_head, threshold=DEFAULT_THRESHOLD_I):
    joint_head_names = ["head", "nose", "left eye", "left ear", "right eye", "right ear"]
    joint_hand_names = ["wrist", "hand", "hand tip", "thumb"]

    for hand_index, joint_hand_position in enumerate(joint_positions_hand_right[-1]):  # Últimas posiciones de la mano
        for head_index, joint_head_position in enumerate(joint_positions_head):  # Todas las posiciones de la cabeza
            distance = np.linalg.norm(joint_hand_position - joint_head_position)
            if distance < threshold:
                print(f'Impact detected between {joint_hand_names[hand_index]} and {joint_head_names[head_index]}: {distance}')
                return True, joint_hand_names[hand_index], joint_head_names[head_index], hand_index, head_index
    
    return False, "No impact", "No impact", "No Impact", "No Impact"





if __name__ == "__main__":

    video_filename = "output.mkv"

	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)

	# Start playback
    playback = pykinect.start_playback(video_filename)

    playback_config = playback.get_record_configuration()

    print(playback_config)

    playback_calibration = playback.get_calibration()

	# Start body tracker
    bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)

    cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)

    """ device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

    device = pykinect.start_device(config=device_config)
    bodyTracker = pykinect.start_body_tracker(model_type=pykinect.K4ABT_DEFAULT_MODEL)
    """

    joint_positions_hand_right = []
    joint_positions_head = []

    movement_detected = False
    impact_detected = False
    tracking_movement = False
    movement_start_time = None
    initial_time_set = False
    start_time = time.time()
    impact_info = None
    initial_position = None


    # Bucle principal
    while True:
            ret, capture = playback.update()

            if not ret:
               break

            body_frame = bodyTracker.update(capture=capture)

            # Get color image
            ret_color, color_image = capture.get_transformed_color_image()

            # Get the colored depth
            ret_depth, depth_color_image = capture.get_colored_depth_image()

            # Get the colored body segmentation
            ret_seg, body_image_color = body_frame.get_segmentation_image()
            
            if not ret_color or not ret_depth or not ret_seg:
               continue

            # Combine both images
            combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
            combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

            for body_id in range(body_frame.get_num_bodies()):
                skeleton_3d = body_frame.get_body(body_id).numpy()

                # Obtener las posiciones de los joints
                hand_joints = [
                    skeleton_3d[pykinect.K4ABT_JOINT_WRIST_RIGHT, :3], 
                    skeleton_3d[pykinect.K4ABT_JOINT_HAND_RIGHT, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_HANDTIP_RIGHT, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_THUMB_RIGHT, :3]
                ]

                joint_positions_hand_right.append(hand_joints)

                
                joint_positions_head = np.array([
                    skeleton_3d[pykinect.K4ABT_JOINT_HEAD, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_NOSE, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_EYE_LEFT, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_EYE_RIGHT, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_EAR_LEFT, :3],
                    skeleton_3d[pykinect.K4ABT_JOINT_EAR_RIGHT, :3]
                ])

                time_intervals.append(time.time())

                elapsed_time = time.time() - start_time

                if not initial_time_set:
                    if elapsed_time >= 3:
                        initial_time_set = True
                        print("Initial 3 seconds elapsed, starting motion detection...")
                    continue

                if tracking_movement:
                    impact_detected, joint_hand_name, joint_head_name, hand_index, head_index = detect_head_impact(joint_positions_hand_right, joint_positions_head)

                    if impact_detected:
                        elapsed_time_trajectory = time.time() - movement_start_time
                        print(f'Time elapsed between movement and impact: {elapsed_time_trajectory:.2f} seconds')

                        # Calcular la velocidad
                        velocity, distance = calculate_velocity([pos[0] for pos in joint_positions_hand_right], time_intervals)

                        # Guardar información del impacto
                        impact_info = {
                            f"impact_{impact_counter}": {    
                                "velocity ([m/s])": velocity,
                                "delta_distance ([m])": distance,
                                "delta_time [s]": elapsed_time_trajectory,
                                "trajectory": [
                                    {
                                        "x": pos[hand_index][0],
                                        "y": pos[hand_index][1],
                                        "z": pos[hand_index][2],
                                        "t": time_interval
                                    }
                                    for pos, time_interval in zip(joint_positions_hand_right, time_intervals)
                                ],
                                "contact_part": {
                                    "hand": joint_hand_name,
                                    "head": joint_head_name
                                }
                            }
                        }

                        # Añadir el impacto a la lista de impactos
                        impacts_list.append(impact_info)

                        joint_positions_hand_right = []
                        tracking_movement = False
                        movement_start_time = None
                        print("Resetting positions and times after impact.")

                        # Incrementar el contador de impactos
                        impact_counter += 1
                    #volver a preguntar si hay movimiento
                    else:
                        movement_detected = detect_movement(joint_positions_hand_right)
                        if not movement_detected:
                            tracking_movement = False
                            joint_positions_hand_right = []
                            movement_start_time = None
                            time_intervals = []
                            initial_position = None
                            print("Resetting positions and times after no movement.")

                else:
                    if len(joint_positions_hand_right) > 1:
                        movement_detected = detect_movement(joint_positions_hand_right)

                        if movement_detected:
                            tracking_movement = True
                            movement_start_time = time.time()
                            initial_position = joint_positions_hand_right[-1]  # Guardar la posición inicial
                            time_intervals = [time.time()]  # Inicializar la lista de intervalos de tiempo
                            print("Movement detected, tracking time...")

            combined_image = body_frame.draw_bodies(combined_image, pykinect.K4A_CALIBRATION_TYPE_DEPTH, False, movement_detected, impact_detected, None)
            cv2.imshow('Depth image with skeleton', combined_image)

            if cv2.waitKey(1) == ord('q'):
                # Guardar todos los impactos en JSON al final del programa
                save_impacts_to_json(impacts_list)
                break