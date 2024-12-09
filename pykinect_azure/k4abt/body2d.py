import numpy as np
import cv2

from pykinect_azure.k4abt.joint2d import Joint2d
from pykinect_azure.k4abt._k4abtTypes import K4ABT_JOINT_COUNT, K4ABT_SEGMENT_PAIRS
from pykinect_azure.k4abt._k4abtTypes import k4abt_skeleton2D_t, k4abt_body2D_t, body_colors
from pykinect_azure.k4a._k4atypes import K4A_CALIBRATION_TYPE_DEPTH

class Body2d:
	def __init__(self, body2d_handle):

		if body2d_handle:
			self._handle = body2d_handle
			self.id = body2d_handle.id
			self.initialize_skeleton()

	def __del__(self):

		self.destroy()

	def json(self):
		return self._handle.__iter__()

	def numpy(self):
		return np.array([joint.numpy() for joint in self.joints])

	def is_valid(self):
		return self._handle

	def handle(self):
		return self._handle

	def destroy(self):
		if self.is_valid():
			self._handle = None

	def initialize_skeleton(self):
		joints = np.ndarray((K4ABT_JOINT_COUNT,),dtype=np.object_)

		for i in range(K4ABT_JOINT_COUNT):
			joints[i] = Joint2d(self._handle.skeleton.joints2D[i], i)

		self.joints = joints

	def draw(self, image, only_segments=False, movement_detected=False, impact_detected=False, joint_head_name=None, impact_position=None, face_region=None):
		"""
		Dibuja el cuerpo, el movimiento detectado, el impacto y la región del rostro.
		
		:param image: La imagen sobre la cual se dibujará.
		:param only_segments: Si es True, solo se dibujan los segmentos.
		:param movement_detected: Si es True, destaca las articulaciones de la muñeca derecha.
		:param impact_detected: Si es True, dibuja el impacto en rojo.
		:param joint_head_name: Nombre de la articulación en la cabeza impactada.
		:param impact_position: Coordenada 2D del impacto a marcar en rojo.
		:param face_region: Lista de coordenadas 2D para la región del rostro a dibujar.
		:return: La imagen con los dibujos aplicados.
		"""

		# Dibujar región del rostro si se proporciona

		# Dibujar segmentos del cuerpo si no está en modo solo segmentos
		if not only_segments:
			for joint in self.joints:
				joint_position = joint.get_coordinates()
				joint_name = joint.get_name()

				# Dibujar puntos de la mano derecha si se detectó movimiento
				if movement_detected and joint_name in ["right wrist", "right hand", "right thumb", "right handtip"]:
					image = cv2.circle(image, joint_position, 3, (0, 0, 255), -1)

				# Dibujar el impacto si se detectó
			if impact_detected:
				print(joint_position)
				#print(impact_position)
				image = cv2.circle(image, impact_position, 5, (255, 0, 0), -1)

	

		return image



	@staticmethod
	def create(body_handle, calibration, bodyIdx, dest_camera):

		skeleton2d_handle = k4abt_skeleton2D_t()
		body2d_handle = k4abt_body2D_t()

		for jointID,joint in enumerate(body_handle.skeleton.joints): 
			skeleton2d_handle.joints2D[jointID].position = calibration.convert_3d_to_2d(joint.position, K4A_CALIBRATION_TYPE_DEPTH, dest_camera)
			skeleton2d_handle.joints2D[jointID].confidence_level = joint.confidence_level

		body2d_handle.skeleton = skeleton2d_handle
		body2d_handle.id = bodyIdx

		return Body2d(body2d_handle)


	def __str__(self):
		"""Print the current settings and a short explanation"""
		message = f"Body Id: {self.id}\n\n"

		for joint in self.joints:
			message += str(joint)

		return message

