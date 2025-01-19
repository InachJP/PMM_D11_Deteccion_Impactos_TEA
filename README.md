# **Programa de Memoria Multidisciplinaria USM Desafío N11 Deteccion Impactos TEA**

> Este proyecto tiene como objetivo analizar y evaluar técnicas de detección de impactos y movimientos a partir de videos utilizando Kinect. A través de la captura de datos de la posición 3D de la mano y los puntos faciales, se calcula la velocidad y se detectan los impactos en función de la proximidad entre la mano y la cara. Se emplean distintas configuraciones de iluminación (natural y artificial) y se comparan los resultados obtenidos al utilizar o no un filtro de Kalman extendido (EKF), con el fin de mejorar la precisión y fiabilidad de las mediciones. 

---

## **Tabla de Contenidos**

1. [Introducción](#introducción)
2. [Características](#características)
3. [Instalación](#instalación)
4. [Uso](#uso)

---

## **Introducción**

### Descripción del Proyecto

> El objetivo final de este proyecto es desarrollar un software o aplicación en un contexto médico donde especificamente neurologos puedan en sus citas con pacientes susceptibles al trastorno espectro autista medir el impacto de golpes involuntarios e inclusive poder encontrar una relación de los datos que pueda contribuir respecto a que nivel del espectro diagnosticar al paciente. Por lo tanto se desarrollara una interfaz amigable donde el especialista pueda obtener un reporte como por ejemplo de la fuerza ejercida por cada una de los golpes.
---

## **Características**

- Análisis de videos con diferentes técnicas de golpes.
- Evaluación con iluminación natural y artificial.
- Implementación del Extended Kalman Filter para mejorar la precisión de las mediciones.
- Algoritmo para detectar impactos en función de la proximidad de la mano con puntos faciales.
- Cálculo de la velocidad de los movimientos en metros por segundo.

---

## **Instalación**

### Requisitos

#### **Azure Kinect SDK (K4A).**

Azure Kinect SDK is a cross platform (Linux and Windows) user mode SDK to read data from your Azure Kinect device.

  - To use the SDK, please refer to the installation instructions in [usage](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md)

  - API documentation is avaliable [here](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/)

#### **Azure Kinect DK Body Tracking SDK.**

  - Download [here](https://www.microsoft.com/en-us/download/details.aspx?id=104221)


1. **Python 3.9**  
2. **Bibliotecas**:
    - `pykinect_azure`
    - `numpy`
    - `opencv-python`
    - `matplotlib`

Es necesario un entorno virtual con una versión de Python en 3.9 dado que la biblioteca de la API del soporte del SDK en Python de la Azure Kinect DK exista hasta esta versión.

Puedes instalar las dependencias necesarias utilizando `pip`. Aquí un ejemplo:

```bash
pip install -r requirements.txt
```

## **Instrucciones de instalación**
1.Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/InachJP/PMM_D11_Deteccion_Impactos_TEA.git
```
2. Navega al directorio del proyecto:
   
```bash
cd PMM_D11_Deteccion_Impactos_TEA
```
3. Instala las dependencias:
```bash
pip install -r requirements.txt
```
