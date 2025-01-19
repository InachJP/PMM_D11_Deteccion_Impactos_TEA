# **Programa de Memoria Multidisciplinaria USM Desafío N11 Deteccion Impactos TEA**

> Este proyecto tiene como objetivo analizar y evaluar técnicas de detección de impactos y movimientos a partir de videos utilizando Kinect. Se captura la posición 3D de la mano y los puntos faciales para calcular la velocidad de los movimientos y detectar los impactos en función de la proximidad entre la mano y la cara. Se emplean diferentes configuraciones de iluminación, tanto natural como artificial, y se comparan los resultados obtenidos al utilizar o no un filtro de Kalman extendido (EKF), con el propósito de mejorar la precisión y fiabilidad de las mediciones.

---

## **Tabla de Contenidos**

1. [Introducción](#introducción)
2. [Características](#características)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Estructura del Proyecto](#estructura_del_proyecto)
6. [Cómo Funciona](#cómo_Funciona)
7. [Resultados Mediciones](#resultados_mediciones)
---

## **Introducción**

### Descripción del Proyecto

>El objetivo final de este proyecto es desarrollar un software o aplicación en un contexto médico, específicamente para que los neurólogos puedan, durante sus citas con pacientes susceptibles al trastorno del espectro autista, medir el impacto de los golpes involuntarios. Además, el sistema permitirá encontrar una relación entre los datos recopilados que pueda contribuir en el diagnóstico sobre el nivel del espectro en el que se encuentra el paciente. Para ello, se desarrollará una interfaz amigable que permitirá al especialista obtener reportes detallados, como la fuerza ejercida en cada uno de los golpes.
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

Para instalar las dependencias necesarias utilizando `pip`. Aquí un ejemplo:

```bash
pip install -r requirements.txt
```

## **Instrucciones de instalación**

1. Clona este repositorio en tu máquina local:

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

## **Uso**

```bash
python prototipo_calculovelocidad.py
```

## **Estructura del Proyecto**

```bash
nombre-del-repositorio/
│── aplicaciones/                                               #Directorio con todos los modulos y programas principales (grabar, playback, utils, prototipov3, etc)
├── capturas_impactos/                                         # Directorio donde se guardan las imágenes de los impactos
├── videos/                                                     # Videos de entrada para el análisis
│
├── utils1.py                                                  # Funciones auxiliares para el procesamiento
├── prototipo_calculavelocidad_v{x}.py                       # Script principal de ejecución
├── requirements.txt                                           # Dependencias del proyecto
├── impact_data.json                                           # Archivo de resultados generados
└── README.md                       
```
## **Resultados Mediciones**
[Enlace resultados](https://usmcl-my.sharepoint.com/:f:/g/personal/jonathan_pedraza_usm_cl/Eh41hIklhihFjHqmEK4RTAcBaJb0hIz6LDjaoa4LvAC2zQ?e=3Pc2Vq)
