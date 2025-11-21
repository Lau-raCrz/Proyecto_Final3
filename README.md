# Proyecto tercer corte

## Primer Punto 
### Creacion del dataset


Para la construcción del dataset se recopilaron imágenes de 10 dispositivos electrónicos utilizando técnicas de web scraping, descarga automática desde DuckDuckGo e imágenes capturadas por el usuario.
Cada categoría contiene entre 200 y 400 imágenes.

Las herramientas incluidas fueron:

Multímetro electrónico

Osciloscopio

Cautín para soldar

Protoboard

Capacitor electrónico

Resistencia electrónica

LED electrónico

Destornillador electrónico

Fuente de alimentación DC

Transformador electrónico

Raspberry Pi

Cada herramienta tiene su propia carpeta en el proyecto, y dentro de cada una se guardan todas las imágenes descargadas.

Imagen 1

### Web Scraping con una API de DuckDuckGo

Inicialmente se intentó hacer el scraping con Selenium y Bing Images, pero estas dos herramientas presentaban problemas como que aparecian bloqueos por parte del navegador, tambien habia alta cantidad de imágenes no relacionadas, habia límites en modo headless, baja precisión en URLs de imágenes reales.

Por esos motivos, se decidió utilizar la librería:
```
ddgs
```
Ya que es mas rapido, mas estable, sin ventanas ni problemas de headless y es ideal para el scraping masivo

La función utilizada para obtener URLs fue:

```

from ddgs import DDGS

def obtener_urls(elemento, cantidad=200):
    urls = []
    with DDGS() as ddgs:
        resultados = ddgs.images(query=elemento, max_results=cantidad)
        for r in resultados:
            url = r["image"]
            if url.startswith("http"):
                urls.append(url)
    return urls
```

Esta función recibe un término de búsqueda (por ejemplo, "Multimetro electronico") y devuelve una lista de URLs de imágenes reales.

### Descarga de imágenes utilizando hilos, mutex y semáforo

Para cumplir con los requisitos la descarga del dataset se implementó usando:

✔ Hilos para descargar varias imágenes al mismo tiempo y acelerar el proceso.

✔ Se utilizo semáforo para limitar cuántos hilos pueden descargar simultáneamente.
```
semaforo = threading.Semaphore(10)
```
Esto significa que hay un máximo 10 descargas simultáneas.

✔ Se utilizo  Mutex para proteger la sección crítica, que en este caso es el momento de guardar el archivo en el sistema, evitando que dos hilos escriban el mismo archivo al mismo tiempo.

```
mutex = threading.Lock()
```

✔ Sección Crítica esta parte del codigo donde SOLO un hilo puede entrar:

```
with mutex:
    nombre = f"img_{int(time.time()*1000)}.jpg"
    ruta = os.path.join(carpeta, nombre)
    img.save(ruta)
```

✔ Descarga paralela de imágenes, esta fue la función encargada de cada descarga:
 ```
def descargar_imagen(url, carpeta):
    semaforo.acquire()     # Limitar descargas simultáneas
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))

        # ---- SECCIÓN CRÍTICA ----
        with mutex:
            nombre = f"img_{int(time.time()*1000)}.jpg"
            ruta = os.path.join(carpeta, nombre)
            img.save(ruta)

    except:
        pass

    semaforo.release()

 ```

 Finalmente, se creó un archivo llamado dataset_builder.py que:

Toma cada nombre de carpeta, lo convierte en un término de búsqueda, luego llama a descargar_dataset() y por ultimo descarga todas las imágenes necesarias automáticamente.


```

descargar_dataset(
    elemento="Raspberry Pi",
    carpeta_destino="Raspberry_Pi",
    cantidad=120
)
```
Y para correr el archivo se utilizo
```
python3 dataset_builder.py
```

Posteriormente se aplicó limpieza, filtrado, eliminación de duplicados y estandarización de formato.
Finalmente se generó el dataset en formato YOLO con carpetas /images/train, /images/val, /labels/train y /labels/val.
Este dataset se utiliza tanto para el entrenamiento del modelo de clasificación como para el entrenamiento del detector YOLO que permitirá identificar los objetos en tiempo real.

### Entrenamiento del Modelo de Clasificación

El proceso de entrenamiento del modelo YOLOv8 fue uno de los pasos más importantes del proyecto, ya que de este modelo depende la precisión con la que el sistema detecta los 10 elementos electrónicos definidos. Para lograrlo, se siguieron una serie de etapas técnicas que permitieron preparar el dataset, configurarlo según los estándares de YOLO y finalmente entrenar el modelo de detección.


El primer paso fue organizar todas las imágenes recolectadas mediante web scraping, los datos estén distribuidos en carpetas separadas para entrenamiento y validación, tanto para las imágenes como para las etiquetas. como ya se explico anteriormente.

Para que YOLO conozca la ubicación del dataset y los nombres de las clases, se creó un archivo llamado data.yaml. Este archivo contiene las rutas de las carpetas de entrenamiento/validación, el número total de clases (nc) y la lista de nombres de cada clase.

Con la estructura del dataset lista y el archivo de configuración preparado, se procedió a realizar el entrenamiento con el siguiente comando:

```
yolo detect train model=yolov8n.pt data=dataset_yolo/data.yaml epochs=20 imgsz=640

```
Con esta linea de codigo se entrenará un modelo de detección (detect train),se usará el modelo base yolov8n.pt, los datos provienen de data.yaml, se entrenara el modelo a 20 épocas y se utilizará un tamaño de imagen de 640x640.

Durante el entrenamiento YOLO realiza validación automática en cada época, aplica técnicas de data augmentation y ajusta dinámicamente los parámetros de aprendizaje. Esto permite mejorar la capacidad del modelo para generalizar y reconocer los objetos en condiciones diferentes a las del dataset.


### Uso del Modelo Entrenado

Para usar el modelo entrenado dentro de un programa en Python, Este código permite evaluar el modelo con cualquier imagen o integrarlo en una aplicación de detección en tiempo real, como se hizo más adelante con OpenCV.


# Segundo Punto Proceso ETL del Dataset de Herramientas Electrónicas


## Extracción del Dataset (Web Scraping)

Inicialmente, el objetivo era realizar la extracción de imágenes utilizando Selenium y motores como Bing Images, automatizando la búsqueda y descarga de imágenes de forma directa desde el navegador.Sin embargo, durante las pruebas se presentaron múltiples problemas:


- Bloqueos constantes del navegador por protección anti-scraping.

- Bing devolvía una alta cantidad de imágenes irrelevantes.

- El modo headless generaba errores y cargaba mal las páginas.

- La mayoría de URLs encontradas eran de miniaturas o imágenes de baja calidad.

- Consumo excesivo de RAM y CPU.

- Muy lento para scraping masivo.

Debido a estos inconvenientes, Selenium no era viable para recolectar cientos de imágenes por categoría.


Para solucionar los problemas anteriores, se decidió usar la librería:


ddgs (DuckDuckGo Search)

Esta librería permite obtener URLs de imágenes sin abrir un navegador, lo cual aporta mayor velocidad,menos bloqueos, mayor estabilidad, menor consumo de recursos, es ideal para scraping masivo (2000+ imágenes) y sin problemas de headless

Además, sus resultados son más limpios y fáciles de procesar automáticamente.



Uso de hilos (threads) en la descarga

Para acelerar la descarga, implementé un sistema de descarga concurrente, donde varios hilos trabajan simultáneamente.

Un mutex protege la sección crítica para evitar que dos hilos escriban la misma imagen al mismo tiempo:

```
lock = threading.Lock()

def download_image(url, save_path):
    try:
        img_data = requests.get(url, timeout=5).content
        with lock:  # Sección crítica
            with open(save_path, "wb") as f:
                f.write(img_data)
    except:
        pass

```
Esto permitió descargar miles de imágenes de forma rápida y segura.

## Transformacion 

Una vez obtenidas todas las imágenes mediante web scraping, se inició la fase de Transformación, cuyo objetivo fue limpiar, normalizar y preparar los datos antes del entrenamiento del modelo YOLO. Esta etapa fue fundamental para garantizar la calidad del dataset, ya que el scraping genera imágenes repetidas, corruptas, de baja resolución o simplemente no relacionadas con el objeto buscado.

### Limpieza de las imágenes (Validación del dataset)

El primer paso fue verificar una por una las imágenes descargadas. Para ello se creó un script encargado de comprobar si la imagen realmente puede abrirse y si cumple con una mínima calidad visual. Muchas imágenes del scraping pueden estar corruptas o tener tamaños extremadamente pequeños, por lo cual se descartaron automáticamente.

```

import cv2
import os

def is_valid_image(path):
    img = cv2.imread(path)
    if img is None:
        return False
    h, w = img.shape[:2]
    return h >= 100 and w >= 100  # mínimo aceptable

```
Este proceso permitió filtrar únicamente imágenes útiles, dejando el dataset final en un estado óptimo para el entrenamiento.

Luego de la limpieza, todas las imágenes fueron redimensionadas a 640x640 píxeles, que es un tamaño recomendado por YOLOv8 para garantizar un aprendizaje estable y sin errores de resolución. Esto ayuda a que el modelo procese todas las imágenes bajo una misma escala, evitando inconsistencias durante el entrenamiento.
El siguiente paso consistió en reorganizar toda la información en la estructura requerida por YOLO
Además, se creó el archivo data.yaml, donde se definen las rutas del dataset y los nombres de las clases.

Finalmente, el dataset fue dividido por 80% para entrenamiento y 20% para validaciónsiguiendo las buenas prácticas de aprendizaje automático.


## Carga del dataset y entrenamiento del modelo

La fase final del ETL consiste en cargar el dataset dentro del modelo YOLO y entrenarlo. En este proyecto se utilizó el modelo base yolov8n.pt, ya que es liviano, rápido y adecuado para proyectos académicos.

El entrenamiento se realizó con el siguiente comando ya anteriormente explicado.
```
yolo detect train model=yolov8n.pt data=dataset_yolo/data.yaml epochs=20 imgsz=320

```

El entrenamiento generó automáticamente la carpeta:
```
runs/detect/train4/
```

Dentro de ella se encuentran los modelos finales, el archivo best.pt es el modelo entrenado definitivo que luego se utilizó en los scripts de detección en tiempo real.

La carga del modelo en los scripts de detección se realizó usando:
```
from ultralytics import YOLO
model = YOLO("best.pt")
results = model(frame)

```

Esto permitió detectar herramientas electrónicas directamente desde la cámara, completando así la fase de Carga del proceso ETL.

# tercer punto

## Detección de personas y cálculo de velocidad – Explicación resumida

Para la parte del proyecto enfocada en identificar personas y calcular su velocidad en tiempo real, desarrollamos un sistema que divide el procesamiento entre varios threads para reducir la latencia y evitar bloqueos en la interfaz. Inicialmente se intentó usar MediaPipe, pero presentaba limitaciones importantes: solo detectaba una persona a la vez, el rendimiento bajaba con movimiento rápido y la precisión se afectaba con cambios de iluminación. Por esto se decidió utilizar YOLO-Pose, que permite detectar varias personas simultáneamente y proporciona los keypoints necesarios para calcular posiciones del cuerpo con mayor estabilidad.

El sistema está dividido en tres hilos principales, además del hilo principal de captura y renderizado:

1) Hilo YOLO – detección de herramientas

Este hilo procesa únicamente la parte izquierda de la pantalla: la clasificación de herramientas del laboratorio que aparecen en el frame. El hilo corre en paralelo usando:

- Locks (mutex) para proteger el acceso al último resultado (last_yolo_results)

- Buffers sincronizados para leer siempre el frame más actualizado sin bloquear

Esto permite que YOLO trabaje constantemente sin detener el flujo de la cámara ni interferir con los otros hilos.

2) Hilo YOLO-Pose – detección de personas

En este hilo se realiza la detección de personas y extracción de sus keypoints. Cada frame procesado genera la lista de personas detectadas, keypoints de cada una de las personas y Frame asociado y tiempo entre frames

Este hilo también usa un mutex exclusivo (results_lock) para actualizar el resultado compartido sin generar condiciones de carrera. Gracias a esto, ambos modelos (YOLO y YOLO-Pose) pueden correr simultáneamente sin afectar el rendimiento.

3) Hilo principal – captura, sincronización y renderizado

El hilo principal se encarga de capturar el video de la cámara, medir el tiempo entre frames para calcular velocidades, empujar cada frame a un buffer sincronizado tipo deque, tomar los resultados producidos por los otros hilos, dibujar bounding boxes, esqueletos, paneles y textos, combinar ambos lados en una sola interfaz y mantener la ventana activa sin congelarse.

La interfaz final está dividida en dos áreas:

Izquierda: clasificación de herramientas utilizando YOLO
Derecha: detección de personas + cálculo de velocidades usando YOLO-Pose

### Cálculo de velocidad y seguimiento de múltiples personas

Para poder calcular la velocidad de cada persona, primero utilizamos los keypoints de YOLO-Pose y calculamos un “centro del cuerpo” utilizando hombros y caderas. Este centro es más estable que usar la cabeza o las manos, que cambian mucho de posición.

Luego entra en acción el PersonTracker, que ahora sí es totalmente seguro en entornos paralelos porque usa un mutex interno. El tracker asigna IDs a cada nueva persona detectada, calcula la distancia recorrida entre frames, estima velocidad real en px/s usando Δdistancia / Δtiempo, mantiene la identidad de cada persona aunque se cruce con otra, elimina IDs antiguos usando limpieza protegida

Ejemplo del uso del lock dentro del tracker:
```
with self.lock:
    # Acceso seguro al diccionario de personas

```
Esto protege la estructura compartida de accesos simultáneos desde el hilo de pose y el hilo principal.



Por ultimo el sistema utiliza varios mecanismos para garantizar estabilidad:

✔ Lock() — Mutua exclusión, evitan que dos hilos modifiquen la misma variable al mismo tiempo.

✔ Event() — señal global de apagado, permite detener todos los hilos de manera segura cuando el usuario presiona q.

✔ deque(maxlen=3) con lock, sirve como un buffer circular de frames que suaviza la captura y evita que la detección se quede congelada cuando un hilo se demora más que otro.
Tener un buffer pequeño (3 frames) evita la acumulación de retraso (“lag”).

# Cuarto punto 

## Containerización con Docker

El Dockerfile es un archivo de texto con instrucciones para construir nuestra imagen Docker. Cada línea crea una "capa" que Docker puede guardar en caché para acelerar construcciones futuras. El diseño siguió un proceso de optimización para mantener la imagen pequeña y rápida de construir.
Iniciamos con una imagen base de Python ligera. Existen varias opciones: Elegimos python:3.10-slim porque incluye lo esencial sin componentes innecesarios:

```
FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
```


Luego instalamos las dependencias del sistema operativo que OpenCV necesita para funcionar. OpenCV requiere bibliotecas específicas para procesar imágenes y acceder a la cámara web. Las instalamos todas en un solo comando para minimizar el tamaño de la imagen:

```
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk2.0-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*
```

Finalmente, configuramos cómo iniciará la aplicación cuando el contenedor arranque:

```
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```
Luego de ya haber creado el archivo dockerfile lo que se hace es contruir la imagen y ejecutar el contenedor de la siguiente forma:

```
docker compose build

./run.sh
```
### Docker Compose: Simplificando la Ejecución
Docker Compose que permite definir toda la configuración del contenedor en un archivo YAML en lugar de escribir comandos largos y complicados. Nuestro docker-compose.yml especifica tres cosas importantes:
- Conecta el puerto 8501 del contenedor (donde Streamlit escucha) con el puerto 8501 de tu computadora, permitiendo acceder a la aplicación desde el navegador.
- Mapea directorios del host dentro del contenedor. El mapeo más importante es /tmp/.X11-unix, que permite que el contenedor acceda a la cámara web. También mapeamos el código fuente para poder modificarlo sin reconstruir la imagen.
- Tambien le da al contenedor acceso directo a la cámara web (/dev/video0).

## Despliegue con Streamlit

### Estructura de la Aplicación
La aplicación se organizó en dos modos principales que el usuario puede elegir desde un menú lateral:
Modo Imagen: El usuario sube una foto desde su computadora y la aplicación ejecuta los dos modelos YOLO (herramientas y personas) sobre esa imagen. Los resultados se muestran lado a lado en dos columnas. Este modo es útil para analizar imágenes específicas sin necesidad de video en vivo.
Modo Cámara en Vivo: La aplicación captura video continuamente desde la webcam, procesa cada frame en tiempo real, detecta herramientas y personas, y calcula la velocidad de movimiento de las personas. Los resultados se actualizan en la pantalla constantemente, creando una experiencia de detección en tiempo real.
Cómo Funciona la Cámara en Vivo
El modo de cámara en vivo es el más complejo porque debe capturar video, procesarlo y actualizar la interfaz continuamente. Streamlit normalmente re-ejecuta todo el script cuando el usuario interactúa con algo, pero para video necesitamos un loop continuo. Esto se logró con un checkbox que actúa como interruptor:
```
run = st.checkbox("Iniciar Cámara")

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        # Procesar frame con YOLO
        # Actualizar pantalla
```

Para actualizar la pantalla sin crear scroll infinito, usamos "placeholders" de Streamlit. Creamos espacios vacíos antes del loop y luego los actualizamos en cada iteración:

```
frame_placeholder = st.empty()

while run:
    frame_placeholder.image(frame_procesado)

```
### Calcular la velocidad 
Implementamos una clase PersonTracker que recuerda dónde estaba cada persona y le asigna un ID único. Cuando aparece una nueva detección, busca la persona más cercana de frames anteriores. Si está suficientemente cerca (menos de 150 píxeles), asume que es la misma persona y calcula cuánto se movió:

Calcular la velocidad de personas requirió un sistema de tracking que asocia la misma persona entre frames consecutivos. Simplemente medir distancias entre frames produciría resultados muy ruidosos (la detección puede variar unos píxeles incluso si la persona está quieta).
Implementamos una clase PersonTracker que recuerda dónde estaba cada persona y le asigna un ID único. Cuando aparece una nueva detección, busca la persona más cercana de frames anteriores. Si está suficientemente cerca (menos de 150 píxeles), asume que es la misma persona y calcula cuánto se movió:



```
vel_raw = distancia / tiempo_transcurrido
vel_suave = 0.2 * vel_raw + 0.8 * vel_anterior
```

### Ejecución del Sistema

Para abrir la aplicacion web de streamlit se utilizo 

```
streamlit run app.py
```

