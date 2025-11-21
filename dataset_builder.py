import os
import time
import threading
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS

# ============================================
# SINCRONIZACIÓN: MUTEX Y SEMÁFORO
# ============================================
mutex = threading.Lock()
semaforo = threading.Semaphore(10)  # Máximo 10 descargas simultáneas

# ============================================
# FUNCIÓN: OBTENER URLs DE IMÁGENES
# ============================================
def obtener_urls(elemento, cantidad=200):
    """
    Busca imágenes en DuckDuckGo y devuelve lista de URLs
    """
    print(f"🔍 Buscando {cantidad} URLs para: {elemento}")
    urls = []
    try:
        with DDGS() as ddgs:
            resultados = ddgs.images(
                keywords=elemento,
                max_results=cantidad,
                safesearch='off'
            )
            for r in resultados:
                url = r.get("image", "")
                if url.startswith("http"):
                    urls.append(url)
    except Exception as e:
        print(f"❌ Error al buscar URLs: {e}")
    
    print(f"✅ Se encontraron {len(urls)} URLs válidas")
    return urls

# ============================================
# FUNCIÓN: DESCARGAR UNA IMAGEN
# ============================================
def descargar_imagen(url, carpeta, contador):
    """
    Descarga una imagen desde URL y la guarda en la carpeta
    Usa semáforo para limitar descargas simultáneas
    Usa mutex para proteger escritura de archivos
    """
    semaforo.acquire()  # Esperar turno para descargar
    
    try:
        # Descargar imagen
        response = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        
        if response.status_code == 200:
            # Abrir imagen
            img = Image.open(BytesIO(response.content))
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # ===== SECCIÓN CRÍTICA =====
            with mutex:
                # Generar nombre único
                timestamp = int(time.time() * 1000)
                nombre = f"img_{timestamp}_{contador[0]}.jpg"
                ruta = os.path.join(carpeta, nombre)
                
                # Guardar imagen
                img.save(ruta, 'JPEG')
                contador[0] += 1
                
                print(f"  ✓ Descargada: {nombre}")
            # ===== FIN SECCIÓN CRÍTICA =====
            
    except Exception as e:
        print(f"  ✗ Error en descarga: {str(e)[:50]}")
    
    finally:
        semaforo.release()  # Liberar semáforo

# ============================================
# FUNCIÓN: DESCARGAR DATASET COMPLETO
# ============================================
def descargar_dataset(elemento, carpeta_destino, cantidad=200):
    """
    Descarga un dataset completo para un elemento específico
    usando hilos para paralelizar las descargas
    """
    print("\n" + "="*60)
    print(f"📦 INICIANDO DESCARGA: {elemento}")
    print("="*60)
    
    # Crear carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)
    
    # Obtener URLs
    urls = obtener_urls(elemento, cantidad)
    
    if not urls:
        print(f"❌ No se encontraron URLs para {elemento}")
        return
    
    # Contador compartido (lista para que sea mutable)
    contador = [0]
    
    # Crear hilos para descargar en paralelo
    hilos = []
    print(f"\n🚀 Descargando {len(urls)} imágenes con 10 hilos...")
    
    for i, url in enumerate(urls):
        hilo = threading.Thread(
            target=descargar_imagen,
            args=(url, carpeta_destino, contador)
        )
        hilos.append(hilo)
        hilo.start()
        
        # Pequeña pausa para no saturar
        if i % 10 == 0:
            time.sleep(0.1)
    
    # Esperar a que todos terminen
    for hilo in hilos:
        hilo.join()
    
    print(f"\n✅ COMPLETADO: {contador[0]} imágenes descargadas en '{carpeta_destino}'")
    print("="*60 + "\n")

# ============================================
# CONFIGURACIÓN DEL DATASET
# ============================================
DATASET_CONFIG = [
    {
        'nombre': 'Multimetro electronico',
        'carpeta': 'dataset/Multimetro_electronico',
        'cantidad': 250
    },
    {
        'nombre': 'Osciloscopio',
        'carpeta': 'dataset/Osciloscopio',
        'cantidad': 250
    },
    {
        'nombre': 'Cautin para soldar',
        'carpeta': 'dataset/Cautin_para_soldar',
        'cantidad': 250
    },
    {
        'nombre': 'Protoboard',
        'carpeta': 'dataset/Protoboard',
        'cantidad': 250
    },
    {
        'nombre': 'Capacitor electronico',
        'carpeta': 'dataset/Capacitor_electronico',
        'cantidad': 250
    },
    {
        'nombre': 'Resistencia electronica',
        'carpeta': 'dataset/Resistencia_electronica',
        'cantidad': 250
    },
    {
        'nombre': 'LED electronico',
        'carpeta': 'dataset/LED_electronico',
        'cantidad': 250
    },
    {
        'nombre': 'Destornillador electronica',
        'carpeta': 'dataset/Destornillador_electronica',
        'cantidad': 250
    },
    {
        'nombre': 'Fuente de alimentacion DC',
        'carpeta': 'dataset/Fuente_de_alimentacion_dc',
        'cantidad': 250
    },
    {
        'nombre': 'Transformador electronico',
        'carpeta': 'dataset/Transformador_electronico',
        'cantidad': 250
    },
    {
        'nombre': 'Raspberry Pi',
        'carpeta': 'dataset/Raspberry_Pi',
        'cantidad': 250
    }
]

# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main():
    print("\n" + "🎯 "*30)
    print("CONSTRUCCIÓN AUTOMÁTICA DEL DATASET")
    print("🎯 "*30 + "\n")
    
    print("📋 Configuración:")
    print(f"   • Total de categorías: {len(DATASET_CONFIG)}")
    print(f"   • Imágenes por categoría: ~250")
    print(f"   • Total estimado: ~{len(DATASET_CONFIG) * 250} imágenes")
    print(f"   • Hilos simultáneos: 10")
    print()
    
    input("⏸️  Presiona ENTER para comenzar...")
    
    tiempo_inicio = time.time()
    
    # Descargar cada categoría
    for config in DATASET_CONFIG:
        descargar_dataset(
            elemento=config['nombre'],
            carpeta_destino=config['carpeta'],
            cantidad=config['cantidad']
        )
    
    tiempo_total = time.time() - tiempo_inicio
    
    print("\n" + "🎉 "*30)
    print("DATASET COMPLETADO")
    print("🎉 "*30)
    print(f"\n⏱️  Tiempo total: {tiempo_total/60:.2f} minutos")
    print(f"📁 Ubicación: ./dataset/")
    print()

if __name__ == "__main__":
    main()
