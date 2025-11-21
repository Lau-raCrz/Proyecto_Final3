import os
import time
import threading
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS
import random

# ============================================
# SINCRONIZACIÓN
# ============================================
mutex = threading.Lock()
semaforo = threading.Semaphore(5)  # Reducido a 5 para evitar rate limit

# ============================================
# OBTENER URLs CON DELAYS
# ============================================
def obtener_urls(elemento, cantidad=250):
    """
    Busca imágenes con delays para evitar rate limit
    """
    print(f"🔍 Buscando URLs para: {elemento}")
    urls = []
    
    intentos = 0
    max_intentos = 3
    
    while intentos < max_intentos and len(urls) < cantidad:
        try:
            intentos += 1
            print(f"   Intento {intentos}/{max_intentos}...")
            
            # Delay aleatorio entre 3-7 segundos
            delay = random.uniform(3, 7)
            print(f"   Esperando {delay:.1f}s para evitar rate limit...")
            time.sleep(delay)
            
            with DDGS() as ddgs:
                # Búsqueda en lotes pequeños
                lote = min(100, cantidad - len(urls))
                resultados = ddgs.images(
                    keywords=elemento,
                    region='wt-wt',
                    safesearch='off',
                    size=None,
                    max_results=lote
                )
                
                for r in resultados:
                    url = r.get("image", "")
                    if url and url.startswith("http") and url not in urls:
                        urls.append(url)
                        if len(urls) >= cantidad:
                            break
                
                print(f"   ✓ Encontradas {len(urls)} URLs hasta ahora")
                
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "Ratelimit" in error_msg:
                print(f"   ⚠️  Rate limit detectado. Esperando 30 segundos...")
                time.sleep(30)
            else:
                print(f"   ⚠️  Error: {error_msg[:100]}")
                time.sleep(5)
    
    print(f"✅ Total URLs obtenidas: {len(urls)}")
    return urls

# ============================================
# DESCARGAR IMAGEN
# ============================================
def descargar_imagen(url, carpeta, contador):
    semaforo.acquire()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            with mutex:
                timestamp = int(time.time() * 1000000)
                nombre = f"img_{timestamp}_{contador[0]:04d}.jpg"
                ruta = os.path.join(carpeta, nombre)
                
                img.save(ruta, 'JPEG', quality=85)
                contador[0] += 1
                
                if contador[0] % 10 == 0:
                    print(f"  ✓ Descargadas: {contador[0]}")
            
    except Exception as e:
        pass
    
    finally:
        semaforo.release()
        time.sleep(0.2)  # Pequeño delay entre descargas

# ============================================
# DESCARGAR DATASET
# ============================================
def descargar_dataset(elemento, carpeta_destino, cantidad=250):
    print("\n" + "="*70)
    print(f"📦 {elemento}")
    print("="*70)
    
    os.makedirs(carpeta_destino, exist_ok=True)
    
    # Contar imágenes existentes
    existentes = len([f for f in os.listdir(carpeta_destino) if f.endswith('.jpg')])
    if existentes > 0:
        print(f"ℹ️  Ya existen {existentes} imágenes. Continuando...")
        respuesta = input("¿Descargar más? (s/n): ").lower()
        if respuesta != 's':
            print("⏭️  Saltando categoría...")
            return
    
    urls = obtener_urls(elemento, cantidad)
    
    if not urls:
        print(f"❌ No se pudieron obtener URLs")
        return
    
    contador = [existentes]
    hilos = []
    
    print(f"\n🚀 Descargando {len(urls)} imágenes...")
    
    for i, url in enumerate(urls):
        hilo = threading.Thread(
            target=descargar_imagen,
            args=(url, carpeta_destino, contador)
        )
        hilos.append(hilo)
        hilo.start()
        
        if i % 5 == 0:
            time.sleep(0.5)
    
    for hilo in hilos:
        hilo.join()
    
    print(f"\n✅ COMPLETADO: {contador[0]} imágenes totales en '{carpeta_destino}'")
    print("="*70 + "\n")

# ============================================
# CONFIGURACIÓN
# ============================================
DATASET_CONFIG = [
    ('Multimetro electronico', 'dataset/Multimetro_electronico', 200),
    ('Osciloscopio', 'dataset/Osciloscopio', 200),
    ('Cautin para soldar', 'dataset/Cautin_para_soldar', 200),
    ('Protoboard', 'dataset/Protoboard', 200),
    ('Capacitor electronico', 'dataset/Capacitor_electronico', 200),
    ('Resistencia electronica', 'dataset/Resistencia_electronica', 200),
    ('LED electronico', 'dataset/LED_electronico', 200),
    ('Destornillador electronica', 'dataset/Destornillador_electronica', 200),
    ('Fuente de alimentacion DC', 'dataset/Fuente_de_alimentacion_dc', 200),
    ('Transformador electronico', 'dataset/Transformador_electronico', 200),
    ('Raspberry Pi', 'dataset/Raspberry_Pi', 200),
]

# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "🎯 "*35)
    print("CONSTRUCCIÓN DEL DATASET v2.0 (Con protección anti-rate-limit)")
    print("🎯 "*35 + "\n")
    
    print("📋 Configuración:")
    print(f"   • Categorías: {len(DATASET_CONFIG)}")
    print(f"   • Imágenes por categoría: ~200")
    print(f"   • Hilos simultáneos: 5")
    print(f"   • Delays entre búsquedas: 3-7s aleatorios")
    print(f"   • Total estimado: ~{len(DATASET_CONFIG) * 200} imágenes")
    print()
    
    input("⏸️  Presiona ENTER para comenzar (esto tomará tiempo)...\n")
    
    tiempo_inicio = time.time()
    
    for i, (nombre, carpeta, cantidad) in enumerate(DATASET_CONFIG, 1):
        print(f"\n{'🔵'*35}")
        print(f"Progreso: {i}/{len(DATASET_CONFIG)} categorías")
        print(f"{'🔵'*35}\n")
        
        descargar_dataset(nombre, carpeta, cantidad)
        
        # Delay más largo entre categorías para evitar ban
        if i < len(DATASET_CONFIG):
            delay = random.uniform(10, 20)
            print(f"⏳ Esperando {delay:.1f}s antes de la siguiente categoría...\n")
            time.sleep(delay)
    
    tiempo_total = time.time() - tiempo_inicio
    
    print("\n" + "🎉 "*35)
    print("DATASET COMPLETADO")
    print("🎉 "*35)
    print(f"\n⏱️  Tiempo total: {tiempo_total/60:.1f} minutos")
    
    # Resumen final
    print("\n📊 RESUMEN:")
    for nombre, carpeta, _ in DATASET_CONFIG:
        if os.path.exists(carpeta):
            total = len([f for f in os.listdir(carpeta) if f.endswith('.jpg')])
            print(f"   {nombre}: {total} imágenes")

if __name__ == "__main__":
    main()
