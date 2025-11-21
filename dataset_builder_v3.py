import os
import time
import threading
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS
import random

mutex = threading.Lock()
semaforo = threading.Semaphore(5)

def obtener_urls(elemento, cantidad=200):
    print(f"🔍 Buscando URLs para: {elemento}")
    urls = []
    
    intentos = 0
    max_intentos = 3
    
    while intentos < max_intentos and len(urls) < cantidad:
        try:
            intentos += 1
            print(f"   Intento {intentos}/{max_intentos}...")
            
            delay = random.uniform(3, 7)
            print(f"   Esperando {delay:.1f}s...")
            time.sleep(delay)
            
            ddgs = DDGS()
            lote = min(100, cantidad - len(urls))
            
            # CORRECCIÓN: usar 'query' en lugar de 'keywords'
            resultados = ddgs.images(
                query=elemento,  # ← CAMBIO AQUÍ
                region='wt-wt',
                safesearch='off',
                max_results=lote
            )
            
            for r in resultados:
                url = r.get("image", "")
                if url and url.startswith("http") and url not in urls:
                    urls.append(url)
                    if len(urls) >= cantidad:
                        break
            
            print(f"   ✓ Encontradas {len(urls)} URLs")
            
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "Ratelimit" in error_msg:
                print(f"   ⚠️  Rate limit. Esperando 30s...")
                time.sleep(30)
            else:
                print(f"   ⚠️  Error: {error_msg[:100]}")
                time.sleep(5)
    
    print(f"✅ Total: {len(urls)} URLs")
    return urls

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
            
    except:
        pass
    
    finally:
        semaforo.release()
        time.sleep(0.2)

def descargar_dataset(elemento, carpeta_destino, cantidad=200):
    print("\n" + "="*70)
    print(f"📦 {elemento}")
    print("="*70)
    
    os.makedirs(carpeta_destino, exist_ok=True)
    
    existentes = len([f for f in os.listdir(carpeta_destino) if f.endswith('.jpg')])
    if existentes > 0:
        print(f"ℹ️  Ya existen {existentes} imágenes")
    
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
    
    total_final = len([f for f in os.listdir(carpeta_destino) if f.endswith('.jpg')])
    print(f"\n✅ TOTAL: {total_final} imágenes en '{carpeta_destino}'")

DATASET_CONFIG = [
    ('Multimetro electronico', 'dataset/Multimetro_electronico', 300),
    ('Osciloscopio', 'dataset/Osciloscopio', 300),
    ('Cautin para soldar', 'dataset/Cautin_para_soldar', 300),
    ('Protoboard', 'dataset/Protoboard', 300),
    ('Capacitor electronico', 'dataset/Capacitor_electronico', 300),
    ('Resistencia electronica', 'dataset/Resistencia_electronica', 300),
    ('LED electronico', 'dataset/LED_electronico', 300),
    ('Destornillador electronica', 'dataset/Destornillador_electronica', 300),
    ('Fuente de alimentacion DC', 'dataset/Fuente_de_alimentacion_dc', 300),
    ('Transformador electronico', 'dataset/Transformador_electronico', 300),
    ('Raspberry Pi', 'dataset/Raspberry_Pi', 300),
]

def main():
    print("\n" + "🎯 "*35)
    print("CONSTRUCCIÓN DEL DATASET v3.0 (API corregida)")
    print("🎯 "*35 + "\n")
    
    print("📋 Configuración:")
    print(f"   • Categorías: {len(DATASET_CONFIG)}")
    print(f"   • Imágenes por categoría: ~200")
    print(f"   • Delays anti-rate-limit activados")
    print()
    
    input("⏸️  Presiona ENTER para comenzar...\n")
    
    tiempo_inicio = time.time()
    
    for i, (nombre, carpeta, cantidad) in enumerate(DATASET_CONFIG, 1):
        print(f"\n{'🔵'*35}")
        print(f"Progreso: {i}/{len(DATASET_CONFIG)} categorías")
        print(f"{'🔵'*35}\n")
        
        descargar_dataset(nombre, carpeta, cantidad)
        
        if i < len(DATASET_CONFIG):
            delay = random.uniform(10, 20)
            print(f"\n⏳ Esperando {delay:.1f}s antes de siguiente categoría...")
            time.sleep(delay)
    
    tiempo_total = time.time() - tiempo_inicio
    
    print("\n" + "🎉 "*35)
    print("DATASET COMPLETADO")
    print("🎉 "*35)
    print(f"\n⏱️  Tiempo total: {tiempo_total/60:.1f} minutos")
    
    print("\n📊 RESUMEN FINAL:")
    total_general = 0
    for nombre, carpeta, _ in DATASET_CONFIG:
        if os.path.exists(carpeta):
            total = len([f for f in os.listdir(carpeta) if f.endswith('.jpg')])
            total_general += total
            print(f"   • {nombre}: {total} imágenes")
    print(f"\n📦 TOTAL GENERAL: {total_general} imágenes")

if __name__ == "__main__":
    main()
