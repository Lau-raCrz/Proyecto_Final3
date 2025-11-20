# Proyecto tercer corte

## Primer Punto 
### Creacion del dataset

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

