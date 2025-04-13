import streamlit as st
from datetime import datetime
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, get_sun, SkyCoord, AltAz
import astropy.units as u
import folium
from streamlit_folium import st_folium

# Importar funciones desde el script original (se elimina la importación de mostrar_horizonte_artificial)
from secuencia_sol_luna8 import (
    obtener_parametros, 
    detectar_eventos, 
    obtener_altitud, 
    cargar_estrellas, 
    descargar_catalogo_estrellas
)

# Función auxiliar para generar la URL de PeakFinder con fecha incluida.
# Si no se introducen manualmente azi y alt, se calcularán a partir de la posición del Sol en el máximo eclipse (st.session_state.t_max)
def get_peakfinder_url(lat, lon, elev, eclipse_time, azi=None, alt=None, fov=110, cfg="sm",
                         teleazi=-77.98, telealt=8.91, name="Frómista"):
    from astropy.coordinates import AltAz
    # Si no se ingresan manualmente y ya se ha calculado el máximo, se calcula la dirección a partir del Sol
    if (not azi or azi == "") and "t_max" in st.session_state:
        location_obj = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
        altaz_frame = AltAz(obstime=st.session_state.t_max, location=location_obj)
        sun_pos = get_sun(st.session_state.t_max).transform_to(altaz_frame)
        azi = f"{sun_pos.az.deg:.2f}"
    if (not alt or alt == "") and "t_max" in st.session_state:
        location_obj = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
        altaz_frame = AltAz(obstime=st.session_state.t_max, location=location_obj)
        sun_pos = get_sun(st.session_state.t_max).transform_to(altaz_frame)
        alt = f"{sun_pos.alt.deg:.2f}"
    # Convertir eclipse_time a formato ISO forzado (ej. 2026-08-12T17:30:00Z)
    date_str = eclipse_time.iso.split('.')[0].replace(" ", "T") + "Z"
    base_url = "https://www.peakfinder.com/es/?"
    params = f"lat={lat}&lng={lon}&ele={int(elev)}&zoom=5"
    if azi is not None and azi != "":
        params += f"&azi={azi}"
    if alt is not None and alt != "":
        params += f"&alt={alt}"
    params += f"&fov={fov}&date={date_str}&cfg={cfg}&teleazi={teleazi}&telealt={telealt}&name={name}"
    return base_url + params

# Función auxiliar para generar la URL de ShadeMap
def get_shademap_url(lat, lon, elev, eclipse_time, zoom=13, bearing=0, pitch=0, margin=0):
    base_url = "https://shademap.app/@"
    lat_str = f"{lat:.3f}"
    lon_str = f"{lon:.3f}"
    if eclipse_time is not None:
        ts_ms = int(eclipse_time.unix * 1000)
    else:
        from time import time as current_time
        ts_ms = int(current_time() * 1000)
    url = f"{base_url}{lat_str},{lon_str},{zoom}z,{ts_ms}t,{bearing}b,{pitch}p,{margin}m"
    return url

# Configuración de la página
st.set_page_config(page_title="Simulador de Eclipse Solar", layout="wide")
st.title("🌘 Simulador de Eclipse Solar")
st.markdown(
    "Esta app calcula los eventos de un eclipse solar (contactos, máximo, duración, etc.) "
    "según la fecha, hora y ubicación que selecciones. Además, en las opciones avanzadas "
    "podrás abrir un horizonte artificial (PeakFinder), cargar un fichero con datos reales del horizonte, "
    "descargar el catálogo de estrellas para el campo de totalidad y abrir ShadeMap."
)

# --- Entradas Básicas: Fecha, Hora y Duración ---
col1, col2 = st.columns(2)
with col1:
    # Fecha por defecto: 2026-08-12
    fecha = st.date_input("📅 Fecha del eclipse", datetime(2026, 8, 12))
with col2:
    # Hora por defecto: 17:30
    hora = st.time_input("⏰ Hora inicial (UT)", datetime.strptime("17:30", "%H:%M").time())
# Duración de simulación por defecto: 120 minutos
duracion_min = st.slider("⏱️ Duración de simulación (minutos)", 5, 120, 120)

# Calcular la fecha/hora del eclipse a partir de los inputs
eclipse_time = Time(datetime.combine(fecha, hora))

# --- Selección de Ubicación mediante Mapa Interactivo ---
st.markdown("### 🌍 Selección de ubicación (haz clic en el mapa)")
if "coords" not in st.session_state:
    st.session_state.coords = {"lat": 40.4168, "lng": -3.7038}  # Por defecto: Madrid

m = folium.Map(location=[st.session_state.coords["lat"], st.session_state.coords["lng"]], zoom_start=5)
folium.Marker(
    [st.session_state.coords["lat"], st.session_state.coords["lng"]],
    tooltip="Ubicación seleccionada",
).add_to(m)
map_data = st_folium(m, height=400, width=700)
if map_data["last_clicked"] is not None:
    st.session_state.coords = map_data["last_clicked"]

# --- Opción para introducir manualmente las coordenadas ---
if st.checkbox("¿Deseas ingresar manualmente las coordenadas de observación?"):
    lat_manual = st.number_input("Latitud (manual)", value=st.session_state.coords.get("lat", 40.4168))
    lon_manual = st.number_input("Longitud (manual)", value=st.session_state.coords.get("lng", -3.7038))
    st.session_state.coords = {"lat": lat_manual, "lng": lon_manual}

lat = st.session_state.coords["lat"]
lon = st.session_state.coords["lng"]
st.write(f"📍 Coordenadas seleccionadas: **Latitud:** {lat:.5f}°, **Longitud:** {lon:.5f}°")

# --- Altitud: Obtenerla automáticamente siempre ---
auto_elev = obtener_altitud(lat, lon)
if auto_elev is not None:
    st.session_state.elev = auto_elev
    st.success(f"Altitud obtenida automáticamente: {auto_elev} metros")
else:
    st.session_state.elev = 667
    st.warning("No se pudo obtener la altitud automáticamente. Se usará 667 metros por defecto.")
elev_default = st.session_state.elev
elev = st.number_input("🗻 Elevación del terreno (m)", value=elev_default)

# --- Opciones Avanzadas ---
with st.expander("🔧 Opciones Avanzadas"):
    st.markdown("#### Horizonte Artificial (PeakFinder)")
    # Si el usuario no ingresa manualmente, se utilizarán los valores calculados del Sol en t_max (si existe)
    azi_input = st.text_input("Azimut (°) para horizonte artificial", value="")
    alt_input = st.text_input("Altitud (°) para horizonte artificial", value="")
    if st.button("Generar enlace a PeakFinder"):
        peak_url = get_peakfinder_url(lat, lon, elev, eclipse_time, azi_input, alt_input)
        st.markdown(f"[Abrir Horizonte Artificial en PeakFinder]({peak_url})", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### ShadeMap")
    zoom_input = st.number_input("Zoom para ShadeMap", value=13, step=1, min_value=1)
    bearing_input = st.number_input("Bearing para ShadeMap", value=0, step=1)
    pitch_input = st.number_input("Pitch para ShadeMap", value=0, step=1)
    margin_input = st.number_input("Margin para ShadeMap", value=0, step=1)
    if st.button("Generar enlace a ShadeMap"):
        eclipse_time_shade = st.session_state.t_max if "t_max" in st.session_state else eclipse_time
        shade_url = get_shademap_url(lat, lon, elev, eclipse_time_shade, zoom=zoom_input, bearing=bearing_input, pitch=pitch_input, margin=margin_input)
        st.markdown(f"[Abrir ShadeMap]({shade_url})", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Cargar fichero con datos reales del horizonte")
    uploaded_horizonte = st.file_uploader("Subir fichero (dos columnas: az alt, con cabecera)", type=["txt", "csv"])
    horizon_data = None
    if uploaded_horizonte is not None:
        try:
            horizon_data = np.loadtxt(uploaded_horizonte, skiprows=1)
            st.success("Fichero de horizonte cargado correctamente.")
        except Exception as e:
            st.error("Error al cargar el fichero: " + str(e))
    
    st.markdown("---")
    descargar_catalogo_flag = st.checkbox("Descargar catálogo de estrellas automáticamente (para el campo de totalidad)")

# --- Botón para ejecutar el cálculo de eventos del eclipse ---
if st.button("🔍 Calcular eventos del eclipse"):
    st.info("Calculando eventos del eclipse...")
    try:
        t_ini = Time(datetime.combine(fecha, hora))
        dt = 10  # segundos entre pasos
        pasos = int((duracion_min * 60) / dt) + 1
        tiempos = t_ini + TimeDelta(np.arange(0, pasos * dt, dt), format='sec')
        location_obj = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
        params_array = [obtener_parametros(t, location_obj) for t in tiempos]
        eventos = detectar_eventos(tiempos, params_array, location_obj)
        
        if "Máximo Eclipse" in eventos:
            st.session_state.t_max = eventos["Máximo Eclipse"]["time"]
        
        st.success("🎯 Eventos del eclipse calculados:")
        for nombre, evento in eventos.items():
            tiempo_evento = evento.get("time", "N/A")
            alt_sol = evento.get("alt_sol", None)
            az_sol = evento.get("az_sol", None)
            mag = evento.get("mag", None)
            if alt_sol is not None:
                st.markdown(f"**🟢 {nombre}** — "
                            f"🕒 **UT:** {tiempo_evento.iso}, "
                            f"🔼 Alt: {alt_sol:.2f}°, "
                            f"➡️ Az: {az_sol:.2f}°, "
                            f"🌑 Magnitud: {mag:.3f}")
            else:
                st.markdown(f"**🟢 {nombre}** — 🕒 **UT:** {tiempo_evento.iso}")
        
        if descargar_catalogo_flag:
            if "Segundo Contacto" in eventos and "Tercer Contacto" in eventos:
                t_max_cat = eventos["Máximo Eclipse"]["time"]
                sun_coord = get_sun(t_max_cat)
                star_catalog = descargar_catalogo_estrellas(sun_coord, radius=2.5, mag_limite=10)
                if star_catalog is not None and len(star_catalog) > 0:
                    st.success("Catálogo de estrellas descargado.")
                    import pandas as pd
                    df_stars = pd.DataFrame(star_catalog, columns=["RA_ICRS", "DE_ICRS", "Gmag"])
                    csv_data = df_stars.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar Catálogo de Estrellas (CSV)", data=csv_data, file_name="catalogo_estrellas.csv", mime="text/csv")
                else:
                    st.warning("No se encontró catálogo de estrellas o está vacío.")
            else:
                st.warning("No se detectó eclipse total; el catálogo de estrellas no se descargará.")
    except Exception as e:
        st.error(f"❌ Error durante el cálculo: {e}")







