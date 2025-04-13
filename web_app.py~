import streamlit as st
from datetime import datetime
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, get_sun, get_body, SkyCoord, AltAz
import astropy.units as u
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse
import webbrowser
import pandas as pd
from io import BytesIO
import base64
import tempfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import os

# Importar funciones desde el script original (excepto las de simulación, que se definen localmente aquí)
from secuencia_sol_luna8 import (
    obtener_parametros, 
    detectar_eventos, 
    obtener_altitud, 
    cargar_estrellas, 
    descargar_catalogo_estrellas,
    angular_diameter,
    aplicar_refraction,
    R_SUN,
    R_MOON,
    simular_eclipse_dual  # Aseguramos que se importa correctamente sin parámetros adicionales
)

# Funciones auxiliares para generar URLs (PeakFinder y ShadeMap)
# Si el usuario no introduce manualmente azi/alt, se calcularán a partir de la posición del Sol en t_max.
def get_peakfinder_url(lat, lon, elev, eclipse_time, azi=None, alt=None, fov=110, cfg="sm",
                         teleazi=-77.98, telealt=8.91, name="Frómista"):
    from astropy.coordinates import AltAz
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

# Función para la simulación del eclipse (sin refracción) adaptada para Streamlit.
def simular_eclipse_streamlit(tiempos, location, horizon_data=None, LIMITE=2.5, anim_interval=200):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-LIMITE, LIMITE)
    ax.set_ylim(-LIMITE, LIMITE)
    ax.set_xlabel("Diferencia en Acimut (°)")
    ax.set_ylabel("Diferencia en Altitud (°)")
    ax.set_title("Simulación del Eclipse Solar (Sin refracción)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    sun_patch = Circle((0, 0), radius=1, color='gold', alpha=0.6, ec='darkgoldenrod', lw=1.5)
    ax.add_patch(sun_patch)
    moon_patch = Circle((0, 0), radius=0.1, color='silver', alpha=0.6, ec='dimgray', lw=1.5)
    ax.add_patch(moon_patch)
    horizon_line, = ax.plot([], [], color='blue', linestyle='--', linewidth=1, label="Horizonte")
    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.legend(loc='upper right')

    def init():
        moon_patch.center = (0, 0)
        time_text.set_text("")
        horizon_line.set_data([], [])
        return sun_patch, moon_patch, time_text, horizon_line

    def update(frame):
        obstime = tiempos[frame]
        altaz_frame = AltAz(obstime=obstime, location=location)
        sun_coord = get_sun(obstime)
        d_sun = sun_coord.distance.to(u.km).value
        sol_diam = angular_diameter(R_SUN, d_sun)
        sun_radius = sol_diam / 2.0
        sun_patch.set_radius(sun_radius)
        sun_patch.center = (0, 0)
        sol = sun_coord.transform_to(altaz_frame)
        luna = get_body("moon", obstime, location=location).transform_to(altaz_frame)
        d_az = (luna.az - sol.az).to(u.deg).value
        d_alt = (luna.alt - sol.alt).to(u.deg).value
        d_x = d_az * np.cos(sol.alt.radian)
        d_y = d_alt
        d_luna = luna.distance.to(u.km).value
        luna_diam = angular_diameter(R_MOON, d_luna)
        moon_radius = luna_diam / 2.0
        moon_patch.center = (d_x, d_y)
        moon_patch.set_radius(moon_radius)
        time_text.set_text(f"UT: {obstime.iso}")
        if horizon_data is not None:
            pts = []
            sol_az = sol.az.to(u.deg).value
            sol_alt = sol.alt.to(u.deg).value
            for point in horizon_data:
                az_h, alt_h = point
                d_az_point = ((az_h - sol_az + 180) % 360) - 180
                x_pt = d_az_point * np.cos(sol.alt.radian)
                y_pt = alt_h - sol_alt
                pts.append([x_pt, y_pt])
            pts = np.array(pts)
            horizon_line.set_data(pts[:,0], pts[:,1])
        else:
            horizon_d_y = 0 - sol.alt.to(u.deg).value
            if -LIMITE <= horizon_d_y <= LIMITE:
                x_vals = np.array([-LIMITE, LIMITE])
                y_vals = np.array([horizon_d_y, horizon_d_y])
                horizon_line.set_data(x_vals, y_vals)
            else:
                horizon_line.set_data([], [])
        return sun_patch, moon_patch, time_text, horizon_line

    ani = animation.FuncAnimation(fig, update, frames=len(tiempos),
                                  init_func=init, blit=True, interval=anim_interval, repeat=True)
    plt.close(fig)
    return ani.to_html5_video()
    
# Función para la simulación dual adaptada para Streamlit
def simular_eclipse_dual_streamlit(tiempos, location, horizon_data=None, star_catalog_data=None, t_total_range=None, LIMITE=2.5, anim_interval=200):
    from RefractionShift.refraction_shift import refraction
    Refraction_inst = refraction(288.15, 101325, location.height.value)
    lmbda = 550e-9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax in (ax1, ax2):
        ax.set_xlim(-LIMITE, LIMITE)
        ax.set_ylim(-LIMITE, LIMITE)
        ax.set_xlabel("Diferencia en Acimut (°)")
        ax.set_ylabel("Diferencia en Altitud (°)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_title("Sin refracción")
    ax2.set_title("Con refracción")

    sun_patch1 = Circle((0, 0), radius=1, color='gold', alpha=0.6, ec='darkgoldenrod', lw=1.5)
    sun_patch2 = Circle((0, 0), radius=1, color='gold', alpha=0.6, ec='darkgoldenrod', lw=1.5)
    ax1.add_patch(sun_patch1)
    ax2.add_patch(sun_patch2)

    moon_patch1 = Circle((0, 0), radius=0.1, color='silver', alpha=0.6, ec='dimgray', lw=1.5)
    moon_patch2 = Circle((0, 0), radius=0.1, color='silver', alpha=0.6, ec='dimgray', lw=1.5)
    ax1.add_patch(moon_patch1)
    ax2.add_patch(moon_patch2)

    time_text1 = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    time_text2 = ax2.text(0.05, 0.95, "", transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    def init():
        moon_patch1.center = (0, 0)
        moon_patch2.center = (0, 0)
        time_text1.set_text("")
        time_text2.set_text("")
        return moon_patch1, moon_patch2, time_text1, time_text2

    def update(frame):
        from secuencia_sol_luna8 import aplicar_refraction, angular_diameter, R_SUN, R_MOON
        from astropy.coordinates import get_sun, get_body, AltAz
        import astropy.units as u

        obstime = tiempos[frame]
        altaz_frame = AltAz(obstime=obstime, location=location)

        sun_coord = get_sun(obstime)
        d_sun = sun_coord.distance.to(u.km).value
        sol_diam = angular_diameter(R_SUN, d_sun)
        sun_radius = sol_diam / 2.0
        sun_patch1.set_radius(sun_radius)
        sun_patch2.set_radius(sun_radius)
        sol = sun_coord.transform_to(altaz_frame)
        luna = get_body("moon", obstime, location=location).transform_to(altaz_frame)

        d_az = (luna.az - sol.az).to(u.deg).value
        d_alt = (luna.alt - sol.alt).to(u.deg).value
        d_x = d_az * np.cos(sol.alt.radian)
        d_y = d_alt
        d_luna = luna.distance.to(u.km).value
        luna_diam = angular_diameter(R_MOON, d_luna)
        moon_radius = luna_diam / 2.0
        moon_patch1.center = (d_x, d_y)
        moon_patch1.set_radius(moon_radius)
        time_text1.set_text(f"UT: {obstime.iso}")

        alt_sol = sol.alt.to(u.deg).value
        az_sol = sol.az.to(u.deg).value
        sol_eff_alt, sol_eff_az = aplicar_refraction(alt_sol, az_sol, d_sun, Refraction_inst, lmbda)

        alt_luna = luna.alt.to(u.deg).value
        az_luna = luna.az.to(u.deg).value
        luna_eff_alt, luna_eff_az = aplicar_refraction(alt_luna, az_luna, d_luna, Refraction_inst, lmbda)

        d_az_eff = (luna_eff_az - sol_eff_az)
        d_x_eff = d_az_eff * np.cos(np.deg2rad(sol_eff_alt))
        d_y_eff = luna_eff_alt - sol_eff_alt

        moon_patch2.center = (d_x_eff, d_y_eff)
        moon_patch2.set_radius(moon_radius)
        time_text2.set_text(f"UT: {obstime.iso}")
        return moon_patch1, moon_patch2, time_text1, time_text2

    ani = animation.FuncAnimation(fig, update, frames=len(tiempos),
                                  init_func=init, blit=True, interval=anim_interval, repeat=True)
    plt.close(fig)
    return ani.to_html5_video()    

# ----------------- Configuración de la App -----------------
st.set_page_config(page_title="Simulador de Eclipse Solar", layout="wide")
st.title("🌘 Simulador de Eclipse Solar")
st.markdown(
    "Esta app calcula los eventos de un eclipse solar (contactos, máximo, duración, etc.) "
    "y permite generar animaciones de la simulación (sin refracción) según la fecha, hora y ubicación que selecciones. "
    "Además, en las opciones avanzadas podrás abrir enlaces a PeakFinder y ShadeMap, "
    "cargar un fichero con datos reales del horizonte y descargar el catálogo de estrellas para el campo de totalidad."
)

# --- Entradas Básicas: Fecha, Hora y Duración ---
col1, col2 = st.columns(2)
with col1:
    fecha = st.date_input("📅 Fecha del eclipse", datetime(2026, 8, 12))
with col2:
    hora = st.time_input("⏰ Hora inicial (UT)", datetime.strptime("17:30", "%H:%M").time())
# Duración de simulación por defecto: 120 minutos
duracion_min = st.slider("⏱️ Duración de simulación (minutos)", 5, 120, 120)

eclipse_time = Time(datetime.combine(fecha, hora))

# --- Selección de Ubicación ---
st.markdown("### 🌍 Selección de ubicación (haz clic en el mapa)")
if "coords" not in st.session_state:
    st.session_state.coords = {"lat": 40.4168, "lng": -3.7038}

m = folium.Map(location=[st.session_state.coords["lat"], st.session_state.coords["lng"]], zoom_start=5)
folium.Marker(
    [st.session_state.coords["lat"], st.session_state.coords["lng"]],
    tooltip="Ubicación seleccionada",
).add_to(m)
map_data = st_folium(m, height=400, width=700)
if map_data["last_clicked"] is not None:
    st.session_state.coords = map_data["last_clicked"]

# Opción para ingresar manualmente las coordenadas (opcional)
if st.checkbox("¿Deseas ingresar manualmente las coordenadas de observación?"):
    lat_manual = st.number_input("Latitud (manual)", value=st.session_state.coords.get("lat", 40.4168))
    lon_manual = st.number_input("Longitud (manual)", value=st.session_state.coords.get("lng", -3.7038))
    st.session_state.coords = {"lat": lat_manual, "lng": lon_manual}

lat = st.session_state.coords["lat"]
lon = st.session_state.coords["lng"]
st.write(f"📍 Coordenadas seleccionadas: **Latitud:** {lat:.5f}°, **Longitud:** {lon:.5f}°")

# --- Altitud: Obtenerla automáticamente ---
auto_elev = obtener_altitud(lat, lon)
if auto_elev is not None:
    st.session_state.elev = auto_elev
    st.success(f"Altitud obtenida automáticamente: {auto_elev} metros")
else:
    st.session_state.elev = 667
    st.warning("No se pudo obtener la altitud automáticamente. Se usará 667 metros por defecto.")
elev = st.number_input("🗻 Elevación del terreno (m)", value=st.session_state.elev)

# --- Opciones Avanzadas ---
with st.expander("🔧 Opciones Avanzadas"):
    st.markdown("#### Horizonte Artificial (PeakFinder)")
    # Si el usuario no ingresa manualmente, se usarán los valores calculados del Sol en t_max (si existen)
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
        location_obj = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
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
                    df_stars = pd.DataFrame(star_catalog, columns=["RA_ICRS", "DE_ICIS", "Gmag"])
                    csv_data = df_stars.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar Catálogo de Estrellas (CSV)", data=csv_data, file_name="catalogo_estrellas.csv", mime="text/csv")
                else:
                    st.warning("No se encontró catálogo de estrellas o está vacío.")
            else:
                st.warning("No se detectó eclipse total; el catálogo de estrellas no se descargará.")
    except Exception as e:
        st.error(f"❌ Error durante el cálculo: {e}")

# --- Botón para mostrar la simulación (sin refracción) ---
if st.button("▶️ Mostrar simulación del eclipse (sin refracción)"):
    st.info("Generando simulación...")
    try:
        t_ini_sim = Time(datetime.combine(fecha, hora))
        dt_sim = 10
        pasos_sim = int((duracion_min * 60) / dt_sim) + 1
        tiempos_sim = t_ini_sim + TimeDelta(np.arange(0, pasos_sim * dt_sim, dt_sim), format='sec')
        location_obj = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
        video_html = simular_eclipse_streamlit(tiempos_sim, location_obj, horizon_data=horizon_data, LIMITE=2.5, anim_interval=200)
        video_html_mod = f'''
        <video width="100%" controls playsinline>
            {video_html}
            Tu navegador no soporta la etiqueta de video HTML5.
        </video>'''
        st.markdown(video_html_mod, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error en la simulación: {e}")
        
# --- Simulación dual ---
# --- Botón para mostrar la simulación del eclipse (dual: con y sin refracción) ---
if st.button("▶️ Mostrar simulación del eclipse (dual: con(basico) y sin refracción)[Tarda mucho, no simular más de 5 min]"):
    st.info("Generando simulación dual...")
    try:
        t_ini_sim = Time(datetime.combine(fecha, hora))
        dt_sim = 10
        pasos_sim = int((duracion_min * 60) / dt_sim) + 1
        tiempos_sim = t_ini_sim + TimeDelta(np.arange(0, pasos_sim * dt_sim, dt_sim), format='sec')
        location_obj = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)

        from RefractionShift.refraction_shift import refraction
        Refraction_inst = refraction(288.15, 101325, location_obj.height.value)
        lmbda = 550e-9

        progress_bar = st.progress(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        for ax in (ax1, ax2):
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xlabel("Diferencia en Acimut (°)")
            ax.set_ylabel("Diferencia en Altitud (°)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_title("Sin refracción")
        ax2.set_title("Con refracción")

        sun_patch1 = Circle((0, 0), radius=1, color='gold', alpha=0.6, ec='darkgoldenrod', lw=1.5)
        sun_patch2 = Circle((0, 0), radius=1, color='gold', alpha=0.6, ec='darkgoldenrod', lw=1.5)
        ax1.add_patch(sun_patch1)
        ax2.add_patch(sun_patch2)

        moon_patch1 = Circle((0, 0), radius=0.1, color='silver', alpha=0.6, ec='dimgray', lw=1.5)
        moon_patch2 = Circle((0, 0), radius=0.1, color='silver', alpha=0.6, ec='dimgray', lw=1.5)
        ax1.add_patch(moon_patch1)
        ax2.add_patch(moon_patch2)

        horizon_line1, = ax1.plot([], [], color='blue', linestyle='--', linewidth=1)
        horizon_line2, = ax2.plot([], [], color='blue', linestyle='--', linewidth=1)

        time_text1 = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        time_text2 = ax2.text(0.05, 0.95, "", transform=ax2.transAxes, fontsize=10, verticalalignment='top')

        def init():
            moon_patch1.center = (0, 0)
            moon_patch2.center = (0, 0)
            time_text1.set_text("")
            time_text2.set_text("")
            horizon_line1.set_data([], [])
            horizon_line2.set_data([], [])
            return moon_patch1, moon_patch2, time_text1, time_text2, horizon_line1, horizon_line2

        def update(frame):
            from secuencia_sol_luna8 import aplicar_refraction, angular_diameter, R_SUN, R_MOON
            from astropy.coordinates import get_sun, get_body, AltAz
            import astropy.units as u

            obstime = tiempos_sim[frame]
            altaz_frame = AltAz(obstime=obstime, location=location_obj)
            sol = get_sun(obstime).transform_to(altaz_frame)
            luna = get_body("moon", obstime, location=location_obj).transform_to(altaz_frame)
            d_sun = get_sun(obstime).distance.to(u.km).value
            d_luna = luna.distance.to(u.km).value

            sol_diam = angular_diameter(R_SUN, d_sun)
            luna_diam = angular_diameter(R_MOON, d_luna)
            sun_radius = sol_diam / 2.0
            moon_radius = luna_diam / 2.0

            sun_patch1.set_radius(sun_radius)
            sun_patch2.set_radius(sun_radius)

            d_az = (luna.az - sol.az).to(u.deg).value
            d_alt = (luna.alt - sol.alt).to(u.deg).value
            d_x = d_az * np.cos(sol.alt.radian)
            d_y = d_alt
            moon_patch1.center = (d_x, d_y)
            moon_patch1.set_radius(moon_radius)
            time_text1.set_text(f"UT: {obstime.iso}")

            alt_sol = sol.alt.to(u.deg).value
            az_sol = sol.az.to(u.deg).value
            sol_eff_alt, sol_eff_az = aplicar_refraction(alt_sol, az_sol, d_sun, Refraction_inst, lmbda)

            alt_luna = luna.alt.to(u.deg).value
            az_luna = luna.az.to(u.deg).value
            luna_eff_alt, luna_eff_az = aplicar_refraction(alt_luna, az_luna, d_luna, Refraction_inst, lmbda)

            d_az_eff = (luna_eff_az - sol_eff_az)
            d_x_eff = d_az_eff * np.cos(np.deg2rad(sol_eff_alt))
            d_y_eff = luna_eff_alt - sol_eff_alt
            moon_patch2.center = (d_x_eff, d_y_eff)
            moon_patch2.set_radius(moon_radius)
            time_text2.set_text(f"UT: {obstime.iso}")

            if horizon_data is None:
                horizon_y = 0 - sol.alt.to(u.deg).value
                if -2.5 <= horizon_y <= 2.5:
                    x_vals = np.array([-2.5, 2.5])
                    y_vals = np.array([horizon_y, horizon_y])
                    horizon_line1.set_data(x_vals, y_vals)
                else:
                    horizon_line1.set_data([], [])

                horizon_y_eff = 0 - sol_eff_alt
                if -2.5 <= horizon_y_eff <= 2.5:
                    x_vals = np.array([-2.5, 2.5])
                    y_vals = np.array([horizon_y_eff, horizon_y_eff])
                    horizon_line2.set_data(x_vals, y_vals)
                else:
                    horizon_line2.set_data([], [])
            else:
                pts = [( ((az_h - sol.az.deg + 180) % 360 - 180) * np.cos(sol.alt.radian), alt_h - sol.alt.deg)
                       for az_h, alt_h in horizon_data]
                pts = np.array(pts)
                horizon_line1.set_data(pts[:, 0], pts[:, 1])

                pts2 = [( ((az_h - sol_eff_az + 180) % 360 - 180) * np.cos(np.deg2rad(sol_eff_alt)), alt_h - sol_eff_alt)
                        for az_h, alt_h in horizon_data]
                pts2 = np.array(pts2)
                horizon_line2.set_data(pts2[:, 0], pts2[:, 1])

            progress_bar.progress((frame + 1) / len(tiempos_sim))
            return moon_patch1, moon_patch2, time_text1, time_text2, horizon_line1, horizon_line2

        ani = animation.FuncAnimation(fig, update, frames=len(tiempos_sim), init_func=init, blit=True, interval=200, repeat=True)
        video_path = "/tmp/eclipse_dual.mp4"
        ani.save(video_path, writer="ffmpeg", fps=5)
        progress_bar.empty()
        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode()
            video_html = f'''
            <video width="100%" controls playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Tu navegador no soporta la etiqueta de video HTML5.
            </video>'''
            st.markdown(video_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error en la simulación dual: {e}")







    


