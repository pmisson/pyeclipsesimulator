#!/usr/bin/env python3
"""
Simulador de Eclipse Solar con Streamlit.
Requiere:
  - streamlit, numpy, astropy, folium, streamlit-folium,
    matplotlib, requests, geopy, astroquery, RefractionShift
"""
import streamlit as st
from datetime import datetime
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body, SkyCoord
import astropy.units as u
import requests
from geopy.geocoders import Nominatim
from astroquery.vizier import Vizier
from RefractionShift.refraction_shift import refraction as refraction_shift
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse
import folium
from streamlit_folium import st_folium
import pandas as pd
import base64
import tempfile
import os

# --- Constantes físicas ---
R_SUN = 696340.0  # km
R_MOON = 1737.4   # km

# --- Lista de capitales de provincia en España ---
capitales = {
    "Alcalá de Henares": (40.48198, -3.36354),
    "Bilbao": (43.26300, -2.93400),
    "Burgos": (42.34399, -3.69691),
    "Castellón de la Plana": (39.98641, -0.03688),
    "Cuenca": (40.07039, -2.13742),
    "Gijón": (43.53570, -5.66150),
    "Guadalajara": (40.63333, -3.16667),
    "León": (42.59873, -5.56710),
    "Lleida": (41.61759,  0.62001),
    "Logroño": (42.46645, -2.44566),
    "Oviedo": (43.36030, -5.84480),
    "Palma": (39.56960,  2.65020),
    "Santander": (43.46298, -3.80475),
    "Valencia": (39.46990, -0.37630),
    "Valladolid": (41.65280, -4.72450),
    "Zaragoza": (41.64880, -0.88910)
}

# --- Funciones auxiliares ---
@st.cache_data(ttl=86400)
def obtener_altitud(lat, lng):
    """Devuelve la altitud en metros usando la API Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lng}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "elevation" in data and isinstance(data["elevation"], list):
            return data["elevation"][0]
    except:
        pass
    return None

@st.cache_data(ttl=86400)
def geocode_lugar(nombre):
    geolocator = Nominatim(user_agent="eclipse_app")
    try:
        loc = geolocator.geocode(nombre, language="es", timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except:
        pass
    return None, None, None

@st.cache_data(ttl=86400)
def reverse_geocode(lat, lng):
    geolocator = Nominatim(user_agent="eclipse_app")
    try:
        loc = geolocator.reverse((lat, lng), language="es", timeout=10)
        return loc.address if loc else "Desconocido"
    except:
        return "Error"

@st.cache_data(ttl=86400)
def descargar_catalogo_estrellas(center_coord, radius=2.5, mag_limite=6.5):
    """Descarga catálogo de estrellas cercano al Sol con Astroquery Vizier."""
    Vizier.ROW_LIMIT = -1
    center = SkyCoord(ra=center_coord.ra, dec=center_coord.dec, unit='deg', frame='icrs')
    viz = Vizier(columns=["RA_ICRS","DE_ICRS","Gmag"], row_limit=-1)
    try:
        res = viz.query_region(center, radius=radius*u.deg, catalog="I/355/gaiadr3")
        if res:
            df = res[0].to_pandas()
            df = df[df.Gmag < mag_limite]
            return df
    except:
        pass
    return None

# --- URLs externas: PeakFinder y ShadeMap ---

def get_peakfinder_url(lat, lon, elev, eclipse_time, azi=282, alt=None, fov=110, cfg="sm"):
    """Genera URL para PeakFinder con la posición y orientación dadas."""
    date_str = eclipse_time.iso.replace(' ', 'T') + 'Z'
    base_url = "https://www.peakfinder.com/es/?"
    params = f"lat={lat}&lng={lon}&ele={int(elev)}"
    if azi is not None:
        params += f"&azi={azi}"
    if alt is not None:
        params += f"&alt={alt}"
    params += f"&fov={fov}&date={date_str}&cfg={cfg}"
    return base_url + params


def get_shademap_url(lat, lon, elev, eclipse_time, zoom=13, bearing=0, pitch=0, margin=0):
    """Genera URL para ShadeMap con la posición y timestamp dados."""
    base_url = "https://shademap.app/@"
    lat_str = f"{lat:.3f}"
    lon_str = f"{lon:.3f}"
    ts_ms = int(eclipse_time.unix * 1000)
    return f"{base_url}{lat_str},{lon_str},{zoom}z,{ts_ms}t,{bearing}b,{pitch}p,{margin}m"

# Funciones de simulación y eventos (importar de secuencia)
def angular_diameter(radius, distance):
    return np.degrees(2 * np.arctan(radius / distance))


def aplicar_refraction(alt, az, distance, RefInst, lmbda=550e-9):
    z = np.deg2rad(90 - alt)
    shift = RefInst.get_AngularShift(z, lmbda)
    lateral = RefInst.get_LateralShift(z, lmbda)
    return alt + np.degrees(shift), az + np.degrees(np.arctan(lateral/(distance*1000)))


def obtener_parametros(t, location):
    """Obtiene parámetros del eclipse importados del módulo externo."""
    from secuencia_sol_luna8 import obtener_parametros as get_par
    return get_par(t, location)


def detectar_eventos(tiempos, params, location):
    """Detecta eventos del eclipse importados del módulo externo."""
    from secuencia_sol_luna8 import detectar_eventos as det_ev
    return det_ev(tiempos, params, location)

# Animaciones

def _create_axes(title, limite):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-limite, limite); ax.set_ylim(-limite, limite)
    ax.set_xlabel("Δ Azimut (°)"); ax.set_ylabel("Δ Altitud (°)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    sun = Circle((0,0),1, color='gold', alpha=0.6, ec='darkgoldenrod')
    moon = Circle((0,0),0.1, color='silver', alpha=0.6, ec='dimgray')
    ax.add_patch(sun); ax.add_patch(moon)
    time_txt = ax.text(0.05,0.95,'', transform=ax.transAxes, va='top')
    return fig, ax, sun, moon, time_txt


def simular_eclipse_streamlit(tiempos, location, horizon=None, use_refraction=False, limite=2.5, fps=5):
    """Genera una animación GIF de la simulación de eclipse, opcionalmente con refracción."""
    # Selección de título según tipo
    title = "Con refracción" if use_refraction else "Sin refracción"
    fig, ax, sun_patch, moon_patch, txt = _create_axes(title, limite)
    # Añadir línea de horizonte
    horizon_line, = ax.plot([], [], color='blue', linestyle='--', linewidth=1)
    RefInst = None
    lmbda = None
    if use_refraction:
        RefInst = refraction_shift(288.15, 101325, location.height.value)
        lmbda = 550e-9

    def update(i):
        t = tiempos[i]
        altaz = AltAz(obstime=t, location=location)
        sol = get_sun(t).transform_to(altaz)
        lun = get_body('moon', t, location=location).transform_to(altaz)
        d_sun = get_sun(t).distance.to(u.km).value
        d_lun = get_body('moon', t, location=location).distance.to(u.km).value
        r_sol = angular_diameter(R_SUN, d_sun)/2
        r_lun = angular_diameter(R_MOON, d_lun)/2
        sun_patch.set_radius(r_sol)
        # Cálculo posiciones con/sin refracción
        if use_refraction and RefInst:
            alt_s, az_s = aplicar_refraction(sol.alt.deg, sol.az.deg, d_sun, RefInst, lmbda)
            alt_m, az_m = aplicar_refraction(lun.alt.deg, lun.az.deg, d_lun, RefInst, lmbda)
        else:
            alt_s, az_s = sol.alt.deg, sol.az.deg
            alt_m, az_m = lun.alt.deg, lun.az.deg
        dx = (az_m - az_s)*np.cos(np.deg2rad(alt_s))
        dy = alt_m - alt_s
        moon_patch.center = (dx, dy)
        moon_patch.set_radius(r_lun)
        txt.set_text(t.iso)
        # Dibujar horizonte
        if horizon is not None:
            pts = []
            sol_az = az_s
            sol_alt = alt_s
            for az_h, alt_h in horizon:
                d_azp = ((az_h - sol_az + 180) % 360) - 180
                x_p = d_azp * np.cos(np.deg2rad(sol_alt))
                y_p = alt_h - sol_alt
                pts.append([x_p, y_p])
            pts = np.array(pts)
            horizon_line.set_data(pts[:,0], pts[:,1])
        else:
            # horizonte plano al nivel de 0 altitud
            y0 = - sol.alt.deg
            if -limite <= y0 <= limite:
                x_vals = np.array([-limite, limite])
                y_vals = np.array([y0, y0])
                horizon_line.set_data(x_vals, y_vals)
            else:
                horizon_line.set_data([], [])
        return sun_patch, moon_patch, txt, horizon_line

    ani = animation.FuncAnimation(fig, update, frames=len(tiempos), blit=True)
    tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    ani.save(tmp.name, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return tmp.name
    """Genera una animación GIF de la simulación de eclipse, opcionalmente con refracción."""
    # Selección de título según tipo
    title = "Con refracción" if use_refraction else "Sin refracción"
    fig, ax, sun_patch, moon_patch, txt = _create_axes(title, limite)
    RefInst = None
    lmbda = None
    if use_refraction:
        RefInst = refraction_shift(288.15, 101325, location.height.value)
        lmbda = 550e-9

    def update(i):
        t = tiempos[i]
        altaz = AltAz(obstime=t, location=location)
        sol = get_sun(t).transform_to(altaz)
        lun = get_body('moon', t, location=location).transform_to(altaz)
        d_sun = get_sun(t).distance.to(u.km).value
        d_lun = get_body('moon', t, location=location).distance.to(u.km).value
        r_sol = angular_diameter(R_SUN, d_sun)/2
        r_lun = angular_diameter(R_MOON, d_lun)/2
        sun_patch.set_radius(r_sol)
        if use_refraction and RefInst:
            alt_s, az_s = aplicar_refraction(sol.alt.deg, sol.az.deg, d_sun, RefInst, lmbda)
            alt_m, az_m = aplicar_refraction(lun.alt.deg, lun.az.deg, d_lun, RefInst, lmbda)
        else:
            alt_s, az_s = sol.alt.deg, sol.az.deg
            alt_m, az_m = lun.alt.deg, lun.az.deg
        dx = (az_m - az_s)*np.cos(np.deg2rad(alt_s))
        dy = alt_m - alt_s
        moon_patch.center=(dx, dy); moon_patch.set_radius(r_lun)
        txt.set_text(t.iso)
        return sun_patch, moon_patch, txt

        t = tiempos[i]
        altaz = AltAz(obstime=t, location=location)
        sol = get_sun(t).transform_to(altaz)
        lun = get_body('moon', t, location=location).transform_to(altaz)
        d_sun = get_sun(t).distance.to(u.km).value
        d_lun = get_body('moon', t, location=location).distance.to(u.km).value
        r_sol = angular_diameter(R_SUN, d_sun)/2
        r_lun = angular_diameter(R_MOON, d_lun)/2
        sun_patch.set_radius(r_sol)
        if refraction and RefInst:
            alt_s, az_s = aplicar_refraction(sol.alt.deg, sol.az.deg, d_sun, RefInst, lmbda)
            alt_m, az_m = aplicar_refraction(lun.alt.deg, lun.az.deg, d_lun, RefInst, lmbda)
        else:
            alt_s, az_s = sol.alt.deg, sol.az.deg
            alt_m, az_m = lun.alt.deg, lun.az.deg
        dx = (az_m - az_s)*np.cos(np.deg2rad(alt_s))
        dy = alt_m - alt_s
        moon_patch.center=(dx, dy); moon_patch.set_radius(r_lun)
        txt.set_text(t.iso)
        return sun_patch, moon_patch, txt

    ani = animation.FuncAnimation(fig, update, frames=len(tiempos), blit=True)
    tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    ani.save(tmp.name, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return tmp.name

# --- Configuración de la App ---
st.set_page_config(page_title="Simulador Eclipse Solar", layout="wide")
st.title("🌘 Simulador de Eclipse Solar")
st.markdown(
    "App para calcular eventos de un eclipse, generar animaciones y obtener catálogos de estrellas."
)

# -- Sidebar: Configuración básica --
st.sidebar.header("📅 Fecha y hora")
fecha = st.sidebar.date_input("Fecha del eclipse", datetime(2026,8,12))
hora = st.sidebar.time_input("Hora inicial (UT)", datetime.strptime("17:30","%H:%M").time())
dur_min = st.sidebar.slider("Duración (min)", 5, 240, 120)
eclipse_start = Time(datetime.combine(fecha, hora))

# -- Sidebar: Ubicación --
st.sidebar.header("📍 Ubicación")
if 'coords' not in st.session_state:
    st.session_state.coords = {'lat':40.4168,'lng':-3.7038}
metodo = st.sidebar.radio("Indicar ubicación via:", ("Municipio","Manual","Capitales"))
if metodo == "Municipio":
    lugar = st.sidebar.text_input("Municipio o lugar", "Madrid")
    if st.sidebar.button("Buscar"):
        lat, lng, addr = geocode_lugar(lugar)
        if lat and lng:
            st.session_state.coords = {'lat':lat,'lng':lng}
            st.sidebar.success(addr)
        else:
            st.sidebar.error("No encontrado")
elif metodo == "Manual":
    st.session_state.coords['lat'] = st.sidebar.number_input("Latitud", value=st.session_state.coords['lat'], format="%.6f")
    st.session_state.coords['lng'] = st.sidebar.number_input("Longitud", value=st.session_state.coords['lng'], format="%.6f")
else:
    ciudad = st.sidebar.selectbox("Capital española", list(capitales.keys()))
    lat, lng = capitales[ciudad]
    st.session_state.coords = {'lat':lat,'lng':lng}
    st.sidebar.write(reverse_geocode(lat,lng))

lat = st.session_state.coords['lat']
lng = st.session_state.coords['lng']
st.write(f"**Coordenadas**: {lat:.5f}°, {lng:.5f}°")

# Mapa interactivo
m = folium.Map(location=[lat,lng], zoom_start=6)
folium.Marker([lat,lng], tooltip="Click en el mapa para seleccionar ubicación").add_to(m)
map_data = st_folium(m, width=700, height=300)
if map_data and map_data.get("last_clicked"):
    lat_new = map_data["last_clicked"]["lat"]
    lng_new = map_data["last_clicked"]["lng"]
    st.session_state.coords = {'lat': lat_new, 'lng': lng_new}
    lat, lng = lat_new, lng_new
    st.success(f"Coordenadas seleccionadas: {lat:.5f}°, {lng:.5f}°")
    # Redibujar marcador en nueva ubicación
    m2 = folium.Map(location=[lat,lng], zoom_start=6)
    folium.Marker([lat,lng], tooltip="Ubicación seleccionada").add_to(m2)
    st_folium(m2, width=700, height=300)

# Altitud
elev = obtener_altitud(lat,lng) or 0
st.write(f"**Elevación**: {elev} m")
elev = st.number_input("Altitud (m)", int(elev))
location = EarthLocation(lat=lat*u.deg, lon=lng*u.deg, height=elev*u.m)

# Horizon file
horizon_data = None
up = st.file_uploader("Fichero horizonte (az alt)", type=["txt","csv"])
if up:
    try:
        horizon_data = np.loadtxt(up, skiprows=1)
        st.success("Horizonte cargado")
    except:
        st.error("Error al cargar horizonte")

# Descargar catálogo
stars_flag = st.checkbox("Descargar catálogo de estrellas (Gmag<6.5)")

# Botones principales
envent_df = None
if st.button("🔍 Calcular eventos"):
    pasos = int(dur_min*60/10)+1
    times = eclipse_start + TimeDelta(np.arange(pasos)*10, format='sec')
    # No cache para obtener parámetros y eventos en cada clic
    params = [obtener_parametros(t, location) for t in times]
    eventos = detectar_eventos(times, params, location)
    # Mostrar eventos en orden definido
    order = [
        'Primer Contacto', 'Segundo Contacto', 'Máximo Eclipse',
        'Tercer Contacto', 'Cuarto Contacto', 'Puesta de Sol'
    ]
    rows = []
    for key in order:
        if key in eventos:
            e = eventos[key]
            row = {'Evento': key, 'UT': e['time'].iso}
            if 'alt_sol' in e:
                row.update({'Alt': round(e['alt_sol'],2), 'Az': round(e['az_sol'],2), 'Mag': round(e['mag'],3)})
            rows.append(row)
    envent_df = pd.DataFrame(rows)
    st.dataframe(envent_df)
    # Guardar tiempo máximo para otros usos
    if 'Máximo Eclipse' in eventos:
        st.session_state.t_max = eventos['Máximo Eclipse']['time']

# Descargar catálogo si se solicita
if stars_flag and 't_max' in st.session_state:
    df_cat = descargar_catalogo_estrellas(get_sun(st.session_state.t_max))
    if df_cat is not None and not df_cat.empty:
        st.download_button(
            "Descargar CSV catálogo estrellas",
            df_cat.to_csv(index=False),
            file_name="catalogo_estrellas.csv",
            mime="text/csv"
        )
    else:
        st.warning("No se encontraron estrellas en el catálogo.")

# Simulación sin refracción
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Simulación sin refracción"):
        pasos = int(dur_min*60/10)+1
        times = eclipse_start + TimeDelta(np.arange(pasos)*10, format='sec')
        gif_path = simular_eclipse_streamlit(
            times,
            location,
            horizon=horizon_data,
            use_refraction=False
        )
        st.image(gif_path, use_container_width=True)

# Simulación con refracción (aviso)
st.warning(
    "La simulación con refracción completa tarda mucho y está disponible para cálculo local en https://github.com/pmisson/pyeclipsesimulator"
)



# Enlaces externos: PeakFinder y ShadeMap
st.markdown("---")
azi=282
# Generar URLs base
base_t = st.session_state.get('t_max', eclipse_start)
url_pf = get_peakfinder_url(lat, lng, elev, base_t,azi)
url_sm = get_shademap_url(lat, lng, elev, base_t)

# Botones HTML que abren nueva pestaña
st.sidebar.markdown(
    f'<a href="{url_pf}" target="_blank"><button style="width:100%;padding:8px">▶️ Abrir PeakFinder</button></a>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f'<a href="{url_sm}" target="_blank"><button style="width:100%;padding:8px">▶️ Abrir ShadeMap</button></a>',
    unsafe_allow_html=True
)

# Agradecimientos y logos
st.markdown("---")
st.markdown("### Agradecimientos")
st.markdown(
    "Esta aplicación se ha desarrollado gracias al apoyo de:"
)
st.markdown(
    "- **Beca UNA4CAREER UCM.es** ([UNA4CAREER](https://www.una4career.eu/) | [GUAIX UCM](https://guaix.fis.ucm.es/))"
)
st.markdown(
    "- **Beca EMERGIA**: Financiado por EMERGIA20_DGP_EMEC_2023_00431, Consejería de Universidad, Investigación e Innovación de la Junta de Andalucía, en el [IAA-CSIC](https://www.iaa.csic.es/)."
)
st.markdown(
    "- Financiado por el Programa de Investigación e Innovación Horizonte 2020 de la Unión Europea, Marie Sklodowska-Curie grant No. 847635."
)
col1, col2, col3, col4 = st.columns(4)
col1.image("https://www.juntadeandalucia.es/sites/default/files/2023-02/normal_1.jpg", caption="Junta de Andalucía", width=100)
col2.image("https://www.ucm.es/data/cont/docs/3-2016-07-21-Marca%20UCM%20logo%20negro.png", caption="UCM", width=100)
col3.image("https://somma.es/wp-content/uploads/2022/04/IAA-CSIC.png", caption="IAA-CSIC", width=100)
col4.image("https://www.uib.no/sites/w3.uib.no/files/styles/content_main/public/media/marie_curie_logo.png?itok=zR0htrxL", caption="Marie Curie", width=100)




