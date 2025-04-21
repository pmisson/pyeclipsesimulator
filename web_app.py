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

# --- Configuración de la página (debe ser primera llamada) ---
st.set_page_config(page_title="Solar Eclipse Simulator", layout="wide")

# Inicializar coordenadas en session_state si no existen
if 'coords' not in st.session_state:
    # Valor por defecto: Madrid
    st.session_state.coords = {'lat': 40.4168, 'lng': -3.7038}

# --- Textos de la interfaz para dos idiomas ---
TEXT = {
    'title': {'es':'🌘 Simulador de Eclipse Solar','en':'🌘 Solar Eclipse Simulator'},
    'subtitle': {'es':'App para calcular eventos de un eclipse, generar animaciones y descargar catálogos de estrellas.',
                 'en':'App to calculate eclipse events, create animations, and download star catalogs.'},
    'sidebar_date_header':{'es':'📅 Fecha y hora','en':'📅 Date & Time'},
    'sidebar_date':{'es':'Fecha del eclipse','en':'Eclipse date'},
    'sidebar_time':{'es':'Hora inicial (UT)','en':'Start time (UT)'},
    'sidebar_duration':{'es':'Duración (min)','en':'Duration (min)'},
    'sidebar_location_header':{'es':'📍 Ubicación','en':'📍 Location'},
    'location_method':{'es':'Indicar ubicación vía:','en':'Specify location via:'},
    'method_municipio':{'es':'Municipio','en':'Place name'},
    'method_manual':{'es':'Manual','en':'Manual'},
    'method_capitales':{'es':'Capitales','en':'Capitals'},
    'input_place':{'es':'Municipio o lugar','en':'Place name'},
    'button_search':{'es':'Buscar','en':'Search'},
    'error_not_found':{'es':'No encontrado','en':'Not found'},
    'input_lat':{'es':'Latitud','en':'Latitude'},
    'input_lng':{'es':'Longitud','en':'Longitude'},
    'coordinates':{'es':'Coordenadas','en':'Coordinates'},
    'altitude_label':{'es':'Altitud (m)','en':'Elevation (m)'},
    'horizon_upload':{'es':'Fichero horizonte (az alt)','en':'Horizon file (az alt)'},
    'horizon_success':{'es':'Horizonte cargado','en':'Horizon loaded'},
    'horizon_error':{'es':'Error al cargar horizonte','en':'Error loading horizon'},
    'download_stars':{'es':'Descargar catálogo de estrellas (Gmag<6.5)','en':'Download star catalog (Gmag<6.5)'},
    'calculate_events':{'es':'🔍 Calcular eventos','en':'🔍 Calculate events'},
    'download_csv_stars':{'es':'Descargar CSV catálogo estrellas','en':'Download stars CSV'},
    'warning_no_stars':{'es':'No se encontraron estrellas en el catálogo.','en':'No stars found in catalog.'},
    'sim_no_ref':{'es':'▶️ Simulación sin refracción','en':'▶️ Sim without refraction'},
    'sim_horizon':{'es':'Simulación de horizonte (PeakFinder)','en':'Horizon simulation (PeakFinder)'},
    'sim_shades':{'es':'Mapa de sombras (ShadeMap)','en':'Shadow map (ShadeMap)'},
    'ack_header':{'es':'### Agradecimientos','en':'### Acknowledgements'},
    'ack_text':{'es':'Esta aplicación se ha desarrollado gracias al apoyo de:','en':'This application was developed with support from:'},
    'interactive_map':{'es':'Mapa interactivo','en':'Interactive map'},
    'event_table':{'es':'Eventos del eclipse','en':'Eclipse events'},
    'interactive_map':{'es':'Mapa interactivo','en':'Interactive map'}
}

# --- Selector de idioma y función de traducción ---
lang = st.sidebar.radio("Idioma / Language", ("Español", "English"), index=0)
def t(key):
    return TEXT[key]['en'] if lang == 'English' else TEXT[key]['es']

# --- Constantes físicas ---
R_SUN = 696340.0  # km
R_MOON = 1737.4   # km

# --- Capitales de España para selección rápida ---
capitales = {
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3851, 2.1734),
    "Valencia": (39.4699, -0.3763),
    "Sevilla": (37.3891, -5.9845),
    "Alcalá de Henares": (40.4818, -3.3635),
    "Alcobendas": (40.5475, -3.6419),
    "Avilés": (43.5569, -5.9248),
    "Benavente": (42.0028, -5.6789),
    "Bilbao": (43.2630, -2.9340),
    "Burgos": (42.3439, -3.6969),
    "Castellón de la Plana": (39.9864, -0.0369)
}

# --- Funciones auxiliares ---
@st.cache_data(ttl=86400)
def obtener_altitud(lat, lng):
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

def descargar_catalogo_estrellas(center_coord, radius=2.5, mag_limite=6.5):
    Vizier.ROW_LIMIT = -1
    from astropy.coordinates import SkyCoord
    center = SkyCoord(ra=center_coord.ra, dec=center_coord.dec, unit='deg', frame='icrs')
    viz = Vizier(columns=["RA_ICRS","DE_ICRS","Gmag"], row_limit=-1)
    try:
        res = viz.query_region(center, radius=radius*u.deg, catalog="I/355/gaiadr3")
        if res:
            df = res[0].to_pandas()
            return df[df.Gmag < mag_limite]
    except:
        pass
    return None

# Geocodificación de lugares
@st.cache_data(ttl=86400)
def geocode_lugar(nombre):
    loc = Nominatim(user_agent='eclipse_app').geocode(nombre, language='es')
    return (loc.latitude, loc.longitude) if loc else (None, None)

# URLs externas

def get_peakfinder_url(lat, lon, elev, eclipse_time, azi=None, alt=None, fov=110, cfg="sm"):
    date_str = eclipse_time.iso.replace(' ', 'T') + 'Z'
    base = "https://www.peakfinder.com/es/?"
    params = f"lat={lat}&lng={lon}&ele={int(elev)}&fov={fov}&date={date_str}&cfg={cfg}"
    if azi: params += f"&azi={azi}"
    if alt: params += f"&alt={alt}"
    return base + params

def get_shademap_url(lat, lon, elev, eclipse_time, zoom=13, bearing=0, pitch=0, margin=0):
    ts_ms = int(eclipse_time.unix * 1000)
    return f"https://shademap.app/@{lat:.3f},{lon:.3f},{zoom}z,{ts_ms}t,{bearing}b,{pitch}p,{margin}m"

# Importar funciones de secuencia (cálculo de eventos y parámetros)
from secuencia_sol_luna8 import (
    obtener_parametros,
    detectar_eventos,
    angular_diameter,
    aplicar_refraction
)

# Crear ejes para animación
def _create_axes(title, limite):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.set_xlabel("Δ Azimut (°)")
    ax.set_ylabel("Δ Altitud (°)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    sun = Circle((0,0),1, color='gold', alpha=0.6, ec='darkgoldenrod')
    moon = Circle((0,0),0.1, color='silver', alpha=0.6, ec='dimgray')
    ax.add_patch(sun)
    ax.add_patch(moon)
    time_txt = ax.text(0.05,0.95,'', transform=ax.transAxes, va='top')
    horizon_line, = ax.plot([], [], color='blue', linestyle='--', linewidth=1)
    return fig, ax, sun, moon, time_txt, horizon_line

# Simulación GIF

def simular_eclipse_streamlit(tiempos, location, horizon=None, use_refraction=False, limite=2.5, fps=5):
    title = t('sim_no_ref') if not use_refraction else t('sim_shades')
    fig, ax, sun_patch, moon_patch, txt, horizon_line = _create_axes(title, limite)
    RefInst = None;
    if use_refraction:
        RefInst = refraction_shift(288.15,101325,location.height.value)
        lmbda = 550e-9
    
    def update(i):
        t0 = tiempos[i]
        altaz = AltAz(obstime=t0, location=location)
        sol = get_sun(t0).transform_to(altaz)
        lun = get_body('moon', t0, location=location).transform_to(altaz)
        d_sun = get_sun(t0).distance.to(u.km).value
        d_lun = lun.distance.to(u.km).value
        r_sol = angular_diameter(R_SUN, d_sun)/2
        r_lun = angular_diameter(R_MOON, d_lun)/2
        sun_patch.set_radius(r_sol)
        if use_refraction and RefInst:
            alt_s, az_s = aplicar_refraction(sol.alt.deg, sol.az.deg, d_sun, RefInst, lmbda)
            alt_m, az_m = aplicar_refraction(lun.alt.deg, lun.az.deg, d_lun, RefInst, lmbda)
        else:
            alt_s, az_s = sol.alt.deg, sol.az.deg
            alt_m, az_m = lun.alt.deg, lun.az.deg
        dx = (az_m - az_s)*np.cos(np.deg2rad(alt_s)); dy = alt_m - alt_s
        moon_patch.center=(dx,dy); moon_patch.set_radius(r_lun)
        txt.set_text(t0.iso)
        # Horizonte real o plano
        if horizon is not None:
            pts=[]
            for az_h, alt_h in horizon:
                d_azp = ((az_h-az_s+180)%360)-180
                x_p = d_azp*np.cos(np.deg2rad(alt_s)); y_p = alt_h-alt_s
                pts.append([x_p,y_p])
            pts=np.array(pts); horizon_line.set_data(pts[:,0],pts[:,1])
        else:
            y0 = -alt_s
            if -limite<=y0<=limite:
                horizon_line.set_data([-limite,limite],[y0,y0])
            else:
                horizon_line.set_data([],[])
        return sun_patch, moon_patch, txt, horizon_line

    ani = animation.FuncAnimation(fig, update, frames=len(tiempos), blit=True)
    tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    ani.save(tmp.name, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return tmp.name

# --- Renderizado UI ---
# Encabezado
st.title(t('title'))
st.markdown(t('subtitle'))

# Sidebar: Fecha, hora, duración
st.sidebar.markdown(":alarm_clock: _Todas las horas están en Tiempo Universal (UTC). Los enlaces usarán UTC para el cálculo de vistas._")
fecha = st.sidebar.date_input(t('sidebar_date'), datetime(2026,8,12))
hora = st.sidebar.time_input(t('sidebar_time'), datetime.strptime('17:30','%H:%M').time())
dur_min = st.sidebar.slider(t('sidebar_duration'),5,240,120)
eclipse_start = Time(datetime.combine(fecha,hora), scale='utc')
# Para mostrar hora local también:
try:
    local_dt = eclipse_start.to_datetime().astimezone()
    st.sidebar.write(f"Hora local: {local_dt.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception:
    pass

# Sidebar: Ubicación
st.sidebar.header(t('sidebar_location_header'))
metodo = st.sidebar.radio(t('location_method'), (t('method_municipio'), t('method_manual'), t('method_capitales')))
if metodo == t('method_municipio'):
    lugar = st.sidebar.text_input(t('input_place'), 'Madrid')
    if st.sidebar.button(t('button_search')):
        lat,lng = geocode_lugar(lugar)
        if lat and lng: st.session_state.coords={'lat':lat,'lng':lng}
        else: st.sidebar.error(t('error_not_found'))
elif metodo == t('method_manual'):
    lat = st.sidebar.number_input(t('input_lat'), value=st.session_state.coords.get('lat',40.4168))
    lng = st.sidebar.number_input(t('input_lng'), value=st.session_state.coords.get('lng',-3.7038))
    st.session_state.coords={'lat':lat,'lng':lng}
else:
    ciudad = st.sidebar.selectbox(t('method_capitales'), list(capitales.keys()))
    lat,lng = capitales[ciudad]; st.session_state.coords={'lat':lat,'lng':lng}

# Mostrar coordenadas
st.write(f"{t('coordinates')}: {st.session_state.coords['lat']:.5f}°, {st.session_state.coords['lng']:.5f}°")

# Altitud
alt0 = obtener_altitud(**st.session_state.coords) or 0
elev = st.number_input(t('altitude_label'), value=int(alt0))
location = EarthLocation(lat=st.session_state.coords['lat']*u.deg,
                          lon=st.session_state.coords['lng']*u.deg,
                          height=elev*u.m)

# Horizonte
up = st.file_uploader(t('horizon_upload'), type=['txt','csv'])
if up:
    try:
        horizon_data = np.loadtxt(up, skiprows=1)
        st.success(t('horizon_success'))
    except:
        st.error(t('horizon_error'))
else:
    horizon_data = None

# Mapa interactivo
st.subheader(t('interactive_map'))
m = folium.Map(location=[st.session_state.coords['lat'], st.session_state.coords['lng']], zoom_start=6)
folium.Marker([st.session_state.coords['lat'], st.session_state.coords['lng']], tooltip=t('coordinates')).add_to(m)
map_data = st_folium(m, width=700, height=400)
if map_data and map_data.get('last_clicked'):
    lat_new = map_data['last_clicked']['lat']
    lng_new = map_data['last_clicked']['lng']
    st.session_state.coords = {'lat': lat_new, 'lng': lng_new}
    # Actualizar variable location tras cambiar coords
    location = EarthLocation(lat=lat_new*u.deg,
                              lon=lng_new*u.deg,
                              height=elev*u.m)

# Checkbox para descargar catálogo de estrellas (Gmag<6.5)
stars_flag = st.checkbox(t('download_stars'))

# Botón calcular eventos
if st.button(t('calculate_events')):
    pasos = int(dur_min * 60 / 10) + 1
    times = eclipse_start + TimeDelta(np.arange(pasos) * 10, format='sec')
    params = [obtener_parametros(t0, location) for t0 in times]
    eventos = detectar_eventos(times, params, location)
    # Mostrar tabla de eventos
    rows = []
    for nombre in ['Primer Contacto','Segundo Contacto','Máximo Eclipse','Tercer Contacto','Cuarto Contacto','Puesta de Sol']:
        if nombre in eventos:
            ev = eventos[nombre]
            row = {'Evento': nombre, 'UT': ev['time'].iso}
            if 'alt_sol' in ev:
                row.update({'Alt': round(ev['alt_sol'],2), 'Az': round(ev['az_sol'],2), 'Mag': round(ev['mag'],3)})
            rows.append(row)
    df_events = pd.DataFrame(rows)
    st.markdown(t('event_table'))
    st.dataframe(df_events)
    if 'Máximo Eclipse' in eventos:
        st.session_state.t_max = eventos['Máximo Eclipse']['time']

# Descargar CSV catálogo
if stars_flag and 't_max' in st.session_state:
    df_cat = descargar_catalogo_estrellas(get_sun(st.session_state.t_max))
    if df_cat is not None and not df_cat.empty:
        st.download_button(
            t('download_csv_stars'),
            df_cat.to_csv(index=False),
            file_name='stars.csv'
        )
    else:
        st.warning(t('warning_no_stars'))

# Simulación sin refracción
if st.button(t('sim_no_ref')):
    pasos = int(dur_min * 60 / 10) + 1
    times = eclipse_start + TimeDelta(np.arange(pasos) * 10, format='sec')
    gif = simular_eclipse_streamlit(
        times,
        location,
        horizon=horizon_data,
        use_refraction=False
    )
    st.image(gif, use_container_width=True)

# Enlaces externos: Horizon & Shadow map
col1, col2 = st.sidebar.columns(2)
with col1:
    pf_url = get_peakfinder_url(
        st.session_state.coords['lat'],
        st.session_state.coords['lng'],
        elev,
        eclipse_start.utc  # Aseguramos UTC para URL
    )
    st.sidebar.markdown(
        f'<a href="{pf_url}" target="_blank"><button>{t("sim_horizon")}</button></a>',
        unsafe_allow_html=True
    )
with col2:
    sm_url = get_shademap_url(
        st.session_state.coords['lat'],
        st.session_state.coords['lng'],
        elev,
        eclipse_start.utc  # Aseguramos UTC para URL
    )
    st.sidebar.markdown(
        f'<a href="{sm_url}" target="_blank"><button>{t("sim_shades")}</button></a>',
        unsafe_allow_html=True
    )

# Agradecimientos y logos
st.markdown(t('ack_header'))
st.markdown(t('ack_text'))
col1, col2, col3, col4 = st.columns(4)
col1.image(
    "https://www.juntadeandalucia.es/sites/default/files/2023-02/normal_1.jpg",
    caption="Junta de Andalucía",
    width=150
)
col2.image(
    "https://www.ucm.es/data/cont/docs/3-2016-07-21-Marca%20UCM%20logo%20negro.png",
    caption="UCM",
    width=150
)
col3.image(
    "https://somma.es/wp-content/uploads/2022/04/IAA-CSIC.png",
    caption="IAA-CSIC",
    width=150
)
col4.image(
    "https://www.uib.no/sites/w3.uib.no/files/styles/content_main/public/media/marie_curie_logo.png?itok=zR0htrxL",
    caption="Marie Curie",
    width=150
)


