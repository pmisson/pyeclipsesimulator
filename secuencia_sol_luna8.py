#!/usr/bin/env python3
"""
Script integrado para:
  1. Calcular y tabular eventos relevantes en un eclipse solar.
  2. Permitir cargar un fichero opcional con datos reales del horizonte.
  3. Permitir la selección de la localización mediante tres opciones:
       a) Nombre del municipio (forward geocoding),
       b) Ingresar coordenadas manualmente,
       c) Seleccionar de una lista de capitales de provincia (reverse geocoding).
  4. Obtener la elevación automáticamente usando la API de Open‑Meteo si se deja en blanco.
  5. Cargar o descargar automáticamente un catálogo de estrellas (hasta magnitud 6.5) mediante Astroquery (Vizier).
  6. Mostrar dos simulaciones (dual: sin refracción y con refracción) o bien una simulación individual (sin refracción),
     sobreponiendo las estrellas durante la totalidad.
  7. Ofrecer la opción de abrir el perfil de PeakFinder, configurado con la orientación de la Luna y la hora del eclipse.
  8. También usar ShadeMap para visualizar la zona de sombras con zoom 13 por defecto, según el formato:
       https://shademap.app/@<lat>,<lon>,<zoom>z,<timestamp>t,<bearing>b,<pitch>p,<margin>m
  9. Calcular la duración precisa del eclipse (total y de la totalidad, si existe).

Requiere:
  - astropy, matplotlib, numpy
  - geopy (pip install geopy)
  - requests (pip install requests)
  - astroquery (pip install astroquery)
  - RefractionShift (pip install RefractionShift)
  - AstroAtmosphere (pip install AstroAtmosphere)
"""

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body, SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse
import sys, webbrowser, os, requests
from geopy.geocoders import Nominatim
import time as pytime
import pandas as pd

from astroquery.vizier import Vizier
from RefractionShift.refraction_shift import refraction

# --- Constantes físicas ---
R_SUN = 696340.0         # km
R_MOON = 1737.4          # km

# --- Lista de capitales de provincia en España ---
capitales = {
    "Alcalá de Henares": (40.48198, -3.36354),
    "Alcobendas": (40.54746, -3.64197),
    "Avilés": (43.55694, -5.92483),
    "Benavente": (42.00282, -5.67889),
    "Bilbao": (43.263, -2.934),
    "Burgos": (42.34399, -3.69691),
    "Castellón de la Plana": (39.98641, -0.03688),
    "Cuenca": (40.07039, -2.13742),
    "Ferrol": (43.48333, -8.23333),
    "Gijón": (43.5357, -5.6615),
    "Guadalajara": (40.63333, -3.16667),
    "Ibiza": (38.90883, 1.43296),
    "A Coruña": (43.3713, -8.396),
    "León": (42.59873, -5.5671),
    "Lleida": (41.61759, 0.62001),
    "Logroño": (42.46645, -2.44566),
    "Lugo": (43.00913, -7.55817),
    "Menorca": (39.94961, 4.11045),
    "Oviedo": (43.3603, -5.8448),
    "Palencia": (42.00955, -4.52717),
    "Palma": (39.5696, 2.6502),
    "Portugalete": (43.32082, -3.02064),
    "Reus": (41.15612, 1.10687),
    "San Sebastián de los Reyes": (40.54539, -3.62793),
    "Santander": (43.46298, -3.80475),
    "Barakaldo": (43.29564, -2.99735),
    "Segovia": (40.94808, -4.11839),
    "Soria": (41.76328, -2.46625),
    "Tarragona": (41.1189, 1.2445),
    "Torrejón de Ardoz": (40.45676, -3.4755),
    "Torrent": (39.43701, -0.46536),
    "Valencia": (39.4699, -0.3763),
    "Valladolid": (41.6528, -4.7245),
    "Vitoria-Gasteiz": (42.8467, -2.6716),
    "Zamora": (41.50332, -5.74456),
    "Zaragoza": (41.6488, -0.8891)
}

# --- Funciones auxiliares de geolocalización, catálogo, horizonte, etc. ---
def obtener_altitud(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "elevation" in data and isinstance(data["elevation"], list) and len(data["elevation"]) > 0:
                return data["elevation"][0]
        return None
    except Exception:
        return None

def geocode_lugar(nombre):
    geolocator = Nominatim(user_agent="eclipse_app")
    try:
        location = geolocator.geocode(nombre, language="es")
        if location is not None:
            return (location.latitude, location.longitude, location.address)
        else:
            return (None, None, None)
    except Exception:
        return (None, None, None)

def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="eclipse_app")
    try:
        location = geolocator.reverse((lat, lon), language="es")
        if location is not None:
            return location.address
        else:
            return "Desconocido"
    except Exception:
        return "Error"

def descargar_catalogo_estrellas(center_coord, radius=2.5, mag_limite=10):
    Vizier.ROW_LIMIT = -1
    center_coord = SkyCoord(ra=center_coord.ra, dec=center_coord.dec, unit='deg', frame='icrs')
    columns = ["RA_ICRS", "DE_ICRS", "Gmag"]
    vizier = Vizier(columns=columns, row_limit=-1)
    try:
        result = vizier.query_region(center_coord, radius=radius*u.deg, catalog="I/355/gaiadr3")
    except Exception as e:
        print("Error al consultar Vizier:", e)
        return None
    if result:
        stars = result[0].to_pandas()
        stars_filtered = stars[stars["Gmag"] < mag_limite]
        return stars_filtered[["RA_ICRS", "DE_ICRS", "Gmag"]].to_numpy()
    else:
        print("No se encontraron estrellas en el campo seleccionado.")
        return None

def cargar_estrellas(filename):
    if not os.path.isfile(filename):
        print(f"El fichero de estrellas '{filename}' no existe.")
        return None
    try:
        data = np.loadtxt(filename, skiprows=1)
        if data.ndim == 1:
            data = data.reshape((1, -1))
        return data
    except Exception as e:
        print(f"Error al cargar el fichero de estrellas: {e}")
        return None

def cargar_horizonte(filename):
    if not os.path.isfile(filename):
        print(f"El fichero '{filename}' no existe.")
        return None
    try:
        data = np.loadtxt(filename, skiprows=1)
        if data.ndim == 1:
            data = data.reshape((1, -1))
        return data
    except Exception as e:
        print(f"Error al cargar el fichero: {e}")
        return None

def aplicar_refraction(alt, az, distance, Refraction, lmbda=550e-9):
    z = np.deg2rad(90 - alt)
    shift_rad = Refraction.get_AngularShift(z, lmbda)
    shift_deg = np.degrees(shift_rad)
    lateral_m = Refraction.get_LateralShift(z, lmbda)
    lateral_ang = np.degrees(np.arctan(lateral_m / (distance * 1000)))
    alt_corr = alt + shift_deg
    az_corr = az + lateral_ang
    return alt_corr, az_corr

def angular_diameter(radius, distance):
    diam_rad = 2 * np.arctan(radius / distance)
    return np.degrees(diam_rad)

def calcular_contact_angle(obstime, location):
    altaz_frame = AltAz(obstime=obstime, location=location)
    sol = get_sun(obstime).transform_to(altaz_frame)
    luna = get_body("moon", obstime, location=location).transform_to(altaz_frame)
    d_az = (luna.az - sol.az).to(u.deg).value
    d_alt = (luna.alt - sol.alt).to(u.deg).value
    d_x = d_az * np.cos(sol.alt.radian)
    d_y = d_alt
    return np.degrees(np.arctan2(d_y, d_x))

def calcular_ratio_moon_sun(obstime, location):
    altaz_frame = AltAz(obstime=obstime, location=location)
    sol = get_sun(obstime)
    d_sun = sol.distance.to(u.km).value
    sol_diam = angular_diameter(R_SUN, d_sun)
    luna = get_body("moon", obstime, location=location).transform_to(altaz_frame)
    d_luna = luna.distance.to(u.km).value
    luna_diam = angular_diameter(R_MOON, d_luna)
    return luna_diam / sol_diam, luna_diam, sol_diam

def obtener_parametros(obstime, location):
    sol = get_sun(obstime)
    d_sun = sol.distance.to(u.km).value
    altaz_frame = AltAz(obstime=obstime, location=location)
    sol_trans = sol.transform_to(altaz_frame)
    luna = get_body("moon", obstime, location=location).transform_to(altaz_frame)
    
    sol_diam = angular_diameter(R_SUN, d_sun)
    r_sol = sol_diam / 2.0

    d_luna = luna.distance.to(u.km).value
    luna_diam = angular_diameter(R_MOON, d_luna)
    r_luna = luna_diam / 2.0

    d_az = (luna.az - sol_trans.az).to(u.deg).value
    d_alt = (luna.alt - sol_trans.alt).to(u.deg).value
    d_x = d_az * np.cos(sol_trans.alt.radian)
    d_y = d_alt
    d = np.sqrt(d_x**2 + d_y**2)

    alt_sol = sol_trans.alt.to(u.deg).value
    az_sol = sol_trans.az.to(u.deg).value

    if d < (r_sol + r_luna):
        mag = ((r_sol + r_luna) - d) / (2 * r_sol)
        if r_luna > r_sol and d <= (r_luna - r_sol):
            mag = 1.0
    else:
        mag = 0.0

    return {
        'time': obstime,
        'alt_sol': alt_sol,
        'az_sol': az_sol,
        'r_sol': r_sol,
        'r_luna': r_luna,
        'd': d,
        'mag': mag
    }

def linear_interpolate(t1, t2, v1, v2, target):
    if v2 == v1:
        return t1
    frac = (target - v1) / (v2 - v1)
    dt = (t2 - t1).sec
    return t1 + TimeDelta(frac * dt, format='sec')

def detectar_eventos(time_array, params_array, location):
    eventos = {}
    N = len(time_array)
    T_ext = np.array([p['r_sol'] + p['r_luna'] for p in params_array])
    T_int = np.array([p['r_luna'] - p['r_sol'] for p in params_array])
    d_vals = np.array([p['d'] for p in params_array])
    alt_sol_vals = np.array([p['alt_sol'] for p in params_array])
    az_sol_vals = np.array([p['az_sol'] for p in params_array])
    
    # Primer y Cuarto Contacto (eclipse parcial)
    for i in range(N-1):
        if d_vals[i] > T_ext[i] and d_vals[i+1] <= T_ext[i+1]:
            t_contact = linear_interpolate(time_array[i], time_array[i+1], d_vals[i], d_vals[i+1], T_ext[i])
            elapsed = (t_contact - time_array[0]).sec
            total = (time_array[-1] - time_array[0]).sec
            alt_interp = np.interp(elapsed, [0, total], [alt_sol_vals[0], alt_sol_vals[-1]])
            az_interp = np.interp(elapsed, [0, total], [az_sol_vals[0], az_sol_vals[-1]])
            eventos['Primer Contacto'] = {
                'time': t_contact,
                'alt_sol': alt_interp,
                'az_sol': az_interp,
                'd': T_ext[i],
                'mag': np.interp(elapsed, [0, total], [params_array[i]['mag'], params_array[i+1]['mag']])
            }
            break

    for i in range(N-1):
        if d_vals[i] <= T_ext[i] and d_vals[i+1] > T_ext[i+1]:
            t_contact = linear_interpolate(time_array[i], time_array[i+1], d_vals[i], d_vals[i+1], T_ext[i])
            elapsed = (t_contact - time_array[0]).sec
            total = (time_array[-1] - time_array[0]).sec
            alt_interp = np.interp(elapsed, [0, total], [alt_sol_vals[0], alt_sol_vals[-1]])
            az_interp = np.interp(elapsed, [0, total], [az_sol_vals[0], az_sol_vals[-1]])
            eventos['Cuarto Contacto'] = {
                'time': t_contact,
                'alt_sol': alt_interp,
                'az_sol': az_interp,
                'd': T_ext[i],
                'mag': np.interp(elapsed, [0, total], [params_array[i]['mag'], params_array[i+1]['mag']])
            }
            break

    # Segundo y Tercer Contacto (eclipse total)
    if any([p['r_luna'] > p['r_sol'] for p in params_array]):
        for i in range(N-1):
            if d_vals[i] > T_int[i] and d_vals[i+1] <= T_int[i+1]:
                t_contact = linear_interpolate(time_array[i], time_array[i+1], d_vals[i], d_vals[i+1], T_int[i])
                elapsed = (t_contact - time_array[0]).sec
                total = (time_array[-1] - time_array[0]).sec
                alt_interp = np.interp(elapsed, [0, total], [alt_sol_vals[0], alt_sol_vals[-1]])
                az_interp = np.interp(elapsed, [0, total], [az_sol_vals[0], az_sol_vals[-1]])
                angle_contact = calcular_contact_angle(t_contact, location)
                eventos['Segundo Contacto'] = {
                    'time': t_contact,
                    'alt_sol': alt_interp,
                    'az_sol': az_interp,
                    'd': T_int[i],
                    'mag': 1.0,
                    'angle': angle_contact
                }
                break
        for i in range(N-1):
            if d_vals[i] <= T_int[i] and d_vals[i+1] > T_int[i+1]:
                t_contact = linear_interpolate(time_array[i], time_array[i+1], d_vals[i], d_vals[i+1], T_int[i])
                elapsed = (t_contact - time_array[0]).sec
                total = (time_array[-1] - time_array[0]).sec
                alt_interp = np.interp(elapsed, [0, total], [alt_sol_vals[0], alt_sol_vals[-1]])
                az_interp = np.interp(elapsed, [0, total], [az_sol_vals[0], az_sol_vals[-1]])
                angle_contact = calcular_contact_angle(t_contact, location)
                eventos['Tercer Contacto'] = {
                    'time': t_contact,
                    'alt_sol': alt_interp,
                    'az_sol': az_interp,
                    'd': T_int[i],
                    'mag': 1.0,
                    'angle': angle_contact
                }
                break

    # Máximo Eclipse
    i_min = np.argmin(d_vals)
    eventos['Máximo Eclipse'] = params_array[i_min]

    # Puesta de Sol
    for i in range(N-1):
        if alt_sol_vals[i] > 0 and alt_sol_vals[i+1] <= 0:
            t_sunset = linear_interpolate(time_array[i], time_array[i+1], alt_sol_vals[i], alt_sol_vals[i+1], 0)
            elapsed = (t_sunset - time_array[0]).sec
            total = (time_array[-1] - time_array[0]).sec
            az_interp = np.interp(elapsed, [0, total], [az_sol_vals[0], az_sol_vals[-1]])
            eventos['Puesta de Sol'] = {
                'time': t_sunset,
                'alt_sol': 0.0,
                'az_sol': az_interp,
                'd': np.nan,
                'mag': np.nan
            }
            break

    return eventos

# ============================================================
# FUNCION NUEVA: SIMULACIÓN SIN REFRACCIÓN (basada en la dual, solo 1 panel)
# ============================================================
def simular_eclipse_sin_refraccion(tiempos, location, horizon_data=None, LIMITE=2.5, anim_interval=200):
    """
    Se basa en la función dual y utiliza la parte de "Sin refracción" (panel izquierdo),
    eliminando los cálculos de refracción y el panel derecho.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-LIMITE, LIMITE)
    ax.set_ylim(-LIMITE, LIMITE)
    ax.set_xlabel("Diferencia en Acimut (°)")
    ax.set_ylabel("Diferencia en Altitud (°)")
    ax.set_title("Simulación del Eclipse Solar (Sin refracción)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Usamos los mismos objetos que en el panel izquierdo de la dual
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
        # Actualizar el Sol
        sun_coord = get_sun(obstime)
        d_sun = sun_coord.distance.to(u.km).value
        sol_diam = angular_diameter(R_SUN, d_sun)
        sun_radius = sol_diam / 2.0
        sun_patch.set_radius(sun_radius)
        sun_patch.center = (0, 0)
        sol = sun_coord.transform_to(altaz_frame)
        # Actualizar la Luna
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
        time_text.set_text(f"UT:\n{obstime.iso}")
        
        # Actualizar horizonte
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
    plt.show()

# ============================================================
# FUNCION ORIGINAL: SIMULACIÓN DUAL (con y sin refracción)
# ============================================================
def simular_eclipse_dual(tiempos, location, horizon_data=None, star_catalog_data=None, 
                         t_total_range=None, LIMITE=2.5, anim_interval=200):
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

    horizon_line1, = ax1.plot([], [], color='blue', linestyle='--', linewidth=1, label="Horizonte")
    horizon_line2, = ax2.plot([], [], color='blue', linestyle='--', linewidth=1, label="Horizonte")
    
    stars_scatter1 = ax1.scatter([], [], s=[], c='white', marker='*')
    stars_scatter2 = ax2.scatter([], [], s=[], c='white', marker='*')
    
    time_text1 = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    time_text2 = ax2.text(0.05, 0.95, "", transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    def init():
        moon_patch1.center = (0, 0)
        moon_patch2.center = (0, 0)
        time_text1.set_text("")
        time_text2.set_text("")
        horizon_line1.set_data([], [])
        horizon_line2.set_data([], [])
        return (moon_patch1, moon_patch2, time_text1, time_text2, horizon_line1, horizon_line2, stars_scatter1, stars_scatter2)

    def update(frame):
        obstime = tiempos[frame]
        altaz_frame = AltAz(obstime=obstime, location=location)
        
        if t_total_range is not None:
            t_start, t_end = t_total_range
            if t_start <= obstime <= t_end:
                ax1.set_facecolor('black')
                ax2.set_facecolor('black')
            else:
                ax1.set_facecolor('white')
                ax2.set_facecolor('white')
        
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
        time_text1.set_text(f"UT:\n{obstime.iso}")

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
            horizon_line1.set_data(pts[:,0], pts[:,1])
        else:
            horizon_d_y = 0 - sol.alt.to(u.deg).value
            if -LIMITE <= horizon_d_y <= LIMITE:
                x_vals = np.array([-LIMITE, LIMITE])
                y_vals = np.array([horizon_d_y, horizon_d_y])
                horizon_line1.set_data(x_vals, y_vals)
            else:
                horizon_line1.set_data([], [])

        d_sun = sol.distance.to(u.km).value
        sol_diam = angular_diameter(R_SUN, d_sun)
        sun_radius = sol_diam / 2.0

        alt_sol = sol.alt.to(u.deg).value
        az_sol = sol.az.to(u.deg).value

        sol_eff_alt, sol_eff_az = aplicar_refraction(alt_sol, az_sol, d_sun, Refraction_inst, lmbda)

        alt_top = alt_sol + sun_radius
        alt_top_eff, _ = aplicar_refraction(alt_top, az_sol, d_sun, Refraction_inst, lmbda)
        alt_bot = alt_sol - sun_radius
        alt_bot_eff, _ = aplicar_refraction(alt_bot, az_sol, d_sun, Refraction_inst, lmbda)
        sol_diam_vert_eff = alt_top_eff - alt_bot_eff
        sol_diam_horiz = sol_diam

        sun_patch2.center = (0, 0)
        sun_patch2.width = sol_diam_horiz
        sun_patch2.height = sol_diam_vert_eff
        sun_patch2.angle = 0

        luna = get_body("moon", obstime, location=location).transform_to(altaz_frame)
        d_luna = luna.distance.to(u.km).value
        luna_diam = angular_diameter(R_MOON, d_luna)
        luna_radius = luna_diam / 2.0

        alt_luna = luna.alt.to(u.deg).value
        az_luna = luna.az.to(u.deg).value

        luna_eff_alt, luna_eff_az = aplicar_refraction(alt_luna, az_luna, d_luna, Refraction_inst, lmbda)

        alt_top_luna = alt_luna + luna_radius
        alt_top_luna_eff, _ = aplicar_refraction(alt_top_luna, az_luna, d_luna, Refraction_inst, lmbda)
        alt_bot_luna = alt_luna - luna_radius
        alt_bot_luna_eff, _ = aplicar_refraction(alt_bot_luna, az_luna, d_luna, Refraction_inst, lmbda)
        luna_diam_vert_eff = alt_top_luna_eff - alt_bot_luna_eff
        luna_diam_horiz = luna_diam

        d_az_eff = (luna_eff_az - sol_eff_az)
        d_x_eff = d_az_eff * np.cos(np.deg2rad(sol_eff_alt))
        d_y_eff = luna_eff_alt - sol_eff_alt

        moon_patch2.center = (d_x_eff, d_y_eff)
        moon_patch2.width = luna_diam_horiz
        moon_patch2.height = luna_diam_vert_eff
        moon_patch2.angle = 0

        time_text2.set_text(f"UT:\n{obstime.iso}")

        if horizon_data is not None:
            pts2 = []
            for point in horizon_data:
                az_h, alt_h = point
                d_az_point = ((az_h - sol_eff_az + 180) % 360) - 180
                x_pt = d_az_point * np.cos(np.deg2rad(sol_eff_alt))
                y_pt = alt_h - sol_eff_alt
                pts2.append([x_pt, y_pt])
            pts2 = np.array(pts2)
            horizon_line2.set_data(pts2[:,0], pts2[:,1])
        else:
            horizon_d_y_eff = 0 - sol_eff_alt
            if -LIMITE <= horizon_d_y_eff <= LIMITE:
                x_vals = np.array([-LIMITE, LIMITE])
                y_vals = np.array([horizon_d_y_eff, horizon_d_y_eff])
                horizon_line2.set_data(x_vals, y_vals)
            else:
                horizon_line2.set_data([], [])
        
        stars_x1, stars_y1, stars_sizes1 = [], [], []
        stars_x2, stars_y2, stars_sizes2 = [], [], []
        if star_catalog_data is not None and t_total_range is not None:
            t_start, t_end = t_total_range
            if t_start <= obstime <= t_end:
                ax1.set_facecolor('black')
                ax2.set_facecolor('black')
                for star in star_catalog_data:
                    ra, dec, mag = star
                    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                    coord_altaz = coord.transform_to(AltAz(obstime=obstime, location=location))
                    if coord_altaz.alt.deg > 0:
                        sol_az_deg = az_sol
                        sol_alt_deg = alt_sol
                        d_az_star = ((coord_altaz.az.deg - sol_az_deg + 180) % 360) - 180
                        x_star = d_az_star * np.cos(sol.alt.radian)
                        y_star = coord_altaz.alt.deg - sol_alt_deg
                        stars_x1.append(x_star)
                        stars_y1.append(y_star)
                        size = max(1, (7 - mag) * 3)
                        stars_sizes1.append(size)
                        
                        sol_eff_az_deg = sol_eff_az
                        sol_eff_alt_deg = sol_eff_alt
                        d_az_star_eff = ((coord_altaz.az.deg - sol_eff_az_deg + 180) % 360) - 180
                        x_star_eff = d_az_star_eff * np.cos(np.deg2rad(sol_eff_alt_deg))
                        y_star_eff = coord_altaz.alt.deg - sol_eff_alt_deg
                        stars_x2.append(x_star_eff)
                        stars_y2.append(y_star_eff)
                        stars_sizes2.append(size)
            else:
                ax1.set_facecolor('white')
                ax2.set_facecolor('white')
            
        stars_scatter1.set_offsets(np.column_stack((stars_x1, stars_y1)))
        stars_scatter1.set_sizes(stars_sizes1)
        stars_scatter2.set_offsets(np.column_stack((stars_x2, stars_y2)))
        stars_scatter2.set_sizes(stars_sizes2)
        
        return (moon_patch1, time_text1, horizon_line1, stars_scatter1,
                moon_patch2, time_text2, horizon_line2, stars_scatter2, ax1, ax2)
    update(0)
    ani = animation.FuncAnimation(fig, update, frames=len(tiempos),
                                  init_func=init, blit=True, interval=anim_interval, repeat=True)
    plt.show()

# --- Función para abrir el perfil de PeakFinder ---
def mostrar_peakfinder(lat, lon, eclipse_time=None, luna_azi=None, luna_alt=None,
                       fov=110, cfg="sm", teleazi=-77.98, telealt=8.91, name="Frómista"):
    """
    Abre el perfil de PeakFinder usando el formato:
    https://www.peakfinder.com/es/?lat=<lat>&lng=<lon>&azi=<luna_azi>&alt=<luna_alt>&fov=<fov>&date=<date_iso>&cfg=<cfg>&teleazi=<teleazi>&telealt=<telealt>&name=<name>
    Se convierte eclipse_time a formato ISO con separador "T" (ej.: 2026-08-12T18:29:20Z).
    """
    date_str = ""
    if eclipse_time is not None:
        date_str = eclipse_time.iso.split('.')[0].replace(" ", "T") + "Z"
    base_url = "https://www.peakfinder.com/es/?"
    params = f"lat={lat}&lng={lon}"
    if luna_azi is not None:
        params += f"&azi={luna_azi}"
    if luna_alt is not None:
        params += f"&alt={luna_alt}"
    params += f"&fov={fov}"
    if date_str:
        params += f"&date={date_str}"
    params += f"&cfg={cfg}"
    if teleazi is not None:
        params += f"&teleazi={teleazi}"
    if telealt is not None:
        params += f"&telealt={telealt}"
    if name:
        params += f"&name={name}"
    url = base_url + params
    print("\nAbriendo PeakFinder en el navegador:")
    print(url)
    webbrowser.open(url)

# --- Función para abrir ShadeMap ---
def mostrar_shademap(lat, lon, elev, eclipse_time=None, zoom=13, bearing=0, pitch=0, margin=0):
    """
    Abre ShadeMap utilizando el formato:
    https://shademap.app/@<lat>,<lon>,<zoom>z,<timestamp>t,<bearing>b,<pitch>p,<margin>m
    Se obtiene el timestamp (en milisegundos) a partir de eclipse_time, si se proporciona.
    """
    base_url = "https://shademap.app/@"
    lat_str = f"{lat:.3f}"
    lon_str = f"{lon:.3f}"
    if eclipse_time is not None:
        ts_ms = int(eclipse_time.unix * 1000)
    else:
        from time import time as current_time
        ts_ms = int(current_time() * 1000)
    url = f"{base_url}{lat_str},{lon_str},{zoom}z,{ts_ms}t,{bearing}b,{pitch}p,{margin}m"
    print("\nAbriendo ShadeMap en el navegador:")
    print(url)
    webbrowser.open(url)

def seleccionar_localizacion():
    print("Seleccione la forma de indicar la localización:")
    print("1. Escribir el nombre de un municipio o lugar")
    print("2. Ingresar coordenadas manualmente")
    print("3. Seleccionar de una lista de capitales de provincia y otras ciudades relevantes (España)")
    try:
        opcion = int(input("Ingrese el número de la opción deseada: "))
    except:
        opcion = 0

    if opcion == 1:
        nombre = input("Ingrese el nombre del municipio o lugar: ").strip()
        lat, lon, direccion = geocode_lugar(nombre)
        if lat is None or lon is None:
            print("No se encontraron resultados para ese nombre. Intente nuevamente.")
            return seleccionar_localizacion()
        print(f"Localización encontrada: {direccion} ({lat}, {lon})")
        return lat, lon, direccion
    elif opcion == 2:
        try:
            lat = float(input("Ingrese la latitud (ej. 40.4168): "))
            lon = float(input("Ingrese la longitud (ej. -3.7038): "))
            direccion = reverse_geocode(lat, lon)
            print(f"Localización: {direccion} ({lat}, {lon})")
            return lat, lon, direccion
        except:
            print("Error en la entrada de coordenadas. Intente nuevamente.")
            return seleccionar_localizacion()
    elif opcion == 3:
        print("Lista de capitales de provincia en España:")
        for i, (ciudad, coords) in enumerate(capitales.items(), start=1):
            lat_cap, lon_cap = coords
            direccion = reverse_geocode(lat_cap, lon_cap)
            print(f"{i:2d}. {ciudad}: {direccion} ({lat_cap}, {lon_cap})")
            pytime.sleep(0.2)
        try:
            num = int(input("Ingrese el número de la opción deseada: "))
            ciudades = list(capitales.keys())
            ciudad_seleccionada = ciudades[num-1]
            lat, lon = capitales[ciudad_seleccionada]
            direccion = reverse_geocode(lat, lon)
            print(f"Seleccionado: {ciudad_seleccionada} - {direccion} ({lat}, {lon})")
            return lat, lon, direccion
        except:
            print("Opción no válida. Intente nuevamente.")
            return seleccionar_localizacion()
    else:
        print("Opción no válida. Intente nuevamente.")
        return seleccionar_localizacion()

def descargar_catalogo_estrellas(center_coord, radius=2.5, mag_limite=10):
    Vizier.ROW_LIMIT = -1
    center_coord = SkyCoord(ra=center_coord.ra, dec=center_coord.dec, unit='deg', frame='icrs')
    columns = ["RA_ICRS", "DE_ICRS", "Gmag"]
    vizier = Vizier(columns=columns, row_limit=-1)
    try:
        result = vizier.query_region(center_coord, radius=radius*u.deg, catalog="I/355/gaiadr3")
    except Exception as e:
        print("Error al consultar Vizier:", e)
        return None
    if result:
        stars = result[0].to_pandas()
        stars_filtered = stars[stars["Gmag"] < mag_limite]
        return stars_filtered[["RA_ICRS", "DE_ICRS", "Gmag"]].to_numpy()
    else:
        print("No se encontraron estrellas en el campo seleccionado.")
        return None

def main():
    print("=== Cálculo de eventos, horizonte, catálogo de estrellas y simulación de eclipse solar ===\n")
    print("La hora de entrada es en Tiempo Universal (UT).\n")
    
    # Seleccionar localización
    lat, lon, loc_desc = seleccionar_localizacion()
    
    # Elección interactiva de elevación
    alt_input = input("Ingrese la elevación en metros (ej. 667) [deje en blanco para obtener automáticamente]: ").strip()
    if alt_input == "":
        alt_auto = obtener_altitud(lat, lon)
        if alt_auto is not None:
            elev = alt_auto
            print(f"Altitud obtenida automáticamente: {elev} metros.")
        else:
            elev = 667
            print("No se pudo obtener la altitud automáticamente. Se usará 667 metros.")
    else:
        try:
            elev = float(alt_input)
        except:
            elev = 667
            print("Entrada inválida. Se usará 667 metros.")
    
    # Hora por defecto actualizada a "2026-08-12 17:30:00"
    inicio_str = input("Ingrese la fecha/hora de inicio (YYYY-MM-DD HH:MM:SS UT) [default: 2026-08-12 17:30:00]: ") or "2026-08-12 17:30:00"
    duracion_str = input("Ingrese la duración de la simulación en minutos [default: 60]: ") or "60"
    try:
        t_ini = Time(inicio_str)
    except Exception:
        print("Error al interpretar la fecha/hora. Use el formato YYYY-MM-DD HH:MM:SS.")
        sys.exit(1)
    duracion = float(duracion_str)
    
    # Entrada interactiva para definir dt (intervalo de tiempo en segundos)
    dt_input = input("Ingrese el intervalo de tiempo (dt) en segundos para la simulación [default: 10]: ").strip()
    try:
        dt = float(dt_input) if dt_input != "" else 10.0
    except Exception:
        dt = 10.0
        print("Valor de dt inválido. Se usará dt = 10 segundos.")
    
    # Entrada para el intervalo de actualización (anim_interval)
    interval_input = input("Ingrese el intervalo de actualización para la animación en milisegundos [default: 200]: ").strip()
    try:
        anim_interval = int(interval_input) if interval_input.strip() != "" else 200
    except:
        anim_interval = 200
        print("Valor de anim_interval inválido. Se usará 200 ms.")
    
    # Definir LIMITE para la visualización
    LIMITE = 2.5

    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
    
    # Generar el array de tiempos usando dt (en segundos)
    n_steps = int((duracion * 60) / dt) + 1
    tiempos = t_ini + TimeDelta(np.arange(0, n_steps * dt, dt), format='sec')
    
    # Opción de horizonte artificial (PeakFinder)
    resp_horizonte_artif = input("\n¿Desea ver el horizonte artificial (PeakFinder)? (s/n): ").strip().lower()
    eventos = None
    if resp_horizonte_artif == 's':
        params_array = [obtener_parametros(t, location) for t in tiempos]
        eventos = detectar_eventos(tiempos, params_array, location)
        
        if "Máximo Eclipse" in eventos:
            t_max = eventos["Máximo Eclipse"]['time']
            altaz_frame = AltAz(obstime=t_max, location=location)
            luna_pos = get_body("moon", t_max, location=location).transform_to(altaz_frame)
            def_azi = luna_pos.az.deg
            def_alt = luna_pos.alt.deg
        else:
            t_max = None
            def_azi, def_alt = None, None
            
        print("Puede ajustar manualmente azimut y altitud para la vista del horizonte artificial.")
        azi_str = input(f"Ingrese azimut (°) [default: {def_azi if def_azi is not None else 'sin ajustar'}]: ")
        alt_str = input(f"Ingrese altitud (°) [default: {def_alt if def_alt is not None else 'sin ajustar'}]: ")
        try:
            azi_val = float(azi_str) if azi_str.strip() != "" else def_azi
        except:
            azi_val = def_azi
        try:
            alt_val = float(alt_str) if alt_str.strip() != "" else def_alt
        except:
            alt_val = def_alt
        mostrar_peakfinder(lat, lon, eclipse_time=t_max, luna_azi=azi_val, luna_alt=alt_val,
                           fov=110, cfg="sm", teleazi=-77.98, telealt=8.91, name="Frómista")
    else:
        print("No se abrirá el horizonte artificial.")

    # Opción de ShadeMap (zoom 13 por defecto)
    resp_shademap = input("\n¿Desea ver ShadeMap (https://shademap.app/)? (s/n): ").strip().lower()
    if resp_shademap == 's':
        if eventos is not None and "Máximo Eclipse" in eventos:
            t_max = eventos["Máximo Eclipse"]['time']
        else:
            t_max = None
        mostrar_shademap(lat, lon, elev, eclipse_time=t_max, zoom=13, bearing=0, pitch=0, margin=0)
    else:
        print("No se abrirá ShadeMap.")

    # Opción de cargar fichero de horizonte real
    horizonte_data = None
    resp_horizonte_file = input("\n¿Desea cargar un fichero opcional con datos reales del horizonte? (s/n): ").strip().lower()
    if resp_horizonte_file == 's':
        filename = input("Ingrese la ruta del fichero (formato: dos columnas 'az alt', con cabecera): ").strip()
        horizonte_data = cargar_horizonte(filename)
        if horizonte_data is not None:
            print(f"Fichero '{filename}' cargado correctamente.")
        else:
            print("No se pudo cargar el fichero. Se usará el horizonte simple.")

    # Opción de descargar automáticamente el catálogo de estrellas
    star_catalog_auto = None
    resp_star_auto = input("\n¿Desea descargar automáticamente el catálogo de estrellas para el campo de totalidad? (s/n): ").strip().lower()
    t_total_range = None
    if resp_star_auto == 's':
        params_array = [obtener_parametros(t, location) for t in tiempos]
        eventos = detectar_eventos(tiempos, params_array, location)
        if "Segundo Contacto" in eventos and "Tercer Contacto" in eventos:
            t_total_range = (eventos["Segundo Contacto"]["time"], eventos["Tercer Contacto"]["time"])
            print("\nEclipse total detectado. Totalidad entre:")
            print(f"  Segundo Contacto: {t_total_range[0].iso}")
            print(f"  Tercer Contacto:  {t_total_range[1].iso}")
            t_max = eventos["Máximo Eclipse"]["time"]
            sun_coord = get_sun(t_max)
            star_catalog_auto = descargar_catalogo_estrellas(sun_coord, radius=2.5, mag_limite=10)
        else:
            print("\nNo se detectó eclipse total; no se descargará el catálogo automáticamente.")
    else:
        params_array = [obtener_parametros(t, location) for t in tiempos]
        eventos = detectar_eventos(tiempos, params_array, location)

    header = f"{'Evento':<20} {'UT':<25} {'Alt. Sol (°)':<12} {'Az. Sol (°)':<12} {'d (°)':<8} {'Magnitud':<10} {'Ángulo':<10}"
    print("\n" + header)
    print("-" * len(header))
    for evento, data in eventos.items():
        ut_str = data['time'].iso if 'time' in data else "N/A"
        alt_str = f"{data['alt_sol']:.2f}" if 'alt_sol' in data and data['alt_sol'] is not None else "N/A"
        az_str = f"{data['az_sol']:.2f}" if 'az_sol' in data and data['az_sol'] is not None else "N/A"
        d_str = f"{data['d']:.3f}" if 'd' in data and not np.isnan(data['d']) else "N/A"
        mag_str = f"{data['mag']:.3f}" if 'mag' in data and not np.isnan(data['mag']) else "N/A"
        angle_str = f"{data['angle']:.2f}" if 'angle' in data else "N/A"
        print(f"{evento:<20} {ut_str:<25} {alt_str:<12} {az_str:<12} {d_str:<8} {mag_str:<10} {angle_str:<10}")

    if "Máximo Eclipse" in eventos:
        t_max = eventos["Máximo Eclipse"]['time']
        ratio, luna_diam, sol_diam = calcular_ratio_moon_sun(t_max, location)
        print(f"\nEn el momento del Máximo Eclipse:")
        print(f"  Diámetro angular del Sol: {sol_diam:.3f}°")
        print(f"  Diámetro angular de la Luna: {luna_diam:.3f}°")
        print(f"  Moon/Sun size ratio: {ratio:.3f}")

    if "Primer Contacto" in eventos and "Cuarto Contacto" in eventos:
        duracion_total = (eventos["Cuarto Contacto"]['time'] - eventos["Primer Contacto"]['time']).sec
        print(f"\nDuración total del eclipse (Primer a Cuarto Contacto): {duracion_total:.0f} segundos")
    else:
        print("\nNo se pudieron determinar los contactos de entrada/salida para calcular la duración total.")

    if "Segundo Contacto" in eventos and "Tercer Contacto" in eventos:
        duracion_totalidad = (eventos["Tercer Contacto"]['time'] - eventos["Segundo Contacto"]['time']).sec
        print(f"Duración de la totalidad: {duracion_totalidad:.0f} segundos")
    else:
        print("No se detectó eclipse total para calcular la duración de la totalidad.")
    
    # Preguntar si se desea ver la simulación dual o la simulación sin refracción
    resp_sim_dual = input("\n¿Desea ver la simulación dual (sin y con refracción)? (s/n): ").strip().lower()
    if resp_sim_dual == 's':
        simular_eclipse_dual(tiempos, location, horizon_data=horizonte_data,
                             star_catalog_data=star_catalog_auto, t_total_range=t_total_range, LIMITE=LIMITE, anim_interval=anim_interval)
    else:
        resp_sim = input("\n¿Desea ver la simulación animada sin refracción? (s/n): ").strip().lower()
        if resp_sim == 's':
            simular_eclipse_sin_refraccion(tiempos, location, horizon_data=horizonte_data, LIMITE=LIMITE, anim_interval=anim_interval)
        else:
            print("Simulación terminada.")

if __name__ == "__main__":
    main()





