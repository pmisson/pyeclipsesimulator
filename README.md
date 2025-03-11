# Cálculo y Simulación de Eclipses Solares

Este repositorio contiene un script en Python para calcular y visualizar eventos astronómicos relacionados con eclipses solares. Permite la selección de ubicaciones, descarga de catálogos de estrellas, simulaciones con y sin refracción y la generación de tablas de eventos.

## Características Principales
- Cálculo de eventos clave de un eclipse solar (contactos, magnitud, duración, etc.).
- Selección de localización mediante:
  - Nombre de municipio o lugar (geocoding).
  - Coordenadas manuales.
  - Lista de capitales de provincia en España.
- Obtención automática de altitud con la API de Open-Meteo.
- Descarga manual o automática de catálogos de estrellas (Vizier).
- Simulación animada de eclipses con y sin refracción atmosférica.
- Opcionalmente, permite la carga de datos reales del horizonte.
- Apertura de horizonte artificial mediante PeakFinder.

## Requisitos
Este script requiere Python 3.x y las siguientes dependencias:

```bash
pip install astropy matplotlib numpy geopy requests astroquery RefractionShift pandas AstroAtmosphere
```

## Uso
Ejecutar el script:

```bash
python secuencia_sol_luna8.py
```

El usuario deberá seguir las instrucciones interactivas para seleccionar la ubicación y configuración de la simulación.

## Funcionalidades Clave del Script
1. **Selección de ubicación**: permite elegir entre coordenadas manuales, geocoding o una lista predefinida de capitales de provincia en España.
2. **Obtención de altitud**: si no se ingresa manualmente, se obtiene automáticamente desde Open-Meteo.
3. **Cálculo de eventos del eclipse**: se identifican los cuatro contactos principales, el máximo del eclipse y, si aplica, la totalidad.
4. **Descarga y carga de catálogos de estrellas**: se puede cargar un fichero de estrellas manualmente o descargarlo automáticamente desde Vizier.
5. **Simulación del eclipse**:
   - Representación animada con posiciones reales del Sol y la Luna.
   - Simulación dual con refracción atmosférica.
   - Opcion de incluir estrellas visibles durante la totalidad.
6. **Opcionalmente, permite cargar un fichero con datos reales del horizonte.**
7. **Permite abrir la herramienta PeakFinder para visualizar el horizonte artificial.**

## Comparación de Resultados
Los resultados han sido comparados con los de [xjubier.free.fr](http://xjubier.free.fr/) y existen discrepancias de pocos segundos. Aun así, el propósito de este software es puramente demostrativo. Se recomienda revisar el código, el cual ha sido generado en su mayoría por ChatGPT o3_mini_high bajo la supervisión de Alejandro Sánchez de Miguel.

## Licencia
Este proyecto está licenciado bajo la licencia MIT. Puedes usarlo y modificarlo libremente.

