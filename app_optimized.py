import dash
from dash import html, dcc, dash_table, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pycountry_convert as pc
import pycountry

def create_density_heatmap(df):
    # --- 1. Preparación de datos ---
    num_bins_tendencia = 50 
    df_copy = df.copy()
    df_copy['irating_bin'] = pd.cut(df_copy['IRATING'], bins=num_bins_tendencia)
    # MODIFICACIÓN: Añadimos 'mean' al cálculo de agregación
    stats_per_bin = df_copy.groupby('irating_bin')['AVG_INC'].agg(['max', 'min', 'mean']).reset_index()
    stats_per_bin['irating_mid'] = stats_per_bin['irating_bin'].apply(lambda b: b.mid)
    stats_per_bin = stats_per_bin.sort_values('irating_mid').dropna()

    # --- 2. CÁLCULO DE LA REGRESIÓN LINEAL ---
    # Coeficientes para máximos y mínimos (sin cambios)
    max_coeffs = np.polyfit(stats_per_bin['irating_mid'], stats_per_bin['max'], 1)
    max_line_func = np.poly1d(max_coeffs)
    min_coeffs = np.polyfit(stats_per_bin['irating_mid'], stats_per_bin['min'], 1)
    min_line_func = np.poly1d(min_coeffs)
    
    # NUEVO: Coeficientes para la línea de promedios
    mean_coeffs = np.polyfit(stats_per_bin['irating_mid'], stats_per_bin['mean'], 1)
    mean_line_func = np.poly1d(mean_coeffs)

    # Generamos los puntos Y para las líneas rectas
    x_trend = stats_per_bin['irating_mid']
    y_trend_max = max_line_func(x_trend)
    y_trend_min = min_line_func(x_trend)
    y_trend_mean = mean_line_func(x_trend) # NUEVO

    # --- 3. Creación de las trazas del gráfico ---
    heatmap_trace = go.Histogram2d(
        x=df['IRATING'],
        y=df['AVG_INC'],
        colorscale='Plasma',
        nbinsx=100, nbinsy=100, zmin=0, zmax=50,
        name='Densidad'
    )
    
    max_line_trace = go.Scatter(
        x=x_trend, y=y_trend_max, mode='lines',
        name='Tendencia Máximo AVG_INC',
        line=dict(color='red', width=1, dash='dash')
    )
    
    min_line_trace = go.Scatter(
        x=x_trend, y=y_trend_min, mode='lines',
        name='Tendencia Mínimo AVG_INC',
        line=dict(color='lime', width=1, dash='dash')
    )

    # NUEVO: Traza para la línea de promedio
    mean_line_trace = go.Scatter(
        x=x_trend,
        y=y_trend_mean,
        mode='lines',
        name='Tendencia Promedio AVG_INC',
        line=dict(color='black', width=2, dash='solid')
    )

    # --- 4. Combinación de las trazas en una sola figura ---
    # MODIFICACIÓN: Añadimos la nueva traza a la lista de datos
    fig = go.Figure(data=[heatmap_trace, max_line_trace, min_line_trace, mean_line_trace])
    
    fig.update_layout(
        title='Densidad de Pilotos: iRating vs. AVG_INC',
        xaxis_title='iRating',
        yaxis_title='Incidents Per Race',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 12000], showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=[0,25], showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig


def country_to_continent_code(country_code):
    try:
        # Convierte el código de país de 2 letras a código de continente
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        return continent_code
    except (KeyError, TypeError):
        # Devuelve 'Otros' si el código no se encuentra o es inválido
        return 'Otros'

'''def create_continent_map(df):
    # Contamos cuántos pilotos hay en cada país
    country_counts = df['LOCATION'].value_counts().reset_index()
    country_counts.columns = ['LOCATION_2_LETTER', 'PILOTOS']

    # --- CORRECCIÓN: Convertimos códigos de 2 letras a 3 letras ---
    def alpha2_to_alpha3(code):
        try:
            return pycountry.countries.get(alpha_2=code).alpha_3
        except AttributeError:
            return None # Ignora códigos que no se pueden convertir

    country_counts['LOCATION_3_LETTER'] = country_counts['LOCATION_2_LETTER'].apply(alpha2_to_alpha3)
    
    # Eliminamos filas que no se pudieron convertir
    country_counts.dropna(subset=['LOCATION_3_LETTER'], inplace=True)

    fig = px.choropleth(
        country_counts,
        locations="LOCATION_3_LETTER",  # <-- Usamos los códigos de 3 letras
        locationmode="ISO-3",           # <-- Usamos el modo ISO-3
        color="PILOTOS",
        hover_name="LOCATION_2_LETTER", # Mostramos el código de 2 letras en el hover
        color_continuous_scale=px.colors.sequential.Plasma,
        scope="world",
        projection="natural earth"
    )
    
    fig.update_layout(
        title_text='🌍 Pilotos por País',
        template='plotly_dark',
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='#4E5D6C',
            landcolor='#323232',
            subunitcolor='grey'
        ),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig'''

def create_continent_map(df, selected_region='ALL', selected_country='ALL'):
    # La preparación de datos es la misma
    country_counts = df['LOCATION'].value_counts().reset_index()
    country_counts.columns = ['LOCATION_2_LETTER', 'PILOTOS']

    def alpha2_to_alpha3(code):
        try:
            return pycountry.countries.get(alpha_2=code).alpha_3
        except (LookupError, AttributeError):
            return None

    country_counts['LOCATION_3_LETTER'] = country_counts['LOCATION_2_LETTER'].apply(alpha2_to_alpha3)
    country_counts.dropna(subset=['LOCATION_3_LETTER'], inplace=True)

    # Lógica de coloreado y hover avanzada (sin cambios)
    show_scale = True 
    if selected_region != 'ALL':
        show_scale = False
        countries_in_region = iracing_ragions.get(selected_region, [])
        
        def get_color_code(row):
            if row['LOCATION_2_LETTER'] == selected_country: return 2
            elif row['LOCATION_2_LETTER'] in countries_in_region: return 1
            else: return 0

        country_counts['COLOR'] = country_counts.apply(get_color_code, axis=1)
        color_column = 'COLOR'
        color_scale = [[0.0, 'rgba(50, 50, 50, 1)'], [0.5, 'rgba(0, 111, 255, 1)'], [1.0, 'rgba(0, 200, 80, 1)']]
        range_color_val = [0, 2]
    else:
        color_column = 'PILOTOS'
        color_scale = [[0.0, "#EBE8E8"], [0.01, "#BAC5DB"], [0.3, "#60D351"], [1.0, "rgba(0,111,255,1)"]]
        range_color_val = [0, 10000]

    # Creación del mapa base
    fig = px.choropleth(
        country_counts,
        locations="LOCATION_3_LETTER",
        locationmode="ISO-3",
        color=color_column,
        # --- CORRECCIÓN CLAVE AQUÍ ---
        # Ya no usamos hover_name, pasamos todo a custom_data
        custom_data=['LOCATION_2_LETTER', 'PILOTOS'],
        color_continuous_scale=color_scale,
        projection="natural earth",
        range_color=range_color_val
    )
    
    # Actualizamos la plantilla del hover para usar las variables correctas de custom_data
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Pilotos: %{customdata[1]}<extra></extra>"
    )
    
    # Lógica de zoom dinámico (sin cambios)
    if selected_country != 'ALL' and selected_country in country_coords:
        zoom_level = 4 if selected_country not in ['US', 'CA', 'AU', 'BR', 'AR'] else 3
        fig.update_geos(center=country_coords[selected_country], projection_scale=zoom_level)
    elif selected_region != 'ALL':
        countries_in_region = iracing_ragions.get(selected_region, [])
        lats = [country_coords[c]['lat'] for c in countries_in_region if c in country_coords]
        lons = [country_coords[c]['lon'] for c in countries_in_region if c in country_coords]
        if lats and lons:
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            zoom_level = 2
            if len(countries_in_region) < 5: zoom_level = 4
            elif len(countries_in_region) < 15: zoom_level = 3
            fig.update_geos(center={'lat': center_lat, 'lon': center_lon}, projection_scale=zoom_level)
        else:
            fig.update_geos(center={'lat': 20, 'lon': 0}, projection_scale=1)
    else:
        fig.update_geos(center={'lat': 20, 'lon': 0}, projection_scale=1)

    fig.update_layout(
        title_text='🌍 Pilotos por País',
        template='plotly_dark',
        geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#4E5D6C', landcolor='#323232', subunitcolor='grey'),
        margin={"r":0,"t":40,"l":0,"b":0},
        coloraxis_showscale=show_scale
    )
    return fig

def create_histogram_with_percentiles(df, column='IRATING', bin_width=100):
    # Crear bins específicos de 100 en 100
    min_val = df[column].min()
    max_val = df[column].max()
    bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
    
    hist, bin_edges = np.histogram(df[column], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    total = len(df)
    hover_text = []
    
    # Transformación personalizada para hacer más visibles los valores pequeños
    hist_transformed = []
    for i in range(len(hist)):
        # Percentil: % de pilotos con menos iRating que el límite inferior del bin superior
        below = (df[column] < bin_edges[i+1]).sum()
        percentile = below / total * 100
        top_percent = 100 - percentile
        hover_text.append(
            f"Rango: {int(bin_edges[i])}-{int(bin_edges[i+1])}<br>"
            f"Pilotos: {hist[i]}<br>"
            f"Top: {top_percent:.2f}%"
        )
        
        # Transformación: valores pequeños (0-50) los amplificamos
        if hist[i] <= 50 and hist[i] > 0:
            hist_transformed.append(hist[i] + 2)  # Les sumamos 25 para hacerlos más visibles
        else:
            hist_transformed.append(hist[i])  # Valores normales sin cambio
    
    fig = go.Figure(data=go.Bar(
        x=bin_centers,
        y=hist_transformed,  # <-- Usamos los valores transformados para visualización
        width=bin_widths * 1,
        hovertext=hover_text,  # <-- Pero en el hover mostramos los valores reales
        hovertemplate='%{hovertext}<extra></extra>',
        marker=dict(color=hist, colorscale=[
            [0.0, "#FFFFFF"],   # Empieza con un azul claro (LightSkyBlue)
            [0.1, "#A0B8EC"],
            [0.3, "#668CDF"],   # Pasa por un cian muy pálido (LightCyan)
            [1.0, "rgba(0,111,255,1)"]    # Termina en blanco
        ])
    ))
    
    fig.update_layout(
        title='iRating Distribution',
        xaxis_title='iRating',
        yaxis_title='Drivers',
        template='plotly_dark',
        hovermode='x unified',
        # --- AÑADE ESTAS LÍNEAS PARA CAMBIAR EL FONDO ---
        paper_bgcolor='rgba(30,30,38,1)', # Fondo de toda la figura (transparente)
        plot_bgcolor='rgba(30,30,38,1)',  # Fondo del área de las barras (transparente)
        # --- FIN DE LAS LÍNEAS AÑADIDAS ---
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

def create_correlation_heatmap(df):
    # Seleccionar solo columnas numéricas para la correlación
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r', # Rojo-Azul invertido (Rojo=positivo, Azul=negativo)
        zmin=-1, zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size":12}
    ))
    
    fig.update_layout(
        title='🔁 Correlación entre Variables',
        template='plotly_dark',
        margin=dict(l=40, r=20, t=40, b=20)
    )
    return fig

def flag_img(code):
    url = f"https://flagcdn.com/16x12/{code.lower()}.png"
    if code in country_flags:
        return f'![{code}]({url})'
    else:
        return f'`{code}`' 
    

country_flags = {
    'ES': '🇪🇸', 'US': '🇺🇸', 'BR': '🇧🇷', 'DE': '🇩🇪', 'FR': '🇫🇷', 'IT': '🇮🇹',
    'GB': '🇬🇧', 'PT': '🇵🇹', 'NL': '🇳🇱', 'AU': '🇦🇺', 'JP': '🇯🇵', 'CA': '🇨🇦',
    'AR': '🇦🇷', 'MX': '🇲🇽', 'CL': '🇨🇱', 'BE': '🇧🇪', 'FI': '🇫🇮', 'SE': '🇸🇪',
    'NO': '🇳🇴', 'DK': '🇩🇰', 'IE': '🇮🇪', 'CH': '🇨🇭', 'AT': '🇦🇹', 'PL': '🇵🇱',
    # ...agrega los que necesites...
}

country_coords = {
    'ES': {'lat': 40.4, 'lon': -3.7}, 'US': {'lat': 39.8, 'lon': -98.5},
    'BR': {'lat': -14.2, 'lon': -51.9}, 'DE': {'lat': 51.1, 'lon': 10.4},
    'FR': {'lat': 46.2, 'lon': 2.2}, 'IT': {'lat': 41.8, 'lon': 12.5},
    'GB': {'lat': 55.3, 'lon': -3.4}, 'PT': {'lat': 39.3, 'lon': -8.2},
    'NL': {'lat': 52.1, 'lon': 5.2}, 'AU': {'lat': -25.2, 'lon': 133.7},
    'JP': {'lat': 36.2, 'lon': 138.2}, 'CA': {'lat': 56.1, 'lon': -106.3},
    'AR': {'lat': -38.4, 'lon': -63.6}, 'MX': {'lat': 23.6, 'lon': -102.5},
    'CL': {'lat': -35.6, 'lon': -71.5}, 'BE': {'lat': 50.5, 'lon': 4.4},
    'FI': {'lat': 61.9, 'lon': 25.7}, 'SE': {'lat': 60.1, 'lon': 18.6},
    'NO': {'lat': 60.4, 'lon': 8.4}, 'DK': {'lat': 56.2, 'lon': 9.5},
    'IE': {'lat': 53.4, 'lon': -8.2}, 'CH': {'lat': 46.8, 'lon': 8.2},
    'AT': {'lat': 47.5, 'lon': 14.5}, 'PL': {'lat': 51.9, 'lon': 19.1},
}

iracing_ragions = {
    'US':['US'],
    'Mexico':['MX'],
    'Brazil':['BR'],
    'Canada':['CA'],
    'Atlantic':['GL'],
    'Japan':['JP'],
    'South America':['AR','PE','UY','CL','PY','BO','EC','CO','VE','GY','PA','CR','NI','HN','GT','BZ','SV','JM','DO','BS'],
    'Iberia':['ES','PT'],
    'International':['RU','IL'],
    'France':['FR'],
    'UK & I':['GB','IE'], # <-- CORREGIDO: ' IE' -> 'IE' y nombre
    'Africa':['ZA','BW','ZW','ZM','CD','GA','BI','RW','UG','KE','SO','MG','SZ','NG','GH','CI','BF','NE','GW','GM','SN','MR','EH','MA','DZ','LY','TN','EG','DJ'],
    'Italy':['IT'],
    'Central-Eastern Europe':['PL','CZ','SK','HU','SI','HR','RS','ME','AL','RO','MD','UA','BY','EE','LV','LT'],
    'Finland':['FI'],
    'DE-AT-CH':['CH','AT','DE'], # <-- CORREGIDO: Eliminado el ''
    'Scandinavia':['DK','SE','NO'],
    'Australia & NZ':['AU','NZ'],
    'Asia':['SA','JO','IQ','YE','OM','AE','QA','IN','PK','AF','NP','BD','MM','TH','KH','VN','MY','ID','CN','PH','KR','MN','KZ','KG','UZ','TJ','AF','TM','LK']
}

# --- 1. Carga y Preparación de Datos ---
df = pd.read_csv('Sports_Car_driver_stats.csv')
df = df[df['IRATING'] > 100]
df = df[df['STARTS'] > 20]
df = df[df['STARTS'] < 2000]
df = df[df['CLASS'].str.contains('D|C|B|A|P', na=False)]
print(len(df))

# --- AÑADE ESTE BLOQUE PARA CREAR LA COLUMNA 'REGION' ---
# 1. Invertir el diccionario iracing_ragions para un mapeo rápido
country_to_region_map = {country: region 
                         for region, countries in iracing_ragions.items() 
                         for country in countries}

# 2. Crear la nueva columna 'REGION' usando el mapa
df['REGION'] = df['LOCATION'].map(country_to_region_map)

# 3. Rellenar los países no encontrados con un valor por defecto
df['REGION'].fillna('International', inplace=True)
# --- FIN DEL BLOQUE AÑADIDO ---

# --- AÑADE ESTE BLOQUE PARA CALCULAR LOS RANKINGS ---
# --- CORRECCIÓN: Cambiamos method='dense' por method='first' para rankings únicos ---
df['Rank World'] = df['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
df['Rank Region'] = df.groupby('REGION')['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
df['Rank Country'] = df.groupby('LOCATION')['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
# --- FIN DEL BLOQUE AÑADIDO ---

df['CONTINENT'] = df['LOCATION'].apply(country_to_continent_code)

df['CLASS'] = df['CLASS'].str[0]

#df = df[df['IRATING'] < 10000]

# --- MODIFICACIÓN: Definimos las columnas que queremos en la tabla ---
# Usaremos esta lista más adelante para construir el df_table dinámicamente
TABLE_COLUMNS = ['DRIVER', 'IRATING', 'LOCATION', 'Rank World', 'Rank Region', 'Rank Country']
df_table = df[TABLE_COLUMNS]

df_for_graphs = df.copy() # Usamos una copia completa para los gráficos

# --- MODIFICACIÓN: Nos aseguramos de que el df principal tenga todas las columnas necesarias ---
df = df[['DRIVER','IRATING','LOCATION','STARTS','WINS','AVG_START_POS','AVG_FINISH_POS','AVG_INC','TOP25PCNT', 'REGION', 'Rank World', 'Rank Region', 'Rank Country','CLASS']]

# Aplica solo el emoji/código si está en country_flags, si no deja el valor original
# OJO: Esta línea ahora debe ir dentro del callback, ya que las columnas cambian
# df_table['LOCATION'] = df_table['LOCATION'].map(lambda x: flag_img(x) if x in country_flags else x)

#df['LOCATION'] = 'a'
density_heatmap = dcc.Graph(
    id='density-heatmap',
    style={'height': '30vh', 'borderRadius': '15px', 'overflow': 'hidden'},
    figure=create_density_heatmap(df_for_graphs)
)


histogram_irating = dcc.Graph(
    id='histogram-plot',
    style={'height': '40vh','borderRadius': '15px','overflow': 'hidden'},
    figure=create_histogram_with_percentiles(df, 'IRATING', 100)  # 100 = ancho de cada bin
)

# --- MODIFICACIÓN: Simplificamos la definición inicial de la tabla ---
# Las columnas se generarán dinámicamente en el callback
interactive_table = dash_table.DataTable(
    id='datatable-interactiva',
    # columns se define en el callback
    data=[],  # Inicialmente vacía
    sort_action="custom",
    sort_mode="single",
    page_action="custom",
    page_current=0,
    page_size=20,
    page_count=len(df_table) // 20 + (1 if len(df_table) % 20 > 0 else 0),
    active_cell={'row': 0, 'column': 1, 'column_id': 'DRIVER'},
    virtualization=False,
    style_as_list_view=False,
    
    # --- ELIMINAMOS selected_rows Y AÑADIMOS active_cell ---
    # selected_rows=[],  # <-- ELIMINAR ESTA LÍNEA
    
    style_table={
        'overflowX': 'auto',
        'height': '70vh',
        'minHeight': '0',
        'width': '100%',
        'borderRadius': '15px',
        'overflow': 'hidden',
        'backgroundColor': 'rgb(0, 0, 0)'
        
    },
    style_data={
            'textAlign': 'center',
            'fontWeight': 'bold',
            'fontSize': 12,},
    style_cell={
        'textAlign': 'center',
        'padding': '1px',
        'backgroundColor': 'rgb(0, 0, 0)',
        'color': 'rgb(255, 255, 255,1)',
        'border': '1px solid rgba(255, 255, 255, 0)',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    style_header={
        'backgroundColor': 'rgb(10, 10, 10)',
        'fontWeight': 'bold',
        'color': 'white',
        'border': 'none',
        'textAlign': 'center'
    },
    # --- AÑADIMOS ESTILO PARA LA FILA SELECCIONADA Y LAS CLASES ---
    style_data_conditional=[
        {
            'if': {'state': 'active'},
            'backgroundColor': 'rgba(0, 111, 255, 0.3)',
            'border': '1px solid rgb(0, 111, 255)'
        },
        # --- REGLAS MEJORADAS CON BORDES REDONDEADOS ---
        {'if': {'filter_query': '{CLASS} contains "P"','column_id': 'CLASS'}, 
         'backgroundColor': 'black', 'color': 'white', 'fontWeight': 'bold', 'border': '2px solid black', 'borderRadius': '10px'},
        
        {'if': {'filter_query': '{CLASS} contains "A"','column_id': 'CLASS'}, 
         'backgroundColor': 'blue', 'color': 'white', 'fontWeight': 'bold', 'border': '2px solid black', 'borderRadius': '10px'},
        
        {'if': {'filter_query': '{CLASS} contains "B"','column_id': 'CLASS'}, 
         'backgroundColor': 'green', 'color': 'white', 'fontWeight': 'bold', 'border': '2px solid black', 'borderRadius': '10px'},
        
        {'if': {'filter_query': '{CLASS} contains "C"','column_id': 'CLASS'}, 
         'backgroundColor': 'yellow', 'color': 'black', 'fontWeight': 'bold', 'border': '2px solid black', 'borderRadius': '10px'},
        
        {'if': {'filter_query': '{CLASS} contains "D"','column_id': 'CLASS'}, 
         'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold', 'border': '2px solid black', 'borderRadius': '10px'},
        
        {'if': {'filter_query': '{CLASS} contains "R"','column_id': 'CLASS'}, 
         'backgroundColor': 'red', 'color': 'white', 'fontWeight': 'bold', 'border': '2px solid black', 'borderRadius': '10px'},
    ],
    style_cell_conditional=[
        # --- AÑADIR ESTA LÍNEA PARA EL ANCHO DE LA NUEVA COLUMNA ---
        {'if': {'column_id': 'CLASS'}, 'width': '5%'},
        {'if': {'column_id': 'Rank World'},   'width': '10%'},
        {'if': {'column_id': 'Rank Region'},  'width': '10%'},
        {'if': {'column_id': 'Rank Country'},      'width': '10%'},
        {'if': {'column_id': 'DRIVER'},        'width': '25%','textAlign': 'left'},
        {'if': {'column_id': 'IRATING'},       'width': '15%'},
        {'if': {'column_id': 'LOCATION'},      'width': '10%'},
        {'if': {'column_id': 'WINS'},      'width': '10%'},
        {'if': {'column_id': 'STARTS'},      'width': '10%'},
        {'if': {'column_id': 'REGION'},      'width': '20%'},
    ]
     # Renderizado virtual
)

scatter_irating_starts = dcc.Graph(
    id='scatter-irating',
    style={'height': '30vh','borderRadius': '15px','overflow': 'hidden'},
    # Usamos go.Scattergl en lugar de px.scatter para un rendimiento masivo
    figure=go.Figure(data=go.Scattergl(
        x=df['IRATING'],
        y=df['STARTS'],
        mode='markers',
        marker=dict(
            color='rgba(0,111,255,.3)', # Color semitransparente
            size=5,
            line=dict(width=0)
        ),
        # Desactivamos el hover para máxima velocidad
        hoverinfo='none'
    )).update_layout(
        title='Relación iRating vs. Carreras Iniciadas',
        xaxis_title='iRating',
        yaxis_title='Carreras Iniciadas (Starts)',
        template='plotly_dark',
        paper_bgcolor='rgba(30,30,38,1)', # Fondo de toda la figura (transparente)
        plot_bgcolor='rgba(30,30,38,1)',  # Fondo del área de las barras (transparente)
        # --- FIN DE LAS LÍNEAS AÑADIDAS ---
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    ),
    # Hacemos el gráfico estático (no interactivo) para que sea aún más rápido
    config={'staticPlot': True}
)

correlation_heatmap = dcc.Graph(
    id='correlation-heatmap',
    style={'height': '70vh'}, # Ajusta la altura para que quepan los 3 gráficos
    # Usamos las columnas numéricas del dataframe original
    figure=create_correlation_heatmap(df[['IRATING', 'STARTS', 'WINS','TOP25PCNT','AVG_INC','AVG_FINISH_POS']])
)

continent_map = dcc.Graph(
    id='continent-map',
    style={'height': '50vh'},
    figure=create_continent_map(df_for_graphs)
)

# --- 3. Inicialización de la App ---
app = dash.Dash(__name__)

# Layout principal
app.layout = html.Div(
    style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        
        html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[ # Contenedor para título y filtros
            html.H1("Top iRating", style={'textAlign': 'left','font-size':60,'margin-left':'2%','margin-right':'50px'}),
            
            # Filtro por Región
            html.Div([
                html.Label("Filtro por Región:"),
                dcc.Dropdown(
                    id='region-filter',
                    options=[{'label': 'Todas', 'value': 'ALL'}] + 
                           [{'label': region, 'value': region} for region in sorted(iracing_ragions.keys())],
                    value='ALL',
                    style={'width': '200px', 'margin': '10px'},
                    className='iracing-dropdown'
                )
            ], style={'margin-right': '20px'}),

            # Filtro por País
            html.Div([
                html.Label("Filtro por País:"),
                dcc.Dropdown(
                    id='country-filter',
                    options=[{'label': 'Todos', 'value': 'ALL'}], # Se actualizará dinámicamente
                    value='ALL',
                    style={'width': '200px', 'margin': '10px'},
                    className='iracing-dropdown'
                )
            ], style={'margin-right': '20px'}),

            # --- MODIFICACIÓN: Cambiamos Input+Button por un Dropdown de búsqueda ---
            html.Div([
                html.Label("Buscar Piloto:"),
                dcc.Dropdown(
                    id='pilot-search-dropdown',
                    options=[],  # Las opciones se generan en el servidor
                    placeholder='Search Dirver...',
                    style={'width': '250px', 'margin': '10px'},
                    className='iracing-dropdown',
                    searchable=True,
                    clearable=True,
                    # --- PROPIEDAD CLAVE PARA BÚSQUEDA EN SERVIDOR ---
                    search_value='', # Captura el texto de búsqueda
                )
            ], style={'margin-left': '20px'})
            # --- FIN DEL BLOQUE MODIFICADO ---
        ]),
        html.Div(
            style={'display': 'flex', 'flex': 1, 'minHeight': 0},
            children=[
                # Columna Izquierda (Tabla) - Ancho ajustado
                html.Div(
                    style={
                        'width': '25%',
                        'height': '100%',
                        'padding': '10px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'borderRadius': '15px'
                    },
                    children=[
                        html.Div(interactive_table, style={'flex': 1,'borderRadius': '15px'})
                    ]
                ),
                # Columna Derecha (Gráficos) - Ancho ajustado
                html.Div(
                    style={
                        'width': '30%', 
                        'padding': '10px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'borderRadius': '15px'
                    },
                    children=[
                        histogram_irating,      # Gráfico de arriba
                        #scatter_irating_starts, # Gráfico del medio
                        density_heatmap
                        #correlation_heatmap,     # Gráfico de abajo (NUEVO)
                    ]
                ),
                html.Div(
                    style={
                        'width': '40%', 
                        'padding': '1%',
                        'display': 'flex',
                        'flexDirection': 'column'
                    },
                    children=[
                        html.Div(continent_map, style={'flex': '1.2'}) # Damos más peso al mapa
                    ]
                )
            ]
        ),
        # --- NUEVO: Componente para mostrar información del piloto seleccionado ---
        html.Div(
            id='pilot-info-display',
            style={
                'backgroundColor': 'rgb(10, 10, 10)',
                'color': 'white',
                'padding': '10px',
                'borderRadius': '10px',
                'margin': '10px',
                'fontSize': '14px',
                'display': 'none'  # Oculto por defecto
            }
        ),
        dcc.Store(id='shared-data-store', data={})
    ]
)
# --- 4. Callbacks ---

# NUEVO CALLBACK: Actualiza las opciones del filtro de país según la región seleccionada
@app.callback(
    Output('country-filter', 'options'),
    Input('region-filter', 'value')
)
def update_country_options(selected_region):
    if not selected_region or selected_region == 'ALL':
        # Si no hay región o es 'ALL', muestra todos los países
        countries = df['LOCATION'].dropna().unique()
    else:
        # Muestra solo los países de la región seleccionada
        countries = iracing_ragions.get(selected_region, [])
    
    options = [{'label': 'Todos', 'value': 'ALL'}] + \
              [{'label': country, 'value': country} for country in sorted(countries)]
    return options

@app.callback(
    Output('pilot-search-dropdown', 'options'),
    Input('pilot-search-dropdown', 'search_value'),
    State('pilot-search-dropdown', 'value'),  # <-- AÑADIMOS ESTADO para saber el piloto ya seleccionado
    State('region-filter', 'value'),
    State('country-filter', 'value'),
    prevent_initial_call=True,
)
def update_pilot_search_options(search_value, current_selected_pilot, region_filter, country_filter):
    # Si no hay texto de búsqueda, pero ya hay un piloto seleccionado,
    # nos aseguramos de que su opción esté disponible para que no desaparezca.
    if not search_value:
        if current_selected_pilot:
            return [{'label': current_selected_pilot, 'value': current_selected_pilot}]
        return []

    # Mantenemos la optimización de no buscar con texto muy corto
    if len(search_value) < 2:
        return []

    # 1. La lógica de filtrado no cambia
    if not region_filter: region_filter = 'ALL'
    if not country_filter: country_filter = 'ALL'

    filtered_df = df
    if region_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['REGION'] == region_filter]
    if country_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['LOCATION'] == country_filter]

    # 2. La búsqueda de coincidencias no cambia
    matches = filtered_df[filtered_df['DRIVER'].str.contains(search_value, case=False)]
    top_matches = matches.nlargest(20, 'IRATING')

    # 3. Creamos las opciones a partir de las coincidencias
    options = [{'label': row['DRIVER'], 'value': row['DRIVER']} 
               for _, row in top_matches.iterrows()]

    # 4. LA CLAVE: Si el piloto ya seleccionado no está en la nueva lista de opciones
    # (porque borramos el texto, por ejemplo), lo añadimos para que no se borre de la vista.
    if current_selected_pilot and not any(opt['value'] == current_selected_pilot for opt in options):
        options.insert(0, {'label': current_selected_pilot, 'value': current_selected_pilot})
    
    print(f"DEBUG: Búsqueda de '{search_value}' encontró {len(options)} coincidencias.")
    
    return options

# --- CALLBACK para limpiar la búsqueda si cambian los filtros ---
@app.callback(
    Output('pilot-search-dropdown', 'value'),
    Input('region-filter', 'value'),
    Input('country-filter', 'value'),
)
def clear_pilot_search_on_filter_change(region, country):
    # Cuando un filtro principal cambia, reseteamos la selección del piloto
    return None

# --- CALLBACK CONSOLIDADO: BÚSQUEDA Y TABLA ---
@app.callback(
    Output('datatable-interactiva', 'data'),
    Output('datatable-interactiva', 'page_count'),
    Output('datatable-interactiva', 'columns'),
    Output('datatable-interactiva', 'page_current'),
    Output('shared-data-store', 'data'),  # CAMBIAR: active_cell por data
    Output('histogram-plot', 'figure'),
    Output('continent-map', 'figure'),
    Input('region-filter', 'value'),
    Input('country-filter', 'value'),
    Input('pilot-search-dropdown', 'value'),
    Input('datatable-interactiva', 'page_current'),
    Input('datatable-interactiva', 'page_size'),
    Input('datatable-interactiva', 'sort_by')
)
def update_table_and_search(region_filter, country_filter, selected_pilot,
                           page_current, page_size, sort_by):
    
    # Detectar qué input activó el callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # --- LÓGICA PARA DETERMINAR LAS COLUMNAS A MOSTRAR ---
    # --- AÑADIR 'CLASS' A LA LISTA DE COLUMNAS BASE ---
    base_cols = ['DRIVER', 'IRATING', 'LOCATION', 'REGION', 'STARTS', 'WINS','CLASS']
    
    if country_filter and country_filter != 'ALL':
        dynamic_cols = ['Rank Country'] + base_cols
    elif region_filter and region_filter != 'ALL':
        dynamic_cols = ['Rank Region'] + base_cols
    else:
        dynamic_cols = ['Rank World'] + base_cols

    # Crear dataframe filtrado
    filtered_df = df[dynamic_cols].copy()
    filtered_df['LOCATION'] = filtered_df['LOCATION'].map(lambda x: flag_img(x) if x in country_flags else x)

    # --- FILTRADO POR REGIÓN Y PAÍS ---
    if not region_filter: region_filter = 'ALL'
    if not country_filter: country_filter = 'ALL'

    indices_to_keep = df.index
    if region_filter != 'ALL':
        indices_to_keep = indices_to_keep.intersection(df[df['REGION'] == region_filter].index)
    if country_filter != 'ALL':
        indices_to_keep = indices_to_keep.intersection(df[df['LOCATION'] == country_filter].index)

    filtered_df = filtered_df.loc[indices_to_keep]
    
    # --- ORDENAMIENTO ---
    if sort_by:
        sorting_df = df.loc[filtered_df.index]
        sorting_df = sorting_df.sort_values(
            by=sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc'
        )
        filtered_df = filtered_df.loc[sorting_df.index]

    # --- LÓGICA DE BÚSQUEDA Y NAVEGACIÓN ---
    target_page = page_current
    try:
        active_cell = active_cell
    except:
        active_cell = None
    
    # Solo ejecutar búsqueda si hay piloto seleccionado
    if selected_pilot:
        match = filtered_df[filtered_df['DRIVER'] == selected_pilot]
        if not match.empty:
            pilot_index = filtered_df.index.get_loc(match.index[0])
            target_page = pilot_index // page_size
            target_row_on_page = pilot_index % page_size
            
            driver_column_index = list(filtered_df.columns).index('DRIVER')
            
            active_cell = {
                'row': target_row_on_page,
                'column': driver_column_index,
                'column_id': 'DRIVER'
            }
            print(f"DEBUG: Piloto '{selected_pilot}' encontrado en página {target_page}, fila {target_row_on_page}, columna {driver_column_index}")

    # --- ESTABLECER active_cell POR DEFECTO SI NO HAY BÚSQUEDA ---
    if not active_cell:
        driver_column_index = list(filtered_df.columns).index('DRIVER')
        active_cell = {
            'row': 21,
            'column': driver_column_index,
            'column_id': 'DRIVER'
        }
        print(f"DEBUG: Estableciendo active_cell por defecto en fila 0")

    # --- GUARDAR active_cell EN EL STORE ---
    shared_data = {
        'active_cell': active_cell,
        'selected_pilot': selected_pilot or '',
        'timestamp': str(pd.Timestamp.now())
    }
    
    print(f"DEBUG: Guardando en store: {shared_data}")

    # --- GENERACIÓN DE COLUMNAS ---
    columns_definition = []
    for col_name in filtered_df.columns:
        if col_name == "LOCATION":
            columns_definition.append({"name": "Country", "id": col_name, "presentation": "markdown", 'type': 'text'})
        elif col_name == "IRATING":
            columns_definition.append({"name": "iRating", "id": col_name})
        elif col_name.startswith("Rank "):
            columns_definition.append({"name": col_name.replace("Rank ", ""), "id": col_name})
        elif col_name == "CLASS":
            columns_definition.append({"name": "SR", "id": col_name})
        else:
            columns_definition.append({"name": col_name.title(), "id": col_name})

    # --- PAGINACIÓN ---
    start_idx = target_page * page_size
    end_idx = start_idx + page_size
    page_data = filtered_df.iloc[start_idx:end_idx].to_dict('records')
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
    

    # --- GRÁFICOS ---
    graph_indices = filtered_df.index
    updated_histogram_figure = create_histogram_with_percentiles(df.loc[graph_indices], 'IRATING', 100)
    updated_map_figure = create_continent_map(df_for_graphs, region_filter, country_filter)
    
    return (page_data, total_pages, columns_definition, target_page, 
            shared_data, updated_histogram_figure, updated_map_figure)

# --- NUEVO CALLBACK: IMPRIMIR DATOS DEL PILOTO SELECCIONADO ---
@app.callback(
    Output('pilot-info-display', 'children'),  # Necesitarás añadir este componente al layout
    Input('datatable-interactiva', 'active_cell'),
    State('datatable-interactiva', 'data'),
    State('region-filter', 'value'),
    State('country-filter', 'value'),
    prevent_initial_call=True
)
def print_selected_pilot_data(active_cell, table_data, region_filter, country_filter):
    if not active_cell or not table_data:
        return "No hay piloto seleccionado"
    
    # Obtener el nombre del piloto de la fila seleccionada
    selected_row = active_cell['row']
    if selected_row >= len(table_data):
        return "Fila no válida"
    
    pilot_name = table_data[selected_row]['DRIVER']
    
    # Buscar todos los datos del piloto en el DataFrame original
    pilot_data = df[df['DRIVER'] == pilot_name]
    
    if pilot_data.empty:
        return f"No se encontraron datos para {pilot_name}"
    
    # Obtener la primera (y única) fila del piloto
    pilot_info = pilot_data.iloc[0]
    
    # IMPRIMIR EN CONSOLA todos los datos del piloto
    print("\n" + "="*50)
    print(f"DATOS DEL PILOTO SELECCIONADO: {pilot_name}")
    print("="*50)
    for column, value in pilot_info.items():
        print(f"{column}: {value}")
    print("="*50 + "\n")
    
    # También retornar información para mostrar en la interfaz (opcional)
    return f"Piloto seleccionado: {pilot_name} (Ver consola para datos completos)"


@app.callback(
    Output('datatable-interactiva', 'active_cell'),
    Input('shared-data-store', 'data'),
    prevent_initial_call=True
)
def update_active_cell_from_store(shared_data):
    if not shared_data:
        return None
    
    active_cell = shared_data.get('active_cell')
    selected_pilot = shared_data.get('selected_pilot', '')
    
    print(f"DEBUG: Recuperando active_cell del store: {active_cell}")
    print(f"DEBUG: Piloto asociado: {selected_pilot}")
    
    return active_cell

if __name__ == "__main__":
    app.run(debug=True)