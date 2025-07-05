#What is what i want to see on the dashboard

#better what is what i want to know with this data

#1 where % in the top world, region or country i am
#2 who are the top iracers on world, region , country
# comparations of countrys (players, level)
#comparations of regions (player levels)
#Top 20 of 


import dash
from dash import html, dcc, dash_table, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pycountry_convert as pc
import pycountry

def create_irating_starts_scatter(df):
    """
    Crea un gráfico de dispersión optimizado para mostrar la relación
    entre iRating y las carreras iniciadas.
    """
    fig = go.Figure(data=go.Scattergl(
        x=df['IRATING'],
        y=df['STARTS'],
        mode='markers',
        marker=dict(
            color='rgba(0, 111, 255, 0.3)', # Color azul semitransparente para ver la densidad
            size=5,
            line=dict(width=0) # Sin borde en los puntos
        ),
        # Desactivamos el hover para máxima velocidad, ya que son demasiados puntos
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title='Relación iRating vs. Carreras Iniciadas',
        xaxis_title='iRating',
        yaxis_title='Carreras Iniciadas (Starts)',
        template='plotly_dark',
        paper_bgcolor='rgba(11,11,19,1)',
        plot_bgcolor='rgba(11,11,19,1)',
        # Estilo de la cuadrícula igual al histograma y bubble chart
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig


def create_irating_trend_line_chart(df):
    """
    Crea un gráfico de líneas que muestra el promedio de carreras corridas
    para diferentes rangos de iRating, eliminando valores atípicos.
    """
    # --- 1. Eliminar Outliers ---
    # Calculamos el percentil 99 para las carreras y filtramos para
    # que unos pocos pilotos con miles de carreras no desvíen el promedio.
    max_starts = df['STARTS'].quantile(0.95)
    min_starts = df['STARTS'].quantile(0.001)
    print(max_starts)
    print(min_starts)
    df_filtered = df[df['STARTS'] <= max_starts]
    df_filtered = df[df['STARTS'] >= min_starts]

    # --- 2. Agrupar iRating en Bins ---
    # Creamos los rangos de 0 a 11000, en pasos de 1000.
    bins = list(range(0, 12000, 1000))
    labels = [f'{i/1000:.0f}k - {i/1000+1-0.001:.1f}k' for i in bins[:-1]]
    
    df_filtered['irating_bin'] = pd.cut(df_filtered['IRATING'], bins=bins, labels=labels, right=False)

    # --- 3. Calcular el promedio de carreras por bin ---
    trend_data = df_filtered.groupby('irating_bin').agg(
        avg_starts=('STARTS', 'mean'),
        num_pilots=('DRIVER', 'count')
    ).reset_index()

    # --- 4. Crear el Gráfico ---
    fig = go.Figure(data=go.Scatter(
        x=trend_data['irating_bin'],
        y=trend_data['avg_starts'],
        mode='lines+markers', # Líneas y puntos en cada dato
        marker=dict(
            color='rgba(0, 111, 255, 1)',
            size=8,
            line=dict(width=1, color='white')
        ),
        line=dict(color='rgba(0, 111, 255, 0.7)'),
        
        customdata=trend_data['num_pilots'],
        hovertemplate=(
            "<b>Rango iRating:</b> %{x}<br>" +
            "<b>Promedio Carreras:</b> %{y:.0f}<br>" +
            "<b>Pilotos en este rango:</b> %{customdata:,}<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title='Tendencia: Carreras Promedio por Nivel de iRating',
        xaxis_title='Rango de iRating',
        yaxis_title='Promedio de Carreras Corridas',
        template='plotly_dark',
        paper_bgcolor='rgba(11,11,19,1)',
        plot_bgcolor='rgba(11,11,19,1)',
        font=GLOBAL_FONT,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def calculate_competitiveness(df):
    """
    Calcula el iRating promedio de los 100 mejores pilotos para cada región y país.
    Descarta grupos con menos de 100 pilotos.
    """
    # --- Cálculo para Regiones ---
    # Filtramos regiones con al menos 100 pilotos
    region_counts = df['REGION'].value_counts()
    valid_regions = region_counts[region_counts >= 100].index
    
    region_scores = {}
    for region in valid_regions:
        # Tomamos el top 100 por iRating y calculamos el promedio
        top_100 = df[df['REGION'] == region].nlargest(100, 'IRATING')
        region_scores[region] = top_100['IRATING'].mean()

    # Convertimos a DataFrame, ordenamos y tomamos el top 10
    top_regions_df = pd.DataFrame(list(region_scores.items()), columns=['REGION', 'avg_irating'])
    top_regions_df = top_regions_df.sort_values('avg_irating', ascending=False)

    # --- Cálculo para Países ---
    # Mismo proceso para países
    country_counts = df['LOCATION'].value_counts()
    valid_countries = country_counts[country_counts >= 100].index
    
    country_scores = {}
    for country in valid_countries:
        top_100 = df[df['LOCATION'] == country].nlargest(100, 'IRATING')
        country_scores[country] = top_100['IRATING'].mean()

    top_countries_df = pd.DataFrame(list(country_scores.items()), columns=['LOCATION', 'avg_irating'])
    top_countries_df = top_countries_df.sort_values('avg_irating', ascending=False)

    return top_regions_df, top_countries_df

def create_competitiveness_table(top_regions_df, top_countries_df):
    """
    Crea una figura de Plotly con dos tablas para el top 10 de regiones y países.
    """
    fig = go.Figure()

    # --- Tabla de Regiones (Izquierda) ---
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>#</b>', '<b>Top 10 Regiones</b>', '<b>iRating Promedio</b>'],
            fill_color='#1E1E1E',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                list(range(1, 11)),
                top_regions_df['REGION'],
                top_regions_df['avg_irating'].round(0).astype(int)
            ],
            fill_color='#323232',
            align=['center', 'left', 'center'],
            font=dict(color='white', size=11)
        ),
        domain=dict(x=[0, 0.48], y=[0, 1]) # Ocupa la mitad izquierda
    ))

    # --- Tabla de Países (Derecha) ---
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>#</b>', '<b>Top 10 Países</b>', '<b>iRating Promedio</b>'],
            fill_color='#1E1E1E',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                list(range(1, 11)),
                top_countries_df['LOCATION'],
                top_countries_df['avg_irating'].round(0).astype(int)
            ],
            fill_color='#323232',
            align=['center', 'left', 'center'],
            font=dict(color='white', size=11)
        ),
        domain=dict(x=[0.52, 1], y=[0, 1]) # Ocupa la mitad derecha
    ))

    fig.update_layout(
        title_text='🏆 Ranking de Competitividad (Promedio del Top 100)',
        template='plotly_dark',
        paper_bgcolor='#323232',
        plot_bgcolor='#323232',
        margin=dict(l=10, r=10, t=40, b=10),
        height=300
    )
    return fig

def create_region_bubble_chart(df):
    # --- 1. DICCIONARIO DE COLORES POR REGIÓN (¡Personalízalo aquí!) ---
    region_color_map = {
        'US': '#0047AB',
        'Iberia': '#FFC400',
        'UK & I': '#C8102E',
        'Brazil': '#009B3A',
        'Australia & NZ': '#00008B',
        'DE-AT-CH': '#FFCE00',
        'Benelux': '#AE1C28',
        'France': '#0055A4',
        'Scandinavia': '#0065BD',
        'Central EU': '#DC143C',
        'Italy': '#009246',
        'Japan': '#BC002D',
        'Canada': '#FF0000',
        # --- Color por defecto para el resto ---
        'default': '#808080' 
    }

    # 2. Agrupación y cálculo de estadísticas (sin cambios)
    df = df[df['REGION'] != 'Atlantic']
    region_stats = df.groupby('REGION').agg(
        avg_starts=('STARTS', 'mean'),
        avg_irating=('IRATING', 'mean'),
        num_pilots=('DRIVER', 'count')
    ).reset_index()

    region_stats = region_stats.sort_values('num_pilots', ascending=True)
    hover_text_pilots = (region_stats['num_pilots'] / 1000).round(1).astype(str) + 'k'

    # --- 3. Asignar colores a cada región usando el diccionario ---
    region_colors = region_stats['REGION'].map(region_color_map).fillna(region_color_map['default'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=region_stats['avg_irating'],
        y=region_stats['avg_starts'],
        mode='markers+text',
        text=region_stats['REGION'],
        textposition='top center',
        textfont=dict(size=10, color='rgba(255, 255, 255, 0.8)'),

        # --- 4. MODIFICACIONES CLAVE EN EL MARCADOR ---
        marker=dict(
            size=region_stats['num_pilots'],
            sizemode='area',
            sizeref=2.*max(region_stats['num_pilots'])/(50.**2.3), 
            sizemin=6,
            color=region_colors,         # <-- Usamos los colores del diccionario
            showscale=False              # <-- Ocultamos la barra de color
        ),
        # --- El hover no necesita cambios ---
        customdata=np.stack((hover_text_pilots, region_stats['num_pilots']), axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Promedio iRating: %{x:.0f}<br>" +
            "Promedio Carreras: %{y:.1f}<br>" +
            "Cantidad de Pilotos: %{customdata[0]} (%{customdata[1]:,})<extra></extra>"
        )
    ))

    fig.update_layout(
        title='Regiones: iRating, Carreras y Cantidad de Pilotos',
        xaxis_title='iRating Promedio',
        yaxis_title='Promedio de Carreras Corridas',
        template='plotly_dark',
        
        paper_bgcolor='rgba(11,11,19,1)',
        plot_bgcolor='rgba(11,11,19,1)',
        # --- AÑADIMOS ESTILO DE GRID IGUAL AL HISTOGRAMA ---
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        # --- FIN DEL ESTILO DE GRID ---
        margin=dict(l=20, r=20, t=50, b=20),
        
        height=350
    )
    return fig

def create_kpi_global(filtered_df, filter_context="Mundo"):
    total_pilots = len(filtered_df)
    avg_irating = filtered_df['IRATING'].mean() if total_pilots > 0 else 0
    avg_starts = filtered_df['STARTS'].mean() if total_pilots > 0 else 0
    avg_wins = filtered_df['WINS'].mean() if total_pilots > 0 else 0

    fig = go.Figure()
    kpis = [
        {'value': total_pilots, 'title': f"Pilotos en {filter_context}", 'format': ',.0f'},
        {'value': avg_irating, 'title': "iRating Promedio", 'format': ',.0f'},
        {'value': avg_starts, 'title': "Carreras Promedio", 'format': '.1f'},
        {'value': avg_wins, 'title': "Victorias Promedio", 'format': '.2f'}
    ]
    for i, kpi in enumerate(kpis):
        fig.add_trace(go.Indicator(
            mode="number",
            value=kpi['value'],
            number={'valueformat': kpi['format'], 'font': {'size': 28}},
            title={"text": kpi['title'], 'font': {'size': 14}},
            domain={'row': 0, 'column': i}
        ))
    fig.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#323232',
        margin=dict(l=20, r=20, t=30, b=10),
        height=60
    )
    return fig


def create_kpi_pilot(filtered_df, pilot_info=None, filter_context="Mundo"):
    fig = go.Figure()
    
    title_text = "Seleccione un Piloto"
    
    # Si NO hay información del piloto, creamos una figura vacía y ocultamos los ejes.
    if pilot_info is None:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_visible=False,  # <-- OCULTA EL EJE X
            yaxis_visible=False,  # <-- OCULTA EL EJE Y
            height=10
        )
        return fig

    # Si SÍ hay información del piloto, procedemos como antes.
    pilot_name = pilot_info.get('DRIVER', 'Piloto')
    title_text = f"<b>{pilot_name}</b>"

    rank_world = pilot_info.get('Rank World', 0)
    rank_region = pilot_info.get('Rank Region', 0)
    rank_country = pilot_info.get('Rank Country', 0)
    percentil_world = (1 - (rank_world / len(df))) * 100 if len(df) > 0 else 0
    region_df = df[df['REGION'] == pilot_info.get('REGION')]
    percentil_region = (1 - (rank_region / len(region_df))) * 100 if len(region_df) > 0 else 0
    country_df = df[df['LOCATION'] == pilot_info.get('LOCATION')]
    percentil_country = (1 - (rank_country / len(country_df))) * 100 if len(country_df) > 0 else 0
    
    kpis_piloto = [
        {'rank': rank_world, 'percentil': percentil_world, 'title': "World Rank"},
        {'rank': rank_region, 'percentil': percentil_region, 'title': "Region Rank "},
        {'rank': rank_country, 'percentil': percentil_country, 'title': "Country Rank"}
    ]

    for i, kpi in enumerate(kpis_piloto):
        fig.add_trace(go.Indicator(
            mode="number",
            value=kpi['rank'],
            number={'prefix': "#", 'font': {'size': 18}},
            # Eliminamos el <br> y ajustamos el texto para que esté en una línea
            title={"text": f"{kpi['title']} <span style='font-size:0.8em;color:gray'>(Top {100-kpi['percentil']:.2f}%)</span>", 'font': {'size': 14}},
            domain={'row': 0, 'column': i}
        ))
    fig.update_layout(
        title={
            'text': title_text,
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 14}
        },
        grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=10),
        height=80
    )
    return fig

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
        color_scale = [[0.0, "#000000"], [0.01, "#122753"], [0.3, "#238ED6"], [0.6, "#1B2FA5"],[1.0, "rgba(255,255,255,1)"]]
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
        template='plotly_dark',
        # --- BLOQUE MODIFICADO ---
        paper_bgcolor='rgba(0,0,0,0)', # Fondo de toda la figura (transparente)
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo del área del mapa (transparente)
        # --- FIN DE LA MODIFICACIÓN ---
        geo=dict(
            bgcolor='rgba(0,0,0,0)',    # Fondo específico del globo (transparente)
            lakecolor='#4E5D6C', 
            landcolor='#323232', 
            subunitcolor='rgba(0,0,0,0)',
            showframe=False,      # <-- Oculta el marco exterior del globo
            showcoastlines=False  # <-- Oculta las líneas de la costa
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
        coloraxis_showscale=show_scale,
        coloraxis_colorbar=dict(
            title='Pilotos',
            orientation='h',
            yanchor='bottom',
            y=-0.05,
            xanchor='center',
            x=0.5,
            len=0.5,
            thickness=10
        )
    )
    return fig

def create_kpi_indicators(filtered_df, pilot_info=None, filter_context="Mundo"):
    """
    Crea un panel de indicadores (KPIs) para datos generales y de un piloto específico.
    """
    fig = go.Figure()

    # --- FILA 1: DATOS GENERALES (DEL DATAFRAME FILTRADO) ---
    total_pilots = len(filtered_df)
    avg_irating = filtered_df['IRATING'].mean() if total_pilots > 0 else 0
    avg_starts = filtered_df['STARTS'].mean() if total_pilots > 0 else 0
    avg_wins = filtered_df['WINS'].mean() if total_pilots > 0 else 0

    kpis_generales = [
        {'value': total_pilots, 'title': f"Pilotos en {filter_context}", 'format': ',.0f'},
        {'value': avg_irating, 'title': "iRating Promedio", 'format': ',.0f'},
        {'value': avg_starts, 'title': "Carreras Promedio", 'format': '.1f'},
        {'value': avg_wins, 'title': "Victorias Promedio", 'format': '.2f'}
    ]

    for i, kpi in enumerate(kpis_generales):
        fig.add_trace(go.Indicator(
            mode="number",
            value=kpi['value'],
            number={'valueformat': kpi['format'], 'font': {'size': 35}},
            title={"text": kpi['title'], 'font': {'size': 14}},
            domain={'row': 0, 'column': i}
        ))

    # --- FILA 2: DATOS DEL PILOTO SELECCIONADO ---
    if pilot_info is not None:
        rank_world = pilot_info.get('Rank World', 0)
        rank_region = pilot_info.get('Rank Region', 0)
        rank_country = pilot_info.get('Rank Country', 0)
        
        # Calculamos el percentil
        percentil_world = (1 - (rank_world / len(df))) * 100 if len(df) > 0 else 0
        
        region_df = df[df['REGION'] == pilot_info.get('REGION')]
        percentil_region = (1 - (rank_region / len(region_df))) * 100 if len(region_df) > 0 else 0
        
        country_df = df[df['LOCATION'] == pilot_info.get('LOCATION')]
        percentil_country = (1 - (rank_country / len(country_df))) * 100 if len(country_df) > 0 else 0

        kpis_piloto = [
            {'rank': rank_world, 'percentil': percentil_world, 'title': "Rank Mundial"},
            {'rank': rank_region, 'percentil': percentil_region, 'title': "Rank Región"},
            {'rank': rank_country, 'percentil': percentil_country, 'title': "Rank País"}
        ]

        for i, kpi in enumerate(kpis_piloto):
             fig.add_trace(go.Indicator(
                mode="number",
                value=kpi['rank'],
                number={'prefix': "#", 'font': {'size': 30}},
                title={"text": f"{kpi['title']}<br><span style='font-size:0.8em;color:gray'>Top {100-kpi['percentil']:.2f}%</span>", 'font': {'size': 14}},
                domain={'row': 1, 'column': i+1} # Centramos los 3 KPIs del piloto
            ))

    # --- DISEÑO GENERAL DEL GRÁFICO ---
    title_text = f"Estadísticas Globales ({filter_context})"
    if pilot_info is not None:
        title_text = f"<b>{pilot_info['DRIVER']}</b> vs. Global ({filter_context})"

    fig.update_layout(
        grid={'rows': 2, 'columns': 4, 'pattern': "independent"},
        template='plotly_dark',
        paper_bgcolor='#323232',
        plot_bgcolor='#323232',
        title={
            'text': title_text,
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- MODIFICACIÓN: Añadimos parámetros para señalar a un piloto ---
def create_histogram_with_percentiles(df, column='IRATING', bin_width=100, highlight_irating=None, highlight_name=None):
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
            hist_transformed.append(hist[i] + 2)
        else:
            hist_transformed.append(hist[i])
    
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
    
    # --- NUEVO: Lógica para añadir la línea de señalización ---
    if highlight_irating is not None and highlight_name is not None:
        fig.add_vline(
            x=highlight_irating, 
            line_width=.5, 
            line_dash="dot", 
            annotation_position="top left", # Mejor posición para texto vertical
            annotation_textangle=-90, # <-- AÑADE ESTA LÍNEA PARA ROTAR EL TEXTO
            line_color="white",
            annotation_text=f"<b>{highlight_name}</b>",
            annotation_font_size=10,
            annotation_font_color="white"
        )
    # --- FIN DEL BLOQUE NUEVO ---

    fig.update_layout(
        title='Distribución de iRating',
        xaxis_title='iRating',
        yaxis_title='Pilotos',
        template='plotly_dark',
        hovermode='x unified',
        paper_bgcolor='rgba(18,18,26,.5)',
        plot_bgcolor='rgba(255,255,255,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)'),
        
        # --- MODIFICACIÓN: Reducir márgenes y tamaño de fuentes ---
        margin=dict(l=40, r=20, t=40, b=30), # Reduce los márgenes (izquierda, derecha, arriba, abajo)
        title_font_size=16,                  # Reduce el tamaño del título principal
        xaxis_title_font_size=12,            # Reduce el tamaño del título del eje X
        yaxis_title_font_size=12             # Reduce el tamaño del título del eje Y
        # --- FIN DE LA MODIFICACIÓN ---
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

GLOBAL_FONT = {'family': "Lato, sans-serif"}

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
    'Central EU':['PL','CZ','SK','HU','SI','HR','RS','ME','AL','RO','MD','UA','BY','EE','LV','LT'],
    'Finland':['FI'],
    'DE-AT-CH':['CH','AT','DE'], # <-- CORRECCIÓN: Eliminado el ''
    'Scandinavia':['DK','SE','NO'],
    'Australia & NZ':['AU','NZ'],
    'Asia':['SA','JO','IQ','YE','OM','AE','QA','IN','PK','AF','NP','BD','MM','TH','KH','VN','MY','ID','CN','PH','KR','MN','KZ','KG','UZ','TJ','AF','TM','LK'],
    'Benelux':['NL','BE','LU']
}

# --- 1. Carga y Preparación de Datos ---
df = pd.read_csv('ROAD.csv')
df = df[df['IRATING'] > 1]
df = df[df['STARTS'] > 1]
#df = df[df['STARTS'] < 2000]
df = df[df['CLASS'].str.contains('D|C|B|A|P|R', na=False)]
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

# Aplica solo el emoji/código si está in country_flags, si no deja el valor original
# OJO: Esta línea ahora debe ir dentro del callback, ya que las columnas cambian
# df_table['LOCATION'] = df_table['LOCATION'].map(lambda x: flag_img(x) if x in country_flags else x)

#df['LOCATION'] = 'a'
density_heatmap = dcc.Graph(
    id='density-heatmap',
    style={'height': '30vh', 'borderRadius': '15px', 'overflow': 'hidden'},
    figure=create_density_heatmap(df_for_graphs)
)


kpi_global = dcc.Graph(id='kpi-global', style={'height': '3hv', 'marginBottom': '0px'})
kpi_pilot = dcc.Graph(id='kpi-pilot', style={'height': '3hv', 'marginBottom': '10px'})

histogram_irating = dcc.Graph(
    id='histogram-plot',
    # --- MODIFICACIÓN: Ajustamos el estilo del contenedor del gráfico ---
    style={
        'height': '25vh',
        'borderRadius': '10px', # Coincide con el radio de los filtros
        'border': '1px solid #4A4A4A', # Coincide con el borde de los filtros
        'overflow': 'hidden'
    },
    # --- FIN DE LA MODIFICACIÓN ---
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
    virtualization=False,
    style_as_list_view=True,
    active_cell={'row': 21,'column':1},
    
    
    # --- ELIMINAMOS selected_rows Y AÑADIMOS active_cell ---
    # selected_rows=[],  # <-- ELIMINAR ESTA LÍNEA
    
    style_table={
        #'tableLayout': 'fixed', # <-- DESCOMENTA O AÑADE ESTA LÍNEA
        'overflowX': 'auto',
        'height': '70vh',
        'minHeight': '0',
        'width': '100%',
        'borderRadius': '15px',
        'overflow': 'hidden',
        'backgroundColor': 'rgba(11,11,19,1)',
        'textOverflow': 'ellipsis',
        'border': '1px solid #4A4A4A'
        
    },
    
    style_cell={
        'textAlign': 'center',
        'padding': '1px',
        'backgroundColor': 'rgba(11,11,19,1)',
        'color': 'rgb(255, 255, 255,.8)',
        'border': '1px solid rgba(255, 255, 255, 0)',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'whiteSpace': 'nowrap', # <-- AÑADE ESTA LÍNEA
        'maxWidth': 0
    },
    style_header={
        'backgroundColor': 'rgba(30,30,38,1)',
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
            'border': '1px solid rgba(0, 111, 255)'
        },
        {
            'if': {'state': 'selected'},
            'backgroundColor': 'rgba(0, 111, 255, 0)',
            'border': '1px solid rgba(0, 111, 255,0)'
        },
        # --- REGLAS MEJORADAS CON BORDES REDONDEADOS ---
        {'if': {'filter_query': '{CLASS} contains "P"','column_id': 'CLASS'}, 
         'backgroundColor': 'rgba(54,54,62,255)', 'color': 'rgba(166,167,171,255)', 'fontWeight': 'bold','border': '1px solid rgba(134,134,142,255)'},
        
        {'if': {'filter_query': '{CLASS} contains "A"','column_id': 'CLASS'}, 
         'backgroundColor': 'rgba(0,42,102,255)', 'color': 'rgba(107,163,238,255)', 'fontWeight': 'bold','border': '1px solid rgba(35,104,195,255)'},
        
        {'if': {'filter_query': '{CLASS} contains "B"','column_id': 'CLASS'}, 
         'backgroundColor': 'rgba(24,84,14,255)', 'color': 'rgba(139,224,105,255)', 'fontWeight': 'bold','border': '1px solid rgba(126,228,103,255)'},
        
        {'if': {'filter_query': '{CLASS} contains "C"','column_id': 'CLASS'}, 
         'backgroundColor': 'rgba(81,67,6,255)', 'color': 'rgba(224,204,109,255)', 'fontWeight': 'bold','border': '1px solid rgba(220,193,76,255)'},
        
        {'if': {'filter_query': '{CLASS} contains "D"','column_id': 'CLASS'}, 
         'backgroundColor': 'rgba(102,40,3,255)', 'color': 'rgba(255,165,105,255)', 'fontWeight': 'bold','border': '1px solid rgba(208,113,55,255)'},
        
        {'if': {'filter_query': '{CLASS} contains "R"','column_id': 'CLASS'}, 
         'backgroundColor': 'rgba(91,19,20,255)', 'color': 'rgba(225,125,123,255)', 'fontWeight': 'bold','border': '1px solid rgba(172,62,61,255)'},
    ],
    # --- MODIFICACIÓN: Forzamos el ancho de las columnas ---
    style_cell_conditional=[
        {'if': {'column_id': 'CLASS'},        'width': '5%', 'minWidth': '5%', 'maxWidth': '5%'},
        {'if': {'column_id': 'Rank World'},   'width': '10%', 'minWidth': '10%', 'maxWidth': '10%'},
        {'if': {'column_id': 'Rank Region'},  'width': '10%', 'minWidth': '10%', 'maxWidth': '10%'},
        {'if': {'column_id': 'Rank Country'}, 'width': '10%', 'minWidth': '10%', 'maxWidth': '10%'},
        {'if': {'column_id': 'DRIVER'},       'width': '30%', 'minWidth': '30%', 'maxWidth': '30%', 'textAlign': 'left'},
        {'if': {'column_id': 'IRATING'},      'width': '10%', 'minWidth': '10%', 'maxWidth': '10%'},
        {'if': {'column_id': 'LOCATION'},     'width': '10%', 'minWidth': '10%', 'maxWidth': '10%'},
        {'if': {'column_id': 'WINS'},         'width': '5%', 'minWidth': '5%', 'maxWidth': '5%'},
        {'if': {'column_id': 'STARTS'},       'width': '5%', 'minWidth': '5%', 'maxWidth': '5%'},
        {'if': {'column_id': 'REGION'},       'width': '15%', 'minWidth': '15%', 'maxWidth': '15%'},
    ],
    style_data={
            'whiteSpace':'normal',
            'textAlign': 'center',
            'fontSize': 12,},
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
    style={'height': '60vh'},
    figure=create_continent_map(df_for_graphs)
)

region_bubble_chart = dcc.Graph(
    id='region-bubble-chart',
    style={'height': '350px', 'marginTop': '10px','borderRadius': '10px','border': '1px solid #4A4A4A',
        'overflow': 'hidden'},
    figure=create_region_bubble_chart(df)
)

# --- NUEVO: Gráfico de dispersión iRating vs Starts ---
irating_starts_scatter = dcc.Graph(
    id='irating-starts-scatter',
    # Copiamos el estilo del bubble chart para que sea consistente
    style={'height': '350px', 'marginTop': '10px', 'borderRadius': '10px', 'border': '1px solid #4A4A4A', 'overflow': 'hidden'},
    figure=create_irating_trend_line_chart(df), # <-- MODIFICAMOS ESTA LÍNEA
    # Ya no es necesario que sea estático, la interacción es útil aquí
    config={'displayModeBar': False}
)

# --- MODIFICACIÓN: Reemplazamos el dcc.Graph por dos dash_table.DataTable ---
# Primero, preparamos los datos y las columnas para las tablas
top_regions, top_countries = calculate_competitiveness(df)

# Añadimos la columna de ranking '#' a cada dataframe
top_regions.insert(0, '#', range(1, 1 + len(top_regions)))
top_countries.insert(0, '#', range(1, 1 + len(top_countries)))

# Definimos un estilo base para ambas tablas, copiado de la tabla principal
table_style_base = {
    'style_table': {
        'borderRadius': '10px',
        'overflow': 'hidden',
        'border': '1px solid #4A4A4A'
    },
    'style_cell': {
        'textAlign': 'center',
        'padding': '0px',
        'backgroundColor': 'rgba(11,11,19,1)',
        'color': 'rgb(255, 255, 255,.8)',
        'border': 'none',
        'font_size': '12px',
        # --- ELIMINAMOS LA ALTURA FIJA DE LA CELDA Y DE LA TABLA ---
    },
    'style_header': {
        'backgroundColor': 'rgba(30,30,38,1)',
        'fontWeight': 'bold',
        'color': 'white',
        'border': 'none',
        'textAlign': 'center'
    },
    'style_cell_conditional': [
        {'if': {'column_id': '#'}, 'width': '10%', 'textAlign': 'center'},
        {'if': {'column_id': 'REGION'}, 'width': '50%', 'textAlign': 'left'},
        {'if': {'column_id': 'LOCATION'}, 'width': '50%', 'textAlign': 'left'},
        {'if': {'column_id': 'avg_irating'}, 'width': '40%', 'textAlign': 'center'},
    ]
}

# Creamos el contenedor principal que pondrá las tablas una al lado de la otra
competitiveness_tables_container = html.Div(
    style={'display': 'flex', 'gap': '2%', 'marginTop': '10px'},
    children=[
        # Tabla de Regiones
        html.Div(
            dash_table.DataTable(
                columns=[
                    {'name': '#', 'id': '#'},
                    {'name': 'Top Regiones', 'id': 'REGION'},
                    {'name': 'iRating Promedio', 'id': 'avg_irating'}
                ],
                data=top_regions.to_dict('records'),
                virtualization=False,
                page_action='none',
                fixed_rows={'headers': True},
                # --- MODIFICACIÓN: Combinamos los diccionarios de estilo ---
                style_table={**table_style_base['style_table'], 'height': '20vh'},
                style_cell=table_style_base['style_cell'],
                style_header=table_style_base['style_header'],
                style_cell_conditional=table_style_base['style_cell_conditional']
            ),
            style={'width': '49%'}
        ),
        # Tabla de Países
        html.Div(
            dash_table.DataTable(
                columns=[
                    {'name': '#', 'id': '#'},
                    {'name': 'Top Países', 'id': 'LOCATION'},
                    {'name': 'iRating Promedio', 'id': 'avg_irating'}
                ],
                data=top_countries.to_dict('records'),
                virtualization=False,
                page_action='none',
                fixed_rows={'headers': True},
                # --- MODIFICACIÓN: Combinamos los diccionarios de estilo ---
                style_table={**table_style_base['style_table'], 'height': '20vh'},
                style_cell=table_style_base['style_cell'],
                style_header=table_style_base['style_header'],
                style_cell_conditional=table_style_base['style_cell_conditional']
            ),
            style={'width': '49%'}
        )
    ]
)
# --- FIN DE LA MODIFICACIÓN ---

# --- 3. Inicialización de la App ---
app = dash.Dash(__name__)

# Layout principal
app.layout = html.Div(
    #style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column', 'backgroundColor': '#1E1E1E'},
    style={},
    children=[
        
        # --- BARRA SUPERIOR: TÍTULO Y BOTONES DE TIPO DE DASHBOARD ---
        html.Div(
            style={'padding': '10px 20px', 'textAlign': 'center'},
            children=[
                html.H1("Top iRating", style={'fontSize': 48, 'color': 'white', 'margin': '0 0 10px 0'}),
                html.Div([
                     # <-- AÑADIDO
                    html.Button('Sports Car', id='btn-road', n_clicks=0, className='dashboard-type-button'),
                    html.Button('Formula', id='btn-formula', n_clicks=0, className='dashboard-type-button'),
                    html.Button('Oval', id='btn-oval', n_clicks=0, className='dashboard-type-button'),
                    html.Button('Dirt Road', id='btn-dirt-road', n_clicks=0, className='dashboard-type-button'),
                    html.Button('Dirt Oval', id='btn-dirt-oval', n_clicks=0, className='dashboard-type-button'),
                ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px'})
            ]
        ),
        
        # --- CONTENEDOR PRINCIPAL CON 3 COLUMNAS ---
        html.Div(
            id='main-content-container',
            #style={'display': 'flex', 'flex': 1, 'minHeight': 0, 'padding': '0 10px 10px 10px'},
            style={'display': 'flex', 'padding': '0 10px 10px 10px'},
            children=[
                
                # --- COLUMNA IZQUIERDA (FILTROS Y TABLA) ---
                html.Div(
                    id='left-column',
                    style={'width': '25%', 'padding': '1%', 'display': 'flex', 'flexDirection': 'column'},
                    children=[
                        # Contenedor de Filtros
                        html.Div(
                            style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'},
                            children=[
                                html.Div([
                                    html.Label("Filtro por Región:", style={'color': 'white', 'fontSize': 12}),
                                    dcc.Dropdown(
                                        id='region-filter',
                                        options=[{'label': 'Todas', 'value': 'ALL'}] + 
                                               [{'label': region, 'value': region} for region in sorted(iracing_ragions.keys())],
                                        value='ALL',
                                        className='iracing-dropdown',
                                        # --- AÑADIMOS ESTILO INICIAL ---
                                        style={'width': '100%'}
                                    )
                                ], style={'flex': 1, 'marginRight': '10px'}),

                                html.Div([
                                    html.Label("Filtro por País:", style={'color': 'white', 'fontSize': 12}),
                                    dcc.Dropdown(
                                        id='country-filter',
                                        options=[{'label': 'Todos', 'value': 'ALL'}],
                                        value='ALL',
                                        className='iracing-dropdown',
                                        # --- AÑADIMOS ESTILO INICIAL ---
                                        style={'width': '100%'}
                                    )
                                ], style={'flex': 1, 'marginRight': '10px'}),

                                html.Div([
                                    html.Label("Buscar Piloto:", style={'color': 'white', 'fontSize': 12}),
                                    dcc.Dropdown(
                                        id='pilot-search-dropdown',
                                        options=[],
                                        placeholder='Buscar Piloto...',
                                        className='iracing-dropdown',
                                        searchable=True,
                                        clearable=True,
                                        search_value='',
                                        # --- AÑADE ESTA LÍNEA PARA CAMBIAR EL COLOR DEL TEXTO ---
                                        style={'color': 'white'}
                                    )
                                ], style={'flex': 1})
                            ]
                        ),
                        # Contenedor de la Tabla
                        
                        html.Div(
                            kpi_pilot, 
                            style={
                                'marginTop': '2%'      # Pone los KPIs por delante del mapa
                            }
                        ),
                        html.Div(interactive_table, style={'flex': 1})
                    ]
                ),
                
                # --- COLUMNA CENTRAL ---
                html.Div(
                    id='middle-column',
                    style={'width': '50%', 'padding': '1%', 'display': 'flex', 'flexDirection': 'column'},
                    children=[
                        html.Div(
                            kpi_global, 
                            style={
                                'width': '70%',
                                'margin': '0 auto',
                                'position': 'relative', # Necesario para que z-index funcione
                                'z-index': '10'         # Pone los KPIs por delante del mapa
                            }
                        ),
                        html.Div(
                            continent_map, 
                            style={
                                'flex': 1, 
                                'minHeight': 0,
                                'marginTop': '-10%' # <-- Mueve el mapa 80px hacia arriba
                            }
                        ),
                        html.Div(
                            histogram_irating, 
                            style={
                                'width': '95%',
                                'marginTop': '5%',
                                'margin': '3% auto',
                                
                                'position': 'relative', # Necesario para que z-index funcione
                                'z-index': '10'         # Pone los KPIs por delante del mapa
                            }
                        ),
                        
                    ]
                ),
                
                # --- COLUMNA DERECHA ---
                html.Div(
                    id='right-column',
                    style={'width': '25%', 'padding': '1%', 'display': 'flex', 'flexDirection': 'column'},
                    children=[
                        
                        competitiveness_tables_container,
                        region_bubble_chart,
                        irating_starts_scatter # <-- AÑADE ESTA LÍNEA
                        
                    ]
                )
            ]
        ),
        
        # Componentes ocultos
        # --- ELIMINA EL dcc.Store ---
        dcc.Store(id='active-discipline-store', data='ROAD.csv'),
        dcc.Store(id='shared-data-store', data={}),
        dcc.Store(id='shared-data-store_1', data={}),
        html.Div(id='pilot-info-display', style={'display': 'none'})
    ]
)

# --- 4. Callbacks ---

# --- ELIMINA EL CALLBACK update_data_source ---

@app.callback(
    Output('active-discipline-store', 'data'),
    Input('btn-road', 'n_clicks'),
    Input('btn-formula', 'n_clicks'),
    Input('btn-oval', 'n_clicks'),
    Input('btn-dirt-road', 'n_clicks'),
    Input('btn-dirt-oval', 'n_clicks'),
    prevent_initial_call=True
)
def update_active_discipline(road, formula, oval, dr, do):
    ctx = dash.callback_context
    button_id = ctx.triggered_id.split('.')[0]
    
    file_map = {
        'btn-road': 'ROAD.csv',
        'btn-formula': 'FORMULA.csv',
        'btn-oval': 'OVAL.csv',
        'btn-dirt-road': 'DROAD.csv',
        'btn-dirt-oval': 'DOVAL.csv'
    }
    return file_map.get(button_id, 'ROAD.csv')


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

# --- NUEVO CALLBACK: FILTRAR POR PAÍS AL HACER CLIC EN EL MAPA ---
@app.callback(
    Output('country-filter', 'value'),
    Input('continent-map', 'clickData'),
    prevent_initial_call=True
)
def update_country_filter_on_map_click(clickData):
    # Si no hay datos de clic (por ejemplo, al cargar la página), no hacemos nada.
    if not clickData:
        return dash.no_update
    
    # Extraemos el código de país de 2 letras del 'customdata' que definimos en el gráfico.
    # clickData['points'][0] se refiere al primer país clickeado.
    # ['customdata'][0] se refiere al primer elemento de nuestra lista custom_data, que es 'LOCATION_2_LETTER'.
    country_code = clickData['points'][0]['customdata'][0]
    
    # Devolvemos el código del país, que actualizará el valor del dropdown 'country-filter'.
    return country_code

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


# --- CALLBACK PARA ESTILO DE FILTROS ACTIVOS (MODIFICADO) ---
@app.callback(
    Output('region-filter', 'className'),
    Output('country-filter', 'className'),
    Output('pilot-search-dropdown', 'className'), # <-- AÑADIMOS LA SALIDA
    Input('region-filter', 'value'),
    Input('country-filter', 'value'),
    Input('pilot-search-dropdown', 'value') # <-- AÑADIMOS LA ENTRADA
)
def update_filter_styles(region_value, country_value, pilot_value):
    # Clases base para los dropdowns
    default_class = 'iracing-dropdown'
    active_class = 'iracing-dropdown active-filter'

    # Asignar clases según el valor de cada filtro
    region_class = active_class if region_value and region_value != 'ALL' else default_class
    country_class = active_class if country_value and country_value != 'ALL' else default_class
    # NUEVA LÓGICA: El filtro de piloto está activo si tiene un valor
    pilot_class = active_class if pilot_value else default_class

    return region_class, country_class, pilot_class





# --- CALLBACK PARA ESTILO DE BOTONES ACTIVOS ---
@app.callback(
    Output('btn-formula', 'style'), # <-- AÑADIDO
    Output('btn-road', 'style'),
    Output('btn-oval', 'style'),
    Output('btn-dirt-road', 'style'),
    Output('btn-dirt-oval', 'style'),
    Input('btn-formula', 'n_clicks'), # <-- AÑADIDO
    Input('btn-road', 'n_clicks'),
    Input('btn-oval', 'n_clicks'),
   
    Input('btn-dirt-road', 'n_clicks'),
    Input('btn-dirt-oval', 'n_clicks')
)
def update_button_styles(formula_clicks, road_clicks, oval_clicks, dirt_road_clicks, dirt_oval_clicks): # <-- AÑADIDO
    # Estilos base para los botones
    base_style = {'width': '120px'} # Asegura que todos tengan el mismo ancho
    active_style = {
        'backgroundColor': 'rgba(0, 111, 255, 0.3)',
        'border': '1px solid rgb(0, 111, 255)',
        'width': '120px'
    }

    # Determinar qué botón fue presionado
    ctx = dash.callback_context
    if not ctx.triggered_id:
        # Estado inicial: 'Road' activo por defecto
        return base_style, active_style, base_style, base_style, base_style # <-- MODIFICADO

    button_id = ctx.triggered_id

    # Devolver el estilo activo para el botón presionado y el base para los demás
    if button_id == 'btn-formula': # <-- AÑADIDO
        return active_style, base_style, base_style, base_style, base_style
    elif button_id == 'btn-road':
        return base_style, active_style, base_style, base_style, base_style # <-- MODIFICADO
    elif button_id == 'btn-oval':
        return base_style, base_style, active_style, base_style, base_style # <-- MODIFICADO
    elif button_id == 'btn-dirt-road':
        return base_style, base_style, base_style, active_style, base_style # <-- MODIFICADO
    elif button_id == 'btn-dirt-oval':
        return base_style, base_style, base_style, base_style, active_style # <-- MODIFICADO
    
    # Fallback por si acaso
    return base_style, base_style, base_style, base_style, base_style # <-- MODIFICADO


# --- CALLBACK CONSOLIDADO: BÚSQUEDA Y TABLA (MODIFICADO PARA LEER DEL STORE) ---
@app.callback(
    Output('datatable-interactiva', 'data'),
    Output('datatable-interactiva', 'page_count'),
    Output('datatable-interactiva', 'columns'),
    Output('datatable-interactiva', 'page_current'),
    Output('histogram-plot', 'figure'),
    Output('continent-map', 'figure'),
    Output('kpi-global', 'figure'),
    Output('kpi-pilot', 'figure'),
    Output('shared-data-store', 'data'),
    # --- ELIMINAMOS LOS BOTONES COMO INPUTS ---
    # Input('btn-road', 'n_clicks'),
    # Input('btn-formula', 'n_clicks'),
    # Input('btn-oval', 'n_clicks'),
    # Input('btn-dirt-road', 'n_clicks'),
    # Input('btn-dirt-oval', 'n_clicks'),
    # --- LOS INPUTS AHORA EMPIEZAN CON LOS FILTROS ---
    Input('region-filter', 'value'),
    Input('country-filter', 'value'),
    Input('pilot-search-dropdown', 'value'),
    Input('datatable-interactiva', 'page_current'),
    Input('datatable-interactiva', 'page_size'),
    Input('datatable-interactiva', 'sort_by'),
    Input('datatable-interactiva', 'active_cell'),
    # --- AÑADIMOS EL STORE COMO STATE ---
    State('active-discipline-store', 'data'),
    # --- AÑADIMOS UN INPUT DEL STORE PARA REACCIONAR AL CAMBIO ---
    Input('active-discipline-store', 'data')
)
def update_table_and_search(
    region_filter, country_filter, selected_pilot,
    page_current, page_size, sort_by, state_active_cell,
    active_discipline_filename, # <-- Nuevo argumento desde el State
    discipline_change_trigger # <-- Nuevo argumento desde el Input
):
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'region-filter'

    # --- 1. LECTURA DEL ARCHIVO DESDE EL STORE ---
    # Ya no necesitamos el file_map aquí, simplemente usamos el nombre del archivo guardado.
    filename = active_discipline_filename
    
    # --- 2. PROCESAMIENTO DE DATOS (se hace cada vez) ---
    # Leemos y procesamos el archivo seleccionado
    df = pd.read_csv(filename)
    df = df[df['IRATING'] > 1]
    df = df[df['STARTS'] > 1]
    df = df[df['CLASS'].str.contains('D|C|B|A|P|R', na=False)]

    country_to_region_map = {country: region for region, countries in iracing_ragions.items() for country in countries}
    df['REGION'] = df['LOCATION'].map(country_to_region_map).fillna('International')
    
    df['Rank World'] = df['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
    df['Rank Region'] = df.groupby('REGION')['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
    df['Rank Country'] = df.groupby('LOCATION')['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
    
    df['CLASS'] = df['CLASS'].str[0]
    df_for_graphs = df.copy() # Copia para gráficos que no deben ser filtrados

    # --- 3. LÓGICA DE FILTRADO Y VISUALIZACIÓN (sin cambios) ---
    # El resto de la función sigue igual, pero ahora opera sobre el 'df' que acabamos de cargar.
    
    # Lógica de columnas dinámicas
    base_cols = ['DRIVER', 'IRATING', 'LOCATION', 'REGION','CLASS', 'STARTS', 'WINS' ]
    if country_filter and country_filter != 'ALL':
        dynamic_cols = ['Rank Country'] + base_cols
    elif region_filter and region_filter != 'ALL':
        dynamic_cols = ['Rank Region'] + base_cols
    else:
        dynamic_cols = ['Rank World'] + base_cols

    # Filtrado de datos
    if not region_filter: region_filter = 'ALL'
    if not country_filter: country_filter = 'ALL'

    filtered_df = df[dynamic_cols].copy()
    
    if region_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['REGION'] == region_filter]
    if country_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['LOCATION'] == country_filter]

    # --- 3. ORDENAMIENTO ---
    if sort_by:
        filtered_df = filtered_df.sort_values(
            by=sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc'
        )

    # --- 4. LÓGICA DE BÚSQUEDA Y NAVEGACIÓN (CORREGIDA) ---
    target_page = page_current
    new_active_cell = state_active_cell

    # Si el callback fue disparado por un cambio en los filtros,
    # reseteamos la celda activa y la página.
    if triggered_id in [
        'region-filter', 'country-filter', 
        'btn-road', 'btn-formula', 'btn-oval', 'btn-dirt-road', 'btn-dirt-oval'
    ]:
        new_active_cell = None
        target_page = 0  # <-- Esto hace que siempre se muestre la primera página

    # Si la búsqueda de piloto activó el callback, calculamos la nueva página y celda activa
    elif triggered_id == 'pilot-search-dropdown' and selected_pilot:
        match_index = filtered_df.index.get_loc(df[df['DRIVER'] == selected_pilot].index[0])
        if match_index is not None:
            target_page = match_index // page_size
            driver_column_index = list(filtered_df.columns).index('DRIVER')
            new_active_cell = {
                'row': match_index % page_size,
                'row_id': match_index % page_size,
                'column': driver_column_index,
                'column_id': 'DRIVER'
            }

   

    # --- 5. GENERACIÓN DE COLUMNAS PARA LA TABLA ---
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

    # --- 6. PAGINACIÓN ---
    start_idx = target_page * page_size

    end_idx = start_idx + page_size
    
    # Aplicamos el formato de bandera a los datos de la página actual
    page_df = filtered_df.iloc[start_idx:end_idx].copy()
    page_df['LOCATION'] = page_df['LOCATION'].map(lambda x: flag_img(x) if x in country_flags else x)
    page_data = page_df.to_dict('records')
    
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)

    # --- 7. ACTUALIZACIÓN DE GRÁFICOS ---
    graph_indices = filtered_df.index
    highlight_irating = None
    highlight_name = None
    pilot_info_for_kpi = None # Variable para guardar los datos del piloto
    
    pilot_to_highlight = selected_pilot
    
    if triggered_id == 'datatable-interactiva' and new_active_cell:
        row_index_in_df = (target_page * page_size) + new_active_cell['row']
        if row_index_in_df < len(filtered_df):
            pilot_to_highlight = filtered_df.iloc[row_index_in_df]['DRIVER']

    if pilot_to_highlight:
        pilot_data = df[df['DRIVER'] == pilot_to_highlight]
        if not pilot_data.empty:
            pilot_info_for_kpi = pilot_data.iloc[0] # <-- Guardamos toda la info del piloto
            highlight_irating = pilot_info_for_kpi['IRATING']
            highlight_name = pilot_info_for_kpi['DRIVER']
    elif not filtered_df.empty:
        top_pilot_in_view = filtered_df.nlargest(1, 'IRATING').iloc[0]
        highlight_irating = top_pilot_in_view['IRATING']
        highlight_name = top_pilot_in_view['DRIVER']
    
    # --- NUEVO: Generamos el gráfico de KPIs ---
    filter_context = "Mundo"
    if country_filter and country_filter != 'ALL':
        filter_context = country_filter
    elif region_filter and region_filter != 'ALL':
        filter_context = region_filter
        
    kpi_global_fig = create_kpi_global(filtered_df, filter_context)
    kpi_pilot_fig = create_kpi_pilot(filtered_df, pilot_info_for_kpi, filter_context)

    updated_histogram_figure = create_histogram_with_percentiles(
        df.loc[graph_indices], 
        'IRATING', 
        100,
        highlight_irating=highlight_irating,
        highlight_name=highlight_name
    )

    updated_map_figure = create_continent_map(df_for_graphs, region_filter, country_filter)

    shared_data = {
        'active_cell': new_active_cell,
        'selected_pilot': selected_pilot or '',
        'timestamp': str(pd.Timestamp.now())
    }
    
    '''return (page_data, total_pages, columns_definition, target_page, 
            new_active_cell, updated_histogram_figure, updated_map_figure)'''
    
    return (page_data, total_pages, columns_definition, target_page, 
            updated_histogram_figure, updated_map_figure, 
            kpi_global_fig,
            kpi_pilot_fig,
            shared_data)

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
    Output('shared-data-store_1', 'data'),
    

    Input('datatable-interactiva', 'active_cell'),
    State('shared-data-store', 'data'),
    State('shared-data-store_1', 'data'),
    Input('region-filter', 'value'),
    Input('country-filter', 'value'),
    prevent_initial_call=True
)
def update_active_cell_from_store(active_cell,ds,ds1,a,b):
    print(ds1)
    print(ds)
    print(active_cell)


    if not ds:
        return None
    if not ds1:
        ds1 = ds
        if ds.get('selected_pilot', '') == '':

            return active_cell,ds1
        return ds.get('active_cell'),ds1
    
    if ds.get('selected_pilot', '') == ds1.get('selected_pilot', ''):
        ds1 = ds

        if active_cell == ds1.get('active_cell'):
            
            return None,ds1
        ds1['active_cell'] = active_cell

        
        return active_cell,ds1
    else:
        ds1 = ds
        return ds.get('active_cell'),ds1

    
    
    
    '''active_cell = a
    selected_pilot = shared_data.get('selected_pilot', '')
    print('..............')
    print(shared_data)
    
    print(f"DEBUG: Recuperando active_cell del store: {active_cell}")
    print(f"DEBUG: Piloto asociado: {selected_pilot}")
    shared_data['shared_data'] = ''
    
    
    return active_cell'''




if __name__ == "__main__":
       app.run(debug=True)