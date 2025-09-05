import dash
from dash import html, dcc, dash_table, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pycountry_convert as pc
import pycountry
import gunicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
from datetime import datetime  # AÑADIR ESTA IMPORTACIÓN

iracing_ragions = {
    'US':['US'],
    'Mexico':['MX'],
    'Brazil':['BR'],
    'Canada':['CA'],
    'Atlantic':['GL'],
    'Japan':['JP'],
    'South America':['AR','PE','UY','CL','PY','BO','EC','CO','VE','GY','PA','CR','NI','HN','GT','BZ','SV','JM','DO','BS'],
    'Iberia':['ES','PT','AD'],
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


def load_and_process_data(filename):
    """Función para cargar y pre-procesar un archivo de disciplina."""
    print(f"Loading and processing {filename}...")
    df = pd.read_csv(filename)
    '''filename_parquet = filename.replace('.csv', '.parquet')
    df = pd.read_parquet(filename_parquet)'''
    df = df[df['IRATING'] > 1]
    df = df[df['STARTS'] > 1]
    df = df[df['CLASS'].str.contains('D|C|B|A|P|R', na=False)]

    country_to_region_map = {country: region for region, countries in iracing_ragions.items() for country in countries}
    df['REGION'] = df['LOCATION'].map(country_to_region_map).fillna('International')
    
    df['Rank World'] = df['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
    df['Rank Region'] = df.groupby('REGION')['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
    df['Rank Country'] = df.groupby('LOCATION')['IRATING'].rank(method='first', ascending=False).fillna(0).astype(int)
    
    df['CLASS'] = df['CLASS'].str[0]
    print(f"Finished processing {filename}.")
    return df

def update_all_data():
    """Actualiza todos los archivos de datos"""
    print(f"\n{'='*60}")
    print(f"🔄 SCHEDULED DATA UPDATE STARTED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    files_to_update = ['ROAD.csv', 'FORMULA.csv', 'OVAL.csv', 'DROAD.csv', 'DOVAL.csv']
    
    for filename in files_to_update:
        try:
            print(f"🔄 Updating {filename}...")
            new_data = load_and_process_data(filename)
            DISCIPLINE_DATAFRAMES[filename] = new_data
            print(f"✅ {filename} updated successfully! ({len(new_data)} records)")
        except Exception as e:
            print(f"❌ Error updating {filename}: {str(e)}")
    
    print(f"🎉 SCHEDULED DATA UPDATE COMPLETED")
    print(f"{'='*60}\n")

# CARGA INICIAL
DISCIPLINE_DATAFRAMES = {
    'ROAD.csv': load_and_process_data('data/ROAD.csv'),
    'FORMULA.csv': load_and_process_data('data/FORMULA.csv'),
    'OVAL.csv': load_and_process_data('data/OVAL.csv'),
    'DROAD.csv': load_and_process_data('data/DROAD.csv'),
    'DOVAL.csv': load_and_process_data('data/DOVAL.csv')
}

# CONFIGURAR EL SCHEDULER
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=update_all_data,
    trigger=IntervalTrigger(hours=2),  # Ejecutar cada 2 horas
    id='data_update_job',
    name='Update iRacing Data',
    replace_existing=True
)

# INICIAR EL SCHEDULER
scheduler.start()
print("🚀 Automatic data updater started with APScheduler!")
print("📅 Updates scheduled every 2 hours")

# ASEGURAR QUE EL SCHEDULER SE CIERRE AL CERRAR LA APP
atexit.register(lambda: scheduler.shutdown())



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
            "<b>iRating Range:</b> %{x}<br>" +
            "<b>Average Races:</b> %{y:.0f}<br>" +
            "<b>Drivers in this range:</b> %{customdata:,}<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>iRating Range By Avg. Races</b>',
            font=dict(color='white', size=14),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(11,11,19,1)',
        plot_bgcolor='rgba(11,11,19,1)',
        font=GLOBAL_FONT,
        xaxis=dict(
            title_text='iRating Range', # Texto del título
            title_font=dict(size=12),  # Estilo de la fuente del título
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title_text='Avg. Races', # Texto del título
            title_font=dict(size=12), # Estilo de la fuente del título
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(255,255,255,0.1)'
        ),
        margin=dict(l=10, r=10, t=50, b=10),
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

def create_region_bubble_chart(df):
    
    df = df[df['REGION'] != 'Atlantic']
    region_stats = df.groupby('REGION').agg(
        avg_starts=('STARTS', 'mean'),
        avg_irating=('IRATING', 'mean'),
        num_pilots=('DRIVER', 'count')
    ).reset_index()

    # --- ¡CLAVE PARA EL ORDEN! ---
    # Al ordenar de menor a mayor, Plotly dibuja las burbujas pequeñas al final,
    # asegurando que queden por encima de las grandes. ¡Esto ya está correcto!
    region_stats = region_stats.sort_values('num_pilots', ascending=True)
    hover_text_pilots = (region_stats['num_pilots'] / 1000).round(1).astype(str) + 'k'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=region_stats['avg_irating'],
        y=region_stats['avg_starts'],
        mode='markers+text',
        text=region_stats['REGION'],
        textposition='top center',
        # --- MODIFICACIÓN: Añadimos fondo al texto ---
        textfont=dict(
            size=8, 
            color='rgba(255, 255, 255, 0.9)',
            family='Lato, sans-serif'
        ),
        # --- FIN DE LA MODIFICACIÓN ---

        marker=dict(
            size=region_stats['num_pilots'],
            sizemode='area',
            sizeref=2.*max(region_stats['num_pilots'])/(50.**2.3), 
            sizemin=6,
            # --- MODIFICACIÓN: Coloreamos por número de pilotos ---
            color=region_stats['num_pilots'],
            # Usamos la misma escala de colores que el mapa para coherencia
            colorscale=[
                [0.0,  '#050A28'],
                [0.05, '#0A1950'],
                [0.15, '#0050B4'],
                [0.3,  '#006FFF'],
                [0.5,  '#3C96FF'],
                [0.7,  '#82BEFF'],
                [1.0,  '#DCEBFF']
            ],
            cmin=0,
            cmax=20000,
            showscale=False
            
            # --- FIN DE LA MODIFICACIÓN ---
        ),
        customdata=np.stack((hover_text_pilots, region_stats['num_pilots']), axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Avg. iRating: %{x:.0f}<br>" +
            "Avg. Races: %{y:.1f}<br>" +
            "Driver Qty: %{customdata[0]} (%{customdata[1]:,})<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text='<b>Regions (Avg. iRating, Avg. Races, Qty. Drivers)</b>',
            font=dict(color='white', size=14),
            x=0.5,
            xanchor='center'
        ),
        font=GLOBAL_FONT,
        #xaxis_title='Avg. iRating',
        #yaxis_title='Avg. Races',
        template='plotly_dark',
        
        paper_bgcolor='rgba(11,11,19,1)',
        
        plot_bgcolor='rgba(11,11,19,1)',
        # --- AÑADIMOS ESTILO DE GRID IGUAL AL HISTOGRAMA ---
        xaxis=dict(
            title_text='Avg. iRating', # Texto del título
            title_font=dict(size=12),  # Estilo de la fuente del título
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title_text='Avg. Races', # Texto del título
            title_font=dict(size=12), # Estilo de la fuente del título
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(255,255,255,0.1)'
        ),
        # --- FIN DEL ESTILO DE GRID ---
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def create_kpi_global(filtered_df, filter_context="World"):
    total_pilots = len(filtered_df)
    avg_irating = filtered_df['IRATING'].mean() if total_pilots > 0 else 0
    avg_starts = filtered_df['STARTS'].mean() if total_pilots > 0 else 0
    avg_wins = filtered_df['WINS'].mean() if total_pilots > 0 else 0

    fig = go.Figure()
    kpis = [
        {'value': total_pilots, 'title': f"Drivers {filter_context}", 'format': ',.0f'},
        {'value': avg_irating, 'title': "Avg iRating", 'format': ',.0f'},
        {'value': avg_starts, 'title': "Avg Starts", 'format': '.1f'},
        {'value': avg_wins, 'title': "Avg Wins", 'format': '.2f'}
    ]
    for i, kpi in enumerate(kpis):
        fig.add_trace(go.Indicator(
            mode="number",
            value=kpi['value'],
            number={'valueformat': kpi['format'], 'font': {'size': 20}},
            # --- MODIFICACIÓN: Añadimos <b> para poner el texto en negrita ---
            title={"text": f"<b>{kpi['title']}</b>", 'font': {'size': 16}},
            domain={'row': 0, 'column': i}
        ))
    fig.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor="#BD1818",
        margin=dict(l=20, r=20, t=50, b=10),
        height=60,
        font=GLOBAL_FONT
    )
    return fig

def create_kpi_pilot(filtered_df, pilot_info=None, filter_context="World"):
    fig = go.Figure()
    
    title_text = "Select a Driver"
    
    # Si NO hay información del piloto, creamos una figura vacía y ocultamos los ejes.
    if pilot_info is None:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_visible=False,
            yaxis_visible=False,
            height=240,  # Aumentamos la altura considerablemente
            annotations=[
                dict(
                    text="<b>Select or search a driver</b>",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="grey"
                    )
                )
            ]
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
        {'rank': rank_region, 'percentil': percentil_region, 'title': "Region Rank"},
        {'rank': rank_country, 'percentil': percentil_country, 'title': "Country Rank"}
    ]

    # Layout vertical con más espacio
    for i, kpi in enumerate(kpis_piloto):
        fig.add_trace(go.Indicator(
            mode="number",
            value=kpi['rank'],
            number={'prefix': "#", 'font': {'size': 18}},  # Número más grande
            title={
                "text": f"<b>{kpi['title']}</b><span style='font-size:11px;color:gray'>(Top {100-kpi['percentil']:.1f}%)</span>", 
                'font': {'size': 12}  # Título más grande
            },
            domain={
                'row': i, 
                'column': 0,
                # AGREGAMOS ESPACIADO ESPECÍFICO PARA CADA KPI
                'y': [0.75 - i*0.32, 0.95 - i*0.32]  # Da más espacio vertical a cada KPI
            }
        ))
    
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 28}
        },
        # CAMBIO: Eliminamos el grid automático y usamos dominios manuales
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=35, b=15),  # Más margen arriba y abajo
        height=240,  # Altura aumentada considerablemente
        font=GLOBAL_FONT,
        showlegend=False
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
    color_column = 'PILOTOS'
    color_scale = [
        [0.0,  '#050A28'],  # 1. Azul casi negro
        [0.05, '#0A1950'],  # 2. Azul marino oscuro
        [0.15, '#0050B4'],  # 3. Azul estándar
        [0.3,  '#006FFF'],  # 4. Azul Eléctrico (punto focal)
        [0.5,  '#3C96FF'],  # 5. Azul brillante
        [0.7,  '#82BEFF'],  # 6. Azul claro (cielo)
        [1.0,  '#DCEBFF']   # 7. Resplandor azulado (casi blanco)
    ]
    range_color_val = [0, 20000]

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
        hovertemplate="<b>%{customdata[0]}</b><br>Drivers: %{customdata[1]}<extra></extra>"
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
            title='Drivers',
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
            f"Range: {int(bin_edges[i])}-{int(bin_edges[i+1])}<br>"
            f"Drivers: {hist[i]}<br>"
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
        title=dict(
            text='<b>iRating Histogram</b>',
            font=dict(color='white', size=14),
            x=0.5,
            xanchor='center'
        ),
        font=GLOBAL_FONT,
        xaxis=dict(
            title_text='iRating', # Texto del título
            title_font=dict(size=12),  # Estilo de la fuente del título
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title_text='Qty. Drivers', # Texto del título
            title_font=dict(size=12), # Estilo de la fuente del título
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(255,255,255,0.1)'
        ),
        template='plotly_dark',
        hovermode='x unified',
        paper_bgcolor='rgba(18,18,26,.5)',
        plot_bgcolor='rgba(255,255,255,0)',

        
        # --- MODIFICACIÓN: Reducir márgenes y tamaño de fuentes ---
        margin=dict(l=10, r=10, t=50, b=10) # Reduce los márgenes (izquierda, derecha, arriba, abajo)                 # Reduce el tamaño del título principal

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
        title='🔁 Correlation between Variables',
        template='plotly_dark',
        margin=dict(l=40, r=20, t=40, b=20)
    )
    return fig

def flag_img(code):
    url = f"https://flagcdn.com/16x12/{code.lower()}.png"
    # La función ahora asume que si el código llega aquí, es válido.
    # La comprobación se hará una sola vez al crear el diccionario.
    return f'![{code}]({url})'

GLOBAL_FONT = {'family': "Lato, sans-serif"}

country_coords = {
    'ES': {'lat': 40.4, 'lon': -3.7}, 'US': {'lat': 39.8, 'lon': -98.5},
    'BR': {'lat': -14.2, 'lon': -51.9}, 'DE': {'lat': 51.1, 'lon': 10.4},
    'FR': {'lat': 46.2, 'lon': 2.2}, 'IT': {'lat': 41.8, 'lon': 12.5},
    'GB': {'lat': 55.3, 'lon': -3.4}, 'PT': {'lat': 39.3, 'lon': -8.2},
    'NL': {'lat':  52.1, 'lon': 5.2}, 'AU': {'lat': -25.2, 'lon': 133.7},
    'JP': {'lat': 36.2, 'lon': 138.2}, 'CA': {'lat': 56.1, 'lon': -106.3},
    'AR': {'lat': -38.4, 'lon': -63.6}, 'MX': {'lat': 23.6, 'lon': -102.5},
    'CL': {'lat': -35.6, 'lon': -71.5}, 'BE': {'lat': 50.5, 'lon': 4.4},
    'FI': {'lat': 61.9, 'lon': 25.7}, 'SE': {'lat': 60.1, 'lon': 18.6},
    'NO': {'lat': 60.4, 'lon': 8.4}, 'DK': {'lat': 56.2, 'lon': 9.5},
    'IE': {'lat': 53.4, 'lon': -8.2}, 'CH': {'lat': 46.8, 'lon': 8.2},
    'AT': {'lat': 47.5, 'lon': 14.5}, 'PL': {'lat': 51.9, 'lon': 19.1},
}



# --- 1. Carga y Preparación de Datos ---
df = DISCIPLINE_DATAFRAMES['ROAD.csv']
df = df[df['IRATING'] > 1]
df = df[df['STARTS'] > 1]
#df = df[df['STARTS'] < 2000]
df = df[df['CLASS'].str.contains('D|C|B|A|P|R', na=False)]

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
correlation_heatmap = dcc.Graph(
    id='correlation-heatmap',
    style={'height': '70vh'}, # Ajusta la altura para que quepan los 3 gráficos
    # Usamos las columnas numéricas del dataframe original
    figure=create_correlation_heatmap(df[['IRATING', 'STARTS', 'WINS','TOP25PCNT','AVG_INC','AVG_FINISH_POS']])
)


kpi_global = dcc.Graph(id='kpi-global', style={'height': '6vh', 'marginBottom': '0px', 'marginTop': '20px'})
kpi_pilot = dcc.Graph(id='kpi-pilot', style={'height': '3hv', 'marginBottom': '10px'})

histogram_irating = dcc.Graph(
    id='histogram-plot',
    # --- MODIFICACIÓN: Ajustamos el estilo del contenedor del gráfico ---
    style={
        'height': '26vh',
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
    page_size=100,  # CAMBIO: De 20 a 100 elementos por página
    page_count=len(df_table) // 100 + (1 if len(df_table) % 100 > 0 else 0),  # CAMBIO: Actualizar cálculo
    virtualization=False,
    style_as_list_view=True,
    active_cell={'row': 21,'column':1},
    
    style_table={
        'overflowX': 'auto',
        'overflowY': 'auto',  # SCROLL VERTICAL habilitado
        'height': '100%',     # CAMBIO: Usar 100% del contenedor padre
        'maxHeight': '100%',  # CAMBIO: Usar 100% del contenedor padre
        'minHeight': '700px', # CAMBIO: Altura mínima aumentada
        'width': '100%',
        'borderRadius': '15px',
        'overflow': 'hidden',
        'backgroundColor': 'rgba(11,11,19,1)',
        'border': '1px solid #4A4A4A'
    },
    
    style_cell={
        'textAlign': 'center',
        'padding': '6px 3px',  # CAMBIO: Padding más compacto para aprovechar espacio
        'backgroundColor': 'rgba(11,11,19,1)',
        'color': 'rgb(255, 255, 255,.8)',
        'border': '1px solid rgba(255, 255, 255, 0.1)',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'whiteSpace': 'nowrap',
        'maxWidth': 0,
        'fontSize': '11px'  # CAMBIO: Fuente ligeramente más pequeña para más filas
    },
    
    style_header={
        'backgroundColor': 'rgba(30,30,38,1)',
        'fontWeight': 'bold',
        'color': 'white',
        'border': '1px solid rgba(255, 255, 255, 0.2)',
        'textAlign': 'center',
        'fontSize': '12px',
        'position': 'sticky',  # Header pegajoso
        'top': 0,              # Se queda arriba al hacer scroll
        'zIndex': 10,          # Prioridad visual
        'padding': '6px 3px'   # CAMBIO: Padding más compacto
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
            'fontSize': 10,},
     # Renderizado virtual
)

scatter_irating_starts = dcc.Graph(
    id='scatter-irating',
    style={'height': '20vh','borderRadius': '15px','overflow': 'hidden'},
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

continent_map = dcc.Graph(
    id='continent-map',
    style={'height': '55vh'},
    figure=create_continent_map(df_for_graphs)
)

region_bubble_chart = dcc.Graph(
    id='region-bubble-chart',
    style={'height': '33vh','borderRadius': '10px','border': '1px solid #4A4A4A',
        'overflow': 'hidden'},
    figure=create_region_bubble_chart(df)
)

# --- 3. Inicialización de la App ---
app = dash.Dash(__name__)
server = app.server # <-- AÑADE ESTA LÍNEA

# Layout principal MODIFICADO
app.layout = html.Div(
    style={
        'margin': '0',
        'padding': '0',
        'backgroundColor': 'rgba(5,5,15,255)',
        'color': '#ffffff',
        'fontFamily': 'Lato, sans-serif',
        'minHeight': '100vh'
    },
    children=[
        # CONTENEDOR PRINCIPAL CENTRADO
        html.Div(
            style={
                'maxWidth': '1400px',
                'margin': '0 auto',
                'padding': '20px 5vw',
                'minHeight': '100vh',
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '30px'
            },
            children=[
                
                # 1. SECCIÓN HEADER - Título y Botones
                html.Div(
                    style={
                        'textAlign': 'center',
                        'marginBottom': '20px'
                    },
                    children=[
                        html.H1(
                            "🏁 Top iRating Dashboard", 
                            style={
                                'fontSize': 'clamp(36px, 5vw, 48px)',
                                'color': 'white',
                                'margin': '0 0 20px 0',
                                'fontWeight': '900'
                            }
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'gap': '10px',
                                'flexWrap': 'wrap'
                            },
                            children=[
                                html.Button('Sports Car', id='btn-road', n_clicks=0, className='dashboard-type-button'),
                                html.Button('Formula', id='btn-formula', n_clicks=0, className='dashboard-type-button'),
                                html.Button('Oval', id='btn-oval', n_clicks=0, className='dashboard-type-button'),
                                html.Button('Dirt Road', id='btn-dirt-road', n_clicks=0, className='dashboard-type-button'),
                                html.Button('Dirt Oval', id='btn-dirt-oval', n_clicks=0, className='dashboard-type-button'),
                            ]
                        )
                    ]
                ),
                
                # 2. SECCIÓN MAPA Y KPIs GLOBALES - Solo KPIs globales superpuestos
                html.Div(
                    style={
                        'position': 'relative',
                        'top': '0px',
                        'backgroundColor': 'transparent',  # Fondo transparente
                        'borderRadius': '0px',             # Sin bordes redondeados
                        'border': '0px solid #4A4A4A',
                        'padding': '20px',
                        'overflow': 'hidden'
                    },
                    children=[
                        # Solo KPIs globales superpuestos
                        html.Div(
                            style={
                                'position': 'absolute',
                                'top': '0px',
                                'left': '50%',
                                'transform': 'translateX(-50%)',
                                'width': '80%',
                                'maxWidth': '800px',
                                'zIndex': '10',
                                'backgroundColor': 'rgba(11,11,19,0)',
                                'borderRadius': '10px',
                                'border': '0px solid #4A4A4A',
                                'padding': '10px'
                            },
                            children=[
                                dcc.Graph(id='kpi-global', style={'height': '50px', 'margin': '0'})
                            ]
                        ),
                        # Mapa de fondo
                        dcc.Graph(
                            id='continent-map',
                            
                            style={'height': '650px',
                                   'backgroundColor': 'transparent', 
                                    'margin': '0'},
                            config={'displayModeBar': False}
                        )
                    ]
                ),
                
                # 3. SECCIÓN FILTROS Y TABLA - Lado a lado CON KPIs DEL PILOTO
                html.Div(
                    style={
                        'display': 'flex',
                        'gap': '20px',
                        'flexWrap': 'wrap'
                        
                    },
                    children=[
                        # Contenedor de Filtros CON KPIs DEL PILOTO
                        html.Div(
                            style={
                                'flex': '1',
                                'minWidth': '300px',
                                'maxWidth': '600px',
                                'backgroundColor': 'rgba(18,18,26,.5)',
                                'borderRadius': '15px',
                                'border': '1px solid #4A4A4A',
                                'padding': '20px'
                            },
                            children=[
                                html.H3(
                                    "Filters", 
                                    style={
                                        'color': 'white',
                                        'textAlign': 'center',
                                        'marginBottom': '20px',
                                        'fontWeight': '700'
                                    }
                                ),
                                
                                # Filtros de región y país
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'gap': '15px',
                                        'marginBottom': '20px',
                                        'flexDirection': 'column'
                                    },
                                    children=[
                                        html.Div([

                                            html.Label(
                                                "Region:", 
                                                style={
                                                    'color': 'white',
                                                    'fontSize': '14px',
                                                    'marginBottom': '5px',
                                                    'display': 'block',
                                                    'textAlign': 'center'
                                                }
                                            ),
                                            dcc.Dropdown(
                                                id='region-filter',
                                                options=[{'label': 'All', 'value': 'ALL'}] + 
                                                       [{'label': region, 'value': region} for region in sorted(iracing_ragions.keys())],
                                                value='ALL',
                                                className='iracing-dropdown'
                                            )
                                        ]),
                                        
                                        html.Div([
                                            html.Label(
                                                "Country:", 
                                                style={
                                                    'color': 'white',
                                                    'fontSize': '14px',
                                                    'marginBottom': '5px',
                                                    'display': 'block',
                                                    'textAlign': 'center'
                                                }
                                            ),
                                            dcc.Dropdown(
                                                id='country-filter',
                                                options=[{'label': 'All', 'value': 'ALL'}],
                                                value='ALL',
                                                className='iracing-dropdown'
                                            )
                                        ])
                                    ]
                                ),
                                
                                # Búsqueda de piloto
                                html.Div([
                                    html.Label(
                                        "Search Driver:", 
                                        style={
                                            'color': 'white',
                                            'fontSize': '14px',
                                            'marginBottom': '5px',
                                            'display': 'block',
                                            'textAlign': 'center'
                                        }
                                    ),
                                    dcc.Dropdown(
                                        id='pilot-search-dropdown',
                                        options=[],
                                        placeholder='Search Driver...',
                                        className='iracing-dropdown',
                                        searchable=True,
                                        clearable=True,
                                        search_value=''
                                    )
                                ]),
                                
                                # NUEVA SECCIÓN: KPIs del piloto debajo de los filtros
                                html.Div(
                                    style={
                                        'marginTop': '30px',
                                        'padding': '15px',
                                        'backgroundColor': 'rgba(11,11,19,0.8)',
                                        'borderRadius': '10px',
                                        'border': '1px solid #4A4A4A'
                                    },
                                    children=[
                                        html.H4(
                                            "Selected Driver", 
                                            style={
                                                'color': 'white',
                                                'textAlign': 'center',
                                                'marginBottom': '15px',
                                                'fontSize': '16px',
                                                'fontWeight': '700'
                                            }
                                        ), 
                                        dcc.Graph(
                                            id='kpi-pilot', 
                                            style={
                                                'height': '260px',  # Aumentamos altura del contenedor
                                                'margin': '0'
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),
                        
                        # Contenedor de Tabla MODIFICADO
                        html.Div(
                            style={
                                'flex': '2',
                                'minWidth': '300px',
                                'backgroundColor': 'rgba(18,18,26,.5)',
                                'borderRadius': '15px',
                                'border': '1px solid #4A4A4A',
                                'padding': '20px',
                                'display': 'flex',          # CAMBIO: Usar flexbox
                                'flexDirection': 'column',  # CAMBIO: Dirección vertical
                                'height': '1000px'          # CAMBIO: Altura fija del contenedor
                            },
                            children=[
                                html.H3(
                                    "Rankings", 
                                    style={
                                        'color': 'white',
                                        'textAlign': 'center',
                                        'marginBottom': '20px',
                                        'fontWeight': '700',
                                        'flexShrink': 0  # CAMBIO: No se encoge
                                    }
                                ),
                                # Contenedor específico para la tabla
                                html.Div(
                                    style={
                                        'flex': '1',           # CAMBIO: Ocupa todo el espacio restante
                                        'display': 'flex',
                                        'flexDirection': 'column',
                                        'minHeight': 0         # CAMBIO: Permite que se encoja si es necesario
                                    },
                                    children=[
                                        dash_table.DataTable(
                                            id='datatable-interactiva',
                                            data=[],
                                            sort_action="custom",
                                            sort_mode="single",
                                            page_action="custom",
                                            page_current=0,
                                            page_size=100,  # CAMBIO: 100 elementos por página
                                            page_count=len(df_table) // 100 + (1 if len(df_table) % 100 > 0 else 0),
                                            virtualization=False,
                                            style_as_list_view=True,
                                            active_cell={'row': 101,'column':1},
                                            
                                            style_table={
                                                'overflowX': 'auto',
                                                'overflowY': 'auto',
                                                'height': '100%',     # CAMBIO: Usa todo el espacio del contenedor padre
                                                'maxHeight': '100%',  # CAMBIO: No limitar altura
                                                'minHeight': '900px',
                                                'width': '100%',
                                                'borderRadius': '15px',
                                                'overflow': 'hidden',
                                                'backgroundColor': 'rgba(11,11,19,1)',
                                                'border': '1px solid #4A4A4A'
                                            },
                                            
                                            style_cell={
                                                'textAlign': 'center',
                                                'padding': '6px 3px',  # Padding más compacto
                                                'backgroundColor': 'rgba(11,11,19,1)',
                                                'color': 'rgb(255, 255, 255,.8)',
                                                'border': '1px solid rgba(255, 255, 255, 0.1)',
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
                                                'whiteSpace': 'nowrap',
                                                'maxWidth': 0,
                                                'fontSize': '11px'  # Fuente más compacta
                                            },
                                            
                                            style_header={
                                                'backgroundColor': 'rgba(30,30,38,1)',
                                                'fontWeight': 'bold',
                                                'color': 'white',
                                                'border': '1px solid rgba(255, 255, 255, 0.2)',
                                                'textAlign': 'center',
                                                'fontSize': '12px',
                                                'position': 'sticky',
                                                'top': 0,
                                                'zIndex': 10,
                                                'padding': '6px 3px'
                                            },
                                            
                                            # Mantén todos tus estilos existentes
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
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                
                # 4. SECCIÓN HISTOGRAMA - Ancho completo (sin cambios)
                html.Div(
                    style={
                        'backgroundColor': 'rgba(18,18,26,.5)',
                        'borderRadius': '15px',
                        'border': '1px solid #4A4A4A',
                        'padding': '20px'
                    },
                    children=[
                        html.H3(
                            "iRating Distribution", 
                            style={
                                'color': 'white',
                                'textAlign': 'center',
                                'marginBottom': '20px',
                                'fontWeight': '700'
                            }
                        ),
                        dcc.Graph(
                            id='histogram-plot',
                            style={'height': '350px'},
                            config={'displayModeBar': False}
                        )
                    ]
                ),
                
                # 5. SECCIÓN GRÁFICOS ADICIONALES (sin cambios)
                html.Div(
                    style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '20px'
                    },
                    children=[
                        # Primera fila: Tabla de competitividad (sin cambios)
                        html.Div(
                            style={
                                'backgroundColor': 'rgba(18,18,26,.5)',
                                'borderRadius': '15px',
                                'border': '1px solid #4A4A4A',
                                'padding': '20px'
                            },
                            children=[
                                html.H3(
                                    "Top Competitive Regions & Countries", 
                                    style={
                                        'color': 'white',
                                        'textAlign': 'center',
                                        'marginBottom': '10px',
                                        'fontWeight': '700'
                                    }
                                ),
                                html.P(
                                    "Based on average iRating of top 100 drivers per region/country (minimum 100 drivers required)",
                                    style={
                                        'color': '#CCCCCC',
                                        'textAlign': 'center',
                                        'marginBottom': '20px',
                                        'fontSize': '12px',
                                        'fontStyle': 'italic'
                                    }
                                ),
                                html.Div(id='competitiveness-tables-container')
                            ]
                        ),
                        
                        # CAMBIO: Ahora los gráficos están en columna vertical
                        # Primer gráfico: Regional Analysis
                        html.Div(
                            style={
                                'backgroundColor': 'rgba(18,18,26,.5)',
                                'borderRadius': '15px',
                                'border': '1px solid #4A4A4A',
                                'padding': '20px'
                            },
                            children=[
                                html.H3(
                                    "Regional Analysis", 
                                    style={
                                        'color': 'white',
                                        'textAlign': 'center',
                                        'marginBottom': '20px',
                                        'fontWeight': '700'
                                    }
                                ),
                                dcc.Graph(
                                    id='region-bubble-chart',
                                    style={'height': '400px'},  # Aumentamos altura ya que ahora ocupa todo el ancho
                                    config={'displayModeBar': False}
                                )
                            ]
                        ),
                        
                        # Segundo gráfico: Experience vs Performance
                        html.Div(
                            style={
                                'backgroundColor': 'rgba(18,18,26,.5)',
                                'borderRadius': '15px',
                                'border': '1px solid #4A4A4A',
                                'padding': '20px'
                            },
                            children=[
                                html.H3(
                                    "Experience vs Performance", 
                                    style={
                                        'color': 'white',
                                        'textAlign': 'center',
                                        'marginBottom': '20px',
                                        'fontWeight': '700'
                                    }
                                ),
                                dcc.Graph(
                                    id='irating-starts-scatter',
                                    style={'height': '400px'},  # Aumentamos altura ya que ahora ocupa todo el ancho
                                    config={'displayModeBar': False}
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Componentes ocultos (sin cambios)
        dcc.Store(id='active-discipline-store', data='ROAD.csv'),
        dcc.Store(id='shared-data-store', data={}),
        dcc.Store(id='shared-data-store_1', data={}),
        html.Div(id='pilot-info-display', style={'display': 'none'})
    ]
)

# --- 4. Callbacks ---


# --- NUEVO CALLBACK PARA ACTUALIZAR GRÁFICOS DE LA COLUMNA DERECHA ---
@app.callback(
    Output('competitiveness-tables-container', 'children'),
    Output('region-bubble-chart', 'figure'),
    Output('irating-starts-scatter', 'figure'),
    Input('active-discipline-store', 'data')
)
def update_right_column_graphs(filename):
    # 1. Cargar y procesar los datos de la disciplina seleccionada
    df_discipline = DISCIPLINE_DATAFRAMES[filename]
    df_discipline = df_discipline[df_discipline['IRATING'] > 1]
    df_discipline = df_discipline[df_discipline['STARTS'] > 1]
    df_discipline = df_discipline[df_discipline['CLASS'].str.contains('D|C|B|A|P|R', na=False)]
    
    country_to_region_map = {country: region for region, countries in iracing_ragions.items() for country in countries}
    df_discipline['REGION'] = df_discipline['LOCATION'].map(country_to_region_map).fillna('International')

    # 2. Calcular y crear las tablas de competitividad
    top_regions, top_countries = calculate_competitiveness(df_discipline)
    top_regions.insert(0, '#', range(1, 1 + len(top_regions)))
    top_countries.insert(0, '#', range(1, 1 + len(top_countries)))

    # Traducir códigos de país a nombres completos
    def get_country_name(code):
        try:
            return pycountry.countries.get(alpha_2=code).name
        except (LookupError, AttributeError):
            return code

    top_countries['LOCATION'] = top_countries['LOCATION'].apply(get_country_name)

    # ESTILOS CORREGIDOS - Control estricto de ancho
    table_style_base = {
        'style_table': {
            'borderRadius': '10px', 
            'overflow': 'hidden', 
            'border': '1px solid #4A4A4A',
            'backgroundColor': 'rgba(11,11,19,1)',
            'height': '350px',
            'overflowY': 'auto',
            'overflowX': 'hidden',  # CLAVE: Prevenir scroll horizontal
            'width': '100%',
            'maxWidth': '100%'  # CLAVE: Forzar límite de ancho
        },
        'style_cell': {
            'textAlign': 'center', 
            'padding': '6px 4px',  # Padding más compacto
            'backgroundColor': 'rgba(11,11,19,1)', 
            'color': 'rgb(255, 255, 255,.8)', 
            'border': 'none', 
            'fontSize': '11px',  # Fuente más pequeña
            'textOverflow': 'ellipsis',
            'whiteSpace': 'nowrap',
            'overflow': 'hidden',
            'maxWidth': '0'  # CLAVE: Forzar truncado de texto
        },
        'style_header': {
            'backgroundColor': 'rgba(30,30,38,1)', 
            'fontWeight': 'bold', 
            'color': 'white', 
            'border': 'none', 
            'textAlign': 'center',
            'fontSize': '12px',
            'padding': '6px 4px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis'
        },
        'style_cell_conditional': [
            {'if': {'column_id': '#'}, 'width': '10%', 'minWidth': '10%', 'maxWidth': '10%'},
            {'if': {'column_id': 'REGION'}, 'width': '60%', 'minWidth': '60%', 'maxWidth': '60%', 'textAlign': 'left'},
            {'if': {'column_id': 'LOCATION'}, 'width': '60%', 'minWidth': '60%', 'maxWidth': '60%', 'textAlign': 'left'},
            {'if': {'column_id': 'avg_irating'}, 'width': '30%', 'minWidth': '30%', 'maxWidth': '30%'},
        ]
    }

    # CONTENEDOR CON CONTROL ESTRICTO DE ANCHO
    competitiveness_tables = html.Div(
        style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gap': '15px',  # Gap más pequeño
            'width': '100%',
            'maxWidth': '100%',
            'overflow': 'hidden',
            'boxSizing': 'border-box'  # CLAVE: Incluir padding en el ancho
        },
        children=[
            # Tabla de Regiones
            html.Div(
                style={
                    'width': '100%',
                    'maxWidth': '100%',
                    'overflow': 'hidden',
                    'boxSizing': 'border-box'  # CLAVE: Control de ancho
                },
                children=[
                    html.H4(
                        "Top Regions", 
                        style={
                            'color': 'white', 
                            'textAlign': 'center', 
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'margin': '0 0 10px 0'
                        }
                    ),
                    html.Div(
                        style={
                            'width': '100%',
                            'maxWidth': '100%',
                            'overflow': 'hidden'
                        },
                        children=[
                            dash_table.DataTable(
                                columns=[
                                    {'name': '#', 'id': '#'}, 
                                    {'name': 'Region', 'id': 'REGION'}, 
                                    {'name': 'Avg iRating', 'id': 'avg_irating', 'type': 'numeric', 'format': {'specifier': '.0f'}}
                                ],
                                data=top_regions.to_dict('records'),
                                page_action='none',
                                style_table=table_style_base['style_table'],
                                style_cell=table_style_base['style_cell'],
                                style_header=table_style_base['style_header'],
                                style_cell_conditional=table_style_base['style_cell_conditional']
                            )
                        ]
                    )
                ]
            ),
            
            # Tabla de Países
            html.Div(
                style={
                    'width': '100%',
                    'maxWidth': '100%',
                    'overflow': 'hidden',
                    'boxSizing': 'border-box'  # CLAVE: Control de ancho
                },
                children=[
                    html.H4(
                        "Top Countries", 
                        style={
                            'color': 'white', 
                            'textAlign': 'center', 
                            'marginBottom': '10px',
                            'fontSize': '14px',
                            'margin': '0 0 10px 0'
                        }
                    ),
                    html.Div(
                        style={
                            'width': '100%',
                            'maxWidth': '100%',
                            'overflow': 'hidden'
                        },
                        children=[
                            dash_table.DataTable(
                                columns=[
                                    {'name': '#', 'id': '#'}, 
                                    {'name': 'Country', 'id': 'LOCATION'}, 
                                    {'name': 'Avg iRating', 'id': 'avg_irating', 'type': 'numeric', 'format': {'specifier': '.0f'}}
                                ],
                                data=top_countries.to_dict('records'),
                                page_action='none',
                                style_table=table_style_base['style_table'],
                                style_cell=table_style_base['style_cell'],
                                style_header=table_style_base['style_header'],
                                style_cell_conditional=table_style_base['style_cell_conditional']
                            )
                        ]
                    )
                ]
            )
        ]
    )

    # 3. Crear los otros gráficos
    bubble_chart_fig = create_region_bubble_chart(df_discipline)
    line_chart_fig = create_irating_trend_line_chart(df_discipline)

    # 4. Devolver todos los componentes actualizados
    return competitiveness_tables, bubble_chart_fig, line_chart_fig



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
    # --- MODIFICACIÓN: Traducir códigos de país a nombres completos ---
    if not selected_region or selected_region == 'ALL':
        # Si no hay región o es 'ALL', toma todos los códigos de país únicos del dataframe
        country_codes = df['LOCATION'].dropna().unique()
    else:
        # Si se selecciona una región, toma solo los países de esa región
        country_codes = iracing_ragions.get(selected_region, [])
    
    options = [{'label': 'All', 'value': 'ALL'}]
    
    # Crear una lista de tuplas (nombre_completo, codigo) para poder ordenarla por nombre
    country_list_for_sorting = []
    for code in country_codes:
        try:
            # Busca el país por su código de 2 letras y obtiene el nombre
            country_name = pycountry.countries.get(alpha_2=code).name
            country_list_for_sorting.append((country_name, code))
        except (LookupError, AttributeError):
            # Si no se encuentra (código inválido), usa el código como nombre para no romper la app
            country_list_for_sorting.append((code, code))
            
    # Ordenar la lista alfabéticamente por el nombre completo del país
    sorted_countries = sorted(country_list_for_sorting)
    
    # Crear las opciones para el dropdown a partir de la lista ordenada
    for country_name, country_code in sorted_countries:
        options.append({'label': country_name, 'value': country_code})
        
    return options
    # --- FIN DE LA MODIFICACIÓN ---

# --- NUEVO CALLBACK: FILTRO POR PAÍS AL HACER CLIC EN EL MAPA ---
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
    State('pilot-search-dropdown', 'value'),
    State('region-filter', 'value'),
    State('country-filter', 'value'),
    # --- MODIFICACIÓN: Añadimos el State para saber la disciplina activa ---
    State('active-discipline-store', 'data'),
    prevent_initial_call=True,
)
def update_pilot_search_options(search_value, current_selected_pilot, region_filter, country_filter, active_discipline_filename):
    # --- MODIFICACIÓN: Cargamos el DataFrame correcto al inicio de la función ---
    # 1. Cargar los datos de la disciplina actual
    df_current_discipline = DISCIPLINE_DATAFRAMES[active_discipline_filename]
    # Aplicamos los mismos filtros iniciales que en el callback principal
    df_current_discipline = df_current_discipline[df_current_discipline['IRATING'] > 1]
    df_current_discipline = df_current_discipline[df_current_discipline['STARTS'] > 1]
    df_current_discipline = df_current_discipline[df_current_discipline['CLASS'].str.contains('D|C|B|A|P|R', na=False)]
    
    # Asignamos la región para poder filtrar por ella
    country_to_region_map = {country: region for region, countries in iracing_ragions.items() for country in countries}
    df_current_discipline['REGION'] = df_current_discipline['LOCATION'].map(country_to_region_map).fillna('International')
    # --- FIN DE LA MODIFICACIÓN ---

    # Si no hay texto de búsqueda, pero ya hay un piloto seleccionado,
    # nos aseguramos de que su opción esté disponible para que no desaparezca.
    if not search_value:
        if current_selected_pilot:
            return [{'label': current_selected_pilot, 'value': current_selected_pilot}]
        return []

    # Mantenemos la optimización de no buscar con texto muy corto
    if len(search_value) < 2:
        return []

    # 1. La lógica de filtrado ahora usa el DataFrame correcto
    if not region_filter: region_filter = 'ALL'
    if not country_filter: country_filter = 'ALL'

    filtered_df = df_current_discipline
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
    Input('active-discipline-store', 'data') # <-- AÑADE ESTE INPUT
)
def clear_pilot_search_on_filter_change(region, country, discipline_data): # <-- AÑADE EL ARGUMENTO
    # Cuando un filtro principal o la disciplina cambia, reseteamos la selección del piloto
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
    active_discipline_filename,
    discipline_change_trigger
):
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'region-filter'

    # --- 1. LECTURA DEL ARCHIVO DESDE EL STORE ---
    # Ya no necesitamos el file_map aquí, simplemente usamos el nombre del archivo guardado.
    filename = active_discipline_filename
    
    # --- 2. PROCESAMIENTO DE DATOS (se hace cada vez) ---
    # Leemos y procesamos el archivo seleccionado
    #df = pd.read_csv(filename)
    df = DISCIPLINE_DATAFRAMES[active_discipline_filename]

    df_for_graphs = df.copy() # Copia para gráficos que no deben ser filtrados
    
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
        'active-discipline-store'
    ]:
        new_active_cell = None
        target_page = 0  # <-- Esto hace que siempre se muestre la primera página

    # Si la búsqueda de piloto activó el callback, calculamos la nueva página y celda activa
    elif triggered_id == 'pilot-search-dropdown' and selected_pilot:
        match_index = filtered_df.index.get_loc(df[df['DRIVER'] == selected_pilot].index[0])
        if match_index is not None:
            target_page = match_index // 100  # CAMBIO: Usar 100 en lugar de page_size
            driver_column_index = list(filtered_df.columns).index('DRIVER')
            new_active_cell = {
                'row': match_index % 100,  # CAMBIO: Usar 100 en lugar de page_size
                'row_id': match_index % 100,  # CAMBIO: Usar 100
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
    page_size = 100  # FORZAR: Siempre usar 100 elementos por página
    start_idx = target_page * page_size
    end_idx = start_idx + page_size
    
    # Aplicamos el formato de bandera a los datos de la página actual
    page_df = filtered_df.iloc[start_idx:end_idx].copy()
    page_df['LOCATION'] = page_df['LOCATION'].map(lambda x: flag_img(x))
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
    filter_context = "World"
    if country_filter and country_filter != 'ALL':
        try:
            # Traducimos el código de país a nombre para el KPI
            filter_context = pycountry.countries.get(alpha_2=country_filter).name
        except (LookupError, AttributeError):
            filter_context = country_filter # Usamos el código si no se encuentra
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
        return "No driver selected"
    
    # Obtener el nombre del piloto de la fila seleccionada
    selected_row = active_cell['row']
    if selected_row >= len(table_data):
        return "Invalid row"
    
    pilot_name = table_data[selected_row]['DRIVER']
    
    # Buscar todos los datos del piloto en el DataFrame original
    pilot_data = df[df['DRIVER'] == pilot_name]
    
    if pilot_data.empty:
        return f"No data found for {pilot_name}"
    
    # Obtener la primera (y única) fila del piloto
    pilot_info = pilot_data.iloc[0]
    
    # IMPRIMIR EN CONSOLA todos los datos del piloto
    print("\n" + "="*50)
    print(f"SELECTED DRIVER DATA: {pilot_name}")
    print("="*50)
    for column, value in pilot_info.items():
        print(f"{column}: {value}")
    print("="*50 + "\n")
    
    # También retornar información para mostrar en la interfaz (opcional)
    return f"Selected driver: {pilot_name} (See console for full data)"

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


if __name__ == "__main__":
       app.run(debug=True)



