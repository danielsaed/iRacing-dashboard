import dash
from dash import html, dcc, dash_table, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pycountry_convert as pc
import pycountry



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

def create_continent_map(df, selected_country='ALL'):
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

    # Creación del mapa base
    fig = px.choropleth(
        country_counts,
        locations="LOCATION_3_LETTER",
        locationmode="ISO-3",
        color="PILOTOS",
        hover_name="LOCATION_2_LETTER",
        color_continuous_scale=px.colors.sequential.Plasma,
        projection="natural earth" # Usamos una proyección que permite zoom
    )
    
    # --- LÓGICA DE ZOOM DINÁMICO ---
    if selected_country != 'ALL' and selected_country in country_coords:
        # Si se selecciona un país válido, hacemos zoom en él
        zoom_level = 4 if selected_country not in ['US', 'CA', 'AU', 'BR', 'AR'] else 3
        fig.update_geos(
            center=country_coords[selected_country],
            projection_scale=zoom_level
        )
    else:
        # Si es 'ALL' o no está en el dict, vista mundial
        fig.update_geos(
            center={'lat': 20, 'lon': 0},
            projection_scale=1
        )

    fig.update_layout(
        title_text='🌍 Pilotos por País',
        template='plotly_dark',
        geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#4E5D6C', landcolor='#323232', subunitcolor='grey'),
        margin={"r":0,"t":40,"l":0,"b":0}
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
            [1.0, "#2C78DB"]    # Termina en blanco
        ])
    ))
    
    fig.update_layout(
        title='Distribución de iRating (Valores pequeños amplificados)',
        xaxis_title='iRating',
        yaxis_title='Cantidad de Pilotos (escala ajustada)',
        template='plotly_dark',
        hovermode='x unified',
        # --- AÑADE ESTAS LÍNEAS PARA CAMBIAR EL FONDO ---
        paper_bgcolor='rgba(0,0,0,1)', # Fondo de toda la figura (transparente)
        plot_bgcolor='rgba(0,0,0,1)',  # Fondo del área de las barras (transparente)
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

# --- 1. Carga y Preparación de Datos ---
df = pd.read_csv('Sports_Car_driver_stats.csv')
df = df[df['IRATING'] > 100]
df = df[df['STARTS'] > 3]
df = df[df['CLASS'].str.contains('D|C|B|A|P', na=False)]


df['CONTINENT'] = df['LOCATION'].apply(country_to_continent_code)

#df = df[df['IRATING'] < 10000]
df_table = df[['DRIVER','IRATING','LOCATION','STARTS','WINS']]
df_for_graphs = df.copy() # Usamos una copia completa para los gráficos
df = df[['DRIVER','IRATING','LOCATION','STARTS','WINS','AVG_START_POS','AVG_FINISH_POS','AVG_INC','TOP25PCNT']]

def flag_img(code):
    url = f"https://flagcdn.com/16x12/{code.lower()}.png"
    if code in country_flags:
        return f'![{code}]({url})'
    else:
        return f'`{code}`' 

# Aplica solo el emoji/código si está en country_flags, si no deja el valor original
df_table['LOCATION'] = df_table['LOCATION'].map(lambda x: flag_img(x) if x in country_flags else x)

#df['LOCATION'] = 'a'


histogram_irating = dcc.Graph(
    id='histogram-plot',
    style={'height': '40vh','borderRadius': '15px','overflow': 'hidden'},
    figure=create_histogram_with_percentiles(df, 'IRATING', 100)  # 100 = ancho de cada bin
)

# Tabla con paginación del lado del servidor
interactive_table = dash_table.DataTable(
    id='datatable-interactiva',
    columns=[
        {"name": i, "id": i, "presentation": "markdown",'type': 'text'} if i == "LOCATION" else {"name": i, "id": i}
        for i in df_table.columns
    ],
    data=[],  # Inicialmente vacía
    sort_action="custom",
    sort_mode="single",
    page_action="custom",
    page_current=0,
    page_size=20,
    page_count=len(df_table) // 20 + (1 if len(df_table) % 20 > 0 else 0),
    virtualization=False,
    style_as_list_view=True,
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
        'minWidth': '100px',
        'maxWidth': '200px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    style_header={
        'backgroundColor': 'rgb(17, 17, 17)',
        'fontWeight': 'bold',
        'color': 'white',
        'border': 'none',
        'textAlign': 'center'
    }
     # Renderizado virtual
)

scatter_irating_starts = dcc.Graph(
    id='scatter-irating',
    style={'height': '30vh'},
    # Usamos go.Scattergl en lugar de px.scatter para un rendimiento masivo
    figure=go.Figure(data=go.Scattergl(
        x=df['IRATING'],
        y=df['STARTS'],
        mode='markers',
        marker=dict(
            color='rgba(102, 197, 204, .3)', # Color semitransparente
            size=2,
            line=dict(width=0)
        ),
        # Desactivamos el hover para máxima velocidad
        hoverinfo='none'
    )).update_layout(
        title='Relación iRating vs. Carreras Iniciadas',
        xaxis_title='iRating',
        yaxis_title='Carreras Iniciadas (Starts)',
        template='plotly_dark'
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
        html.H1("Dashboard Interactivo - 300k Registros", style={'textAlign': 'center'}),
        html.Div([
            html.Label("Filtro rápido por país:"),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': 'Todos', 'value': 'ALL'}] + 
                       [{'label': country, 'value': country} for country in df['LOCATION'].dropna().unique()],
                value='ALL',
                style={'width': '200px', 'margin': '10px'}
            )
        ]),
        html.Div(
            style={'display': 'flex', 'flex': 1, 'minHeight': 0},
            children=[
                # Columna Izquierda (Tabla) - Ancho ajustado
                html.Div(
                    style={
                        'width': '20%',
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
                        scatter_irating_starts, # Gráfico del medio
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
        )
    ]
)
# --- 4. Callbacks ---
@app.callback(
    Output('datatable-interactiva', 'data'),
    Output('datatable-interactiva', 'page_count'),
    Output('histogram-plot', 'figure'),
    Output('continent-map', 'figure'),   # <-- AÑADE ESTA SALIDA
    Input('datatable-interactiva', 'page_current'),
    Input('datatable-interactiva', 'page_size'),
    Input('datatable-interactiva', 'sort_by'),
    Input('country-filter', 'value')
)
def update_table(page_current, page_size, sort_by, country_filter):
    # --- CORRECCIÓN AQUÍ: Usa df_table en lugar de df ---
    # df_table ya tiene las banderas en formato Markdown
    filtered_df = df_table.copy()
    
    # Para filtrar por país, necesitamos los códigos originales, no el Markdown.
    # Así que usamos el 'df' original para la lógica de filtrado.
    if country_filter != 'ALL':
        # Obtenemos los índices de los pilotos del país filtrado desde el df original
        indices_to_keep = df[df['LOCATION'] == country_filter].index
        # Aplicamos ese filtro a nuestro df_table (que tiene las banderas)
        filtered_df = df_table.loc[indices_to_keep]
    
    # Ordenar datos
    if sort_by:
        # Para ordenar, también necesitamos los valores reales, no el Markdown.
        # Creamos una copia temporal para ordenar y luego aplicamos el orden a filtered_df.
        sorting_df = df.loc[filtered_df.index] # Usamos el df original con los mismos índices
        sorting_df = sorting_df.sort_values(
            by=sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc'
        )
        # Reordenamos filtered_df para que coincida con el orden de sorting_df
        filtered_df = filtered_df.loc[sorting_df.index]

    # Calcular paginación
    start_idx = page_current * page_size
    end_idx = start_idx + page_size
    
    # Obtener datos de la página current
    page_data = filtered_df.iloc[start_idx:end_idx].to_dict('records')
    
    # Calcular número total de páginas
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
    
    # --- Lógica de actualización de gráficos ---
    graph_indices = filtered_df.index
    
    # Actualizar el histograma
    updated_histogram_figure = create_histogram_with_percentiles(df.loc[graph_indices], 'IRATING', 100)
    
    # --- ACTUALIZAR EL MAPA ---
    # Pasamos el dataframe de gráficos y el país seleccionado
    updated_map_figure = create_continent_map(df_for_graphs, country_filter)
    
    # Devuelve los 4 valores
    return page_data, total_pages, updated_histogram_figure, updated_map_figure

if __name__ == "__main__":
    app.run(debug=True)