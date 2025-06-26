import dash
from dash import html, dcc, dash_table, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def create_histogram_with_percentiles(df, column='IRATING', nbins=400):
    hist, bin_edges = np.histogram(df[column], bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    total = len(df)
    hover_text = []
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
    fig = go.Figure(data=go.Bar(
        x=bin_centers,
        y=hist,
        width=bin_widths * 0.9,
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        marker=dict(color=hist, colorscale='Viridis')
    ))
    fig.update_layout(
        title='Distribución de iRating',
        xaxis_title='iRating',
        yaxis_title='Cantidad de Pilotos',
        template='plotly_dark'
    )
    return fig

# --- 1. Carga y Preparación de Datos ---
df = pd.read_csv('Sports_Car_driver_stats.csv')
df = df[df['IRATING'] > 100]
df = df[df['IRATING'] < 10000]
# --- 2. Creación de Componentes Gráficos ---

# Gráfico de dispersión con tema oscuro
'''scatter_plot = dcc.Graph(
    id='scatter-plot',
    style={'height': '50vh'},
    figure=px.histogram(
        df, x="IRATING",
         nbins=400,
        hover_name="DRIVER",
        template='plotly_dark'
    )
)'''
scatter_plot = dcc.Graph(
    id='scatter-plot',
    style={'height': '50vh'},
    figure=create_histogram_with_percentiles(df, 'IRATING', 400)
)

# Tabla con paginación del lado del servidor
interactive_table = dash_table.DataTable(
    id='datatable-interactiva',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=[],  # Inicialmente vacía
    sort_action="custom",  # Ordenamiento del lado del servidor
    sort_mode="single",
    page_action="custom",  # Paginación del lado del servidor
    page_current=0,
    page_size=50,
    page_count=len(df) // 50 + (1 if len(df) % 50 > 0 else 0),
    style_table={'overflowX': 'auto', 'height': '70vh'},
    style_as_list_view=True,
    style_cell={
        'textAlign': 'left',
        'padding': '5px',
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white',
        'minWidth': '100px',
        'maxWidth': '200px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    style_header={
        'backgroundColor': 'rgb(17, 17, 17)',
        'fontWeight': 'bold',
        'color': 'white',
        'border': 'none'
    },
    virtualization=True  # Renderizado virtual
)

# --- 3. Inicialización de la App ---
app = dash.Dash(__name__)

# Layout principal
app.layout = html.Div(children=[
    html.H1("Dashboard Interactivo - 300k Registros", style={'textAlign': 'center'}),
    
    # Controles adicionales
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

    # Contenedor Flex para las columnas
    html.Div(
        style={'display': 'flex'},
        children=[
            # Columna Izquierda (Tabla)
            html.Div(
                style={'width': '50%', 'padding': '10px'},
                children=[
                    html.H2("Tabla de Datos"),
                    interactive_table
                ]
            ),
            # Columna Derecha (Gráfico)
            html.Div(
                style={'width': '50%', 'padding': '10px'},
                children=[
                    scatter_plot
                ]
            )
        ]
    )
])

# --- 4. Callbacks ---
@app.callback(
    Output('datatable-interactiva', 'data'),
    Output('datatable-interactiva', 'page_count'),
    Output('scatter-plot', 'figure'),  # <-- Añade esta línea
    Input('datatable-interactiva', 'page_current'),
    Input('datatable-interactiva', 'page_size'),
    Input('datatable-interactiva', 'sort_by'),
    Input('country-filter', 'value')
)
def update_table(page_current, page_size, sort_by, country_filter):
    # Filtrar datos si es necesario
    filtered_df = df.copy()
    if country_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['LOCATION'] == country_filter]
    
    # Ordenar datos
    if sort_by:
        filtered_df = filtered_df.sort_values(
            by=sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc'
        )
    
    # Calcular paginación
    start_idx = page_current * page_size
    end_idx = start_idx + page_size
    
    # Obtener datos de la página actual
    page_data = filtered_df.iloc[start_idx:end_idx].to_dict('records')
    
    # Calcular número total de páginas
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
    
    # Actualizar el histograma con la data filtrada
    updated_figure = create_histogram_with_percentiles(filtered_df, 'IRATING', 400)
    
    return page_data, total_pages, updated_figure  # <-- Devuelve también el nuevo gráfico


if __name__ == "__main__":
    app.run(debug=True)