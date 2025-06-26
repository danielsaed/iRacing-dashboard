import dash
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd

# --- 1. Carga y Preparación de Datos ---
#df = px.data.gapminder().query("year == 2007")

df = pd.read_csv('Sports_Car_driver_stats.csv')

# --- 2. Creación de Componentes Gráficos ---

# Gráfico de dispersión con tema oscuro
scatter_plot = dcc.Graph(
        id='scatter-plot',
    style={'height': '80vh'},
    figure=px.scatter(
        df, x="IRATING",
        y='STARTS',
        hover_name="DRIVER",
        template='plotly_dark'  # <-- Tema oscuro para el gráfico
    )
)

# Tabla de datos interactiva con tema oscuro
interactive_table = dash_table.DataTable(
    id='datatable-interactiva',
    columns=[
        {"name": i, "id": i} for i in df.columns
    ],
    data=df.to_dict('records'),
    sort_action="native",
    page_action="native",
    page_size=30,
    style_table={'overflowX': 'auto'},
    style_as_list_view=True,
    style_cell={
        'textAlign': 'left',
        'padding': '5px',
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white'
    },
    style_header={
        'backgroundColor': 'rgb(17, 17, 17)',
        'fontWeight': 'bold',
        'color': 'white',
        'border': 'none'
    }
)

# --- 3. Inicialización de la App y Definición del Layout ---
app = dash.Dash(__name__)

# Layout principal con fondo oscuro
app.layout = html.Div(children=[
    html.H1("Dashboard Interactivo", style={'textAlign': 'center'}),

    # Contenedor Flex para las columnas
    html.Div(
        style={'display': 'flex'},  # Activa el layout flexible
        children=[
            # Columna Izquierda (Tabla)
            html.Div(
                style={'width': '40%', 'padding': '10px'},
                children=[
                    html.H2("Tabla de Países"),
                    interactive_table
                ]
            ),
            # Columna Derecha (Gráfico)
            html.Div(
                style={'width': '60%', 'padding': '10px'},
                children=[
                    scatter_plot
                ]
            )
        ]
    )
])

# --- 4. Ejecución del Servidor ---
if __name__ == "__main__":
    app.run(debug=True)