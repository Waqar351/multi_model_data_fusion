import os
import numpy as np
import pandas as pd
import math
from statistics import mean
from sklearn.manifold import TSNE
from datetime import date
from dash import Dash
from dash import dcc, ctx, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from scipy.stats import wasserstein_distance
import base64
import io
import time


df = pd.read_csv("visualization/data/tsne2.csv")

input_features = pd.read_csv("visualization/data/data_subset.csv")

# Lista de columnas
cols = input_features.columns.tolist()

# Encontrar la posición de la columna '2017.12'
cutoff_index = cols.index('2017.12')

# Construir los conjuntos de columnas
cols_df1 = ['Nodo'] + cols[1:cutoff_index + 1]  # Nodo + columnas hasta 2017.12
cols_df2 = ['Nodo'] + [col for col in cols if col not in cols_df1 and col != 'Nodo']

# Crear los DataFrames
input_features_og = input_features[cols_df1]
input_features = input_features[cols_df2]

df['sum'] = input_features_og.iloc[:, 1:].sum(axis=1)

# 2. Categorize in 2 groups (0 and greater than 0)
def categorize(valor):
    if valor == 0:
        return 0
    else:
        return 1

df['category'] = df['sum'].apply(categorize)
df["category"] = df["category"].astype(str)


app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
server = app.server
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# HTML template for Dash
html_dash = """
<!DOCTYPE html>
<html>
   <head>
       {%metas%}
       <title>{%title%}</title>
       {%favicon%}
       {%css%}
       <style>
            .DateInput_input__focused {
                border-bottom: 2px solid #0175ff;
            }
            .DateInput_input, .DateInput_input_1 {
                font-family: Arial;
                font-size: 10px; 
                height: 10px;
            }
            .DateInput, .DateInput_1 {
                width: 40%;
                height: inherit;
            }
       </style>
   </head>
   <body>
       {%app_entry%}
       <footer>
           {%config%}
           {%scripts%}
           {%renderer%}
       </footer>
   </body>
</html>
"""
selected_corners_indices = []
server = app.server
app.index_string = html_dash
df_backup = df.copy()

# Default map configuration
fig = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name=df.index, hover_data=df.columns[:-2],
                        color="category",
                        color_discrete_map={
                            "0": "#0000FF",
                            "1": "red"
                        }, zoom=3, height=300,mapbox_style="carto-positron")
fig.update_layout(mapbox_style="open-street-map",
                  margin={"r": 0, "t": 0, "l": 0, "b": 0},
                  autosize=True,
                  mapbox={'center': {'lon': -46.625290, 'lat': -23.533773}, 'zoom': 10},
                  height=450,
                  showlegend=False)
fig_time_series = go.Figure()
fig_time_series.update_layout(title='Styled Scatter')

# Default scatters
# Suponiendo que tienes columnas: u1, v1, u2, v2, ..., u5, v5
scatters = []  # Para guardar las figuras

for i in range(1, 6):
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(
        x=df[f'u{i}'],
        y=df[f'v{i}'],
        mode='markers',
        marker=dict(color='#00a699', size=3, opacity=0.5)
    ))
    fig2.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        dragmode='select',
        clickmode='event+select'
    )
    scatters.append(fig2)

# Ahora tienes 5 figuras: scatters[0] hasta scatters[4]

for i in range(5):
    scatters[i].update_layout(height=300)  # altura fija

# Dash layout con height estático
app.layout = dbc.Container(
    [
        html.H1(""),
        html.H4(children='Visual Explorer', className="text-center"),
        html.Hr(),

        dbc.Row(html.H5(children='Projection View'), className="mb-3"),

        dbc.Row(
            [
                dbc.Col([html.H6('M1 Static'), dcc.Graph(id="projectionScatter1", figure=scatters[0], style={'height':'300px'})], width=2),
                dbc.Col([html.H6('M1 Dynamic'), dcc.Graph(id="projectionScatter2", figure=scatters[1], style={'height':'300px'})], width=2),
                dbc.Col([html.H6('M2'), dcc.Graph(id="projectionScatter3", figure=scatters[2], style={'height':'300px'})], width=2),
                dbc.Col([html.H6('M3'), dcc.Graph(id="projectionScatter4", figure=scatters[3], style={'height':'300px'})], width=3),
                dbc.Col([html.H6('M4'), dcc.Graph(id="projectionScatter5", figure=scatters[4], style={'height':'300px'})], width=3),
            ]
        ),

        dbc.Row([
            dbc.Col([html.H5("Map View"), dcc.Graph(id="mymap", figure=fig, style={'height':'450px'})], width=6),
            dbc.Col([
                html.H5("Discrete Features Bar Plots"),
                dbc.Row([dbc.Col(dcc.Graph(id="barplot_estacao_metro", style={'height':'300px'}), width=6),
                         dbc.Col(dcc.Graph(id="barplot_estacao_trem", style={'height':'300px'}), width=6)]),
                dbc.Row([dbc.Col(dcc.Graph(id="barplot_terminal_onibus", style={'height':'300px'}), width=6),
                         dbc.Col(dcc.Graph(id="barplot_favela_proxima", style={'height':'300px'}), width=6)]),
            ], width=6)
        ]),
        html.H5(children='Features Box Plots'),
        dbc.Row([   
            dbc.Col(dcc.Graph(id="boxplot_census_global", style={'height':'400px'}), width=4),
            dbc.Col(dcc.Graph(id="boxplot_census_selected", style={'height':'400px'}), width=4),
            dbc.Col(dcc.Graph(id="boxplot_census_combined", style={'height':'400px'}), width=4),
        ]),
        dbc.Row([   
            dbc.Col([html.H5('Time Series Crime'), dcc.Graph(id='time_series_crime', figure=fig_time_series, style={'height':'300px'})], width=12),
        ]),

        dcc.Store(id="features-detail-df"),
        dcc.Store(id="features-general-df"),
        dcc.Store(id="input_features"),
    ],
    fluid=True
)
global dataframes_array

# Projection View

# ## Mixed
def scatterplot(x, y, title="", axis_type="Linear", width=650, height=650):  # puedes borrar width y height
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode='markers',
        marker=dict(color='#00a699', size=5, opacity=0.5)))
    #fig.update_layout(title=title, xaxis_type=axis_type, yaxis_type=axis_type, width=width, height=height)
    return fig


@app.callback(
    Output("barplot_estacao_metro", "figure"),
    Output("barplot_estacao_trem", "figure"),
    Output("barplot_terminal_onibus", "figure"),
    Output("barplot_favela_proxima", "figure"),
    Output('time_series_crime', 'figure'),
    Output("boxplot_census_combined", "figure"),
    Output("boxplot_census_global", "figure"),
    Output("boxplot_census_selected", "figure"),
    Output("projectionScatter1", "figure"),
    Output("projectionScatter2", "figure"),
    Output("projectionScatter3", "figure"),
    Output("projectionScatter4", "figure"),
    Output("projectionScatter5", "figure"),
    Output("mymap", "figure"),
    Output('features-detail-df', 'data'),
    Output('features-general-df', 'data'),
    [
        Input("projectionScatter1", "selectedData"),
        Input("projectionScatter2", "selectedData"),
        Input("projectionScatter3", "selectedData"),
        Input("projectionScatter4", "selectedData"),
        Input("projectionScatter5", "selectedData"),
        Input("mymap", "selectedData")
    ], prevent_initial_call=True
)
def update_cb(selectedData1, selectedData2, selectedData3, selectedData4, selectedData5, selectedDataMap):
    global selected_corners_indices

    triggered_id = ctx.triggered_id
    selected_indices = []

    if triggered_id == "mymap" and selectedDataMap:
        selected_indices = list({pt["hovertext"] for pt in selectedDataMap["points"]})
    elif "projectionScatter" in str(triggered_id):
        selectedData = eval(triggered_id.replace("projectionScatter", "selectedData"))
        if selectedData:
            selected_indices = list({pt["pointIndex"] for pt in selectedData["points"] if pt["curveNumber"] == 0})

    if not selected_indices:
        selected_corners_indices = list(df.index)
        temp = df
        temp_census = input_features
    else:
        selected_corners_indices = selected_indices
        temp = df.iloc[selected_indices]
        temp_census = input_features.iloc[selected_indices]

    selected_corners_indices = selected_indices

    feature_name_map = {
        'Estacao_de_metro': 'Subway Station Nearby',
        'Estacao_de_trem': 'Train Station Nearby',
        'Terminal_de_onibus': 'Bus Terminal Nearby',
        'Favela_proxima': 'Favela Nearby',
        'Renda_media_por_domicilio': 'Income per Household',
        'Renda_media_responsaveis': 'Income per Responsible',
        'Responsaveis_sem_renda_taxa': '# Household heads with no income',
        'Alfabetizados_de_7_a_15_anos': '% Literate Age 7 to 15',
        'menores_de_18_anos_taxa': '% Under 18',
        '18_a_65_anos_taxa': '% 18 to 65',
        'maiores_de_65_anos_taxa': '% Over 65'
    }

    categorical_cols = list(feature_name_map.keys())[:4]
    barplot_figs = []

    for col in categorical_cols:
        global_counts = input_features[col].value_counts().sort_index()
        selected_counts = temp_census[col].value_counts().sort_index()
        index_union = sorted(set(global_counts.index).union(set(selected_counts.index)))

        global_values = [global_counts.get(i, 0) for i in index_union]
        selected_values = [selected_counts.get(i, 0) for i in index_union]
        bin_labels = ['No', 'Yes'] if index_union == [0, 1] else [str(i) for i in index_union]
        title = feature_name_map[col]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=bin_labels, y=global_values, name="Global", marker_color="blue", opacity=0.6))
        fig_bar.add_trace(go.Bar(x=bin_labels, y=selected_values, name="Selected", marker_color="red", opacity=0.6))
        fig_bar.update_layout(
            barmode='group',
            title=title,
            height=300,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            xaxis=dict(title="Value"),
            yaxis=dict(title="Count")
        )
        barplot_figs.append(fig_bar)

    def translate_label(label):
        return feature_name_map.get(label, label.replace("_", " ").title())

    temp = df.iloc[selected_indices]
    temp_census = input_features.iloc[selected_indices]
    cols_to_plot = [col for col in input_features.columns if col not in ['Pontos_de_onibus', 'lat', 'long', 'Nodo']]

    def normalize(series,original_series):
        series = series.dropna()
        if series.empty:
            return series
        min_val = original_series.min()
        max_val = original_series.max()
        if max_val > min_val:
            return (series - min_val) / (max_val - min_val)
        else:
            return series * 0
    # Altura y eje y uniforme para los tres plots
    common_layout = dict(
        height=400,
        margin={'l': 0, 't': 40, 'b': 30, 'r': 0},
        yaxis=dict(range=[0, 1.05])
    )

    # ========= 1. GLOBAL FEATURES =========
    fig_box_global = go.Figure()
    for col in cols_to_plot:
        col_data = input_features[col]
        y_norm = normalize(col_data, col_data)

        q1 = np.percentile(col_data, 25)
        q2 = np.percentile(col_data, 50)
        q3 = np.percentile(col_data, 75)

        hover_info = (
            f"<b>{translate_label(col)}</b><br>"
            f"Q1: {q1:.2f}<br>"
            f"Median: {q2:.2f}<br>"
            f"Q3: {q3:.2f}<br>"
        )

        fig_box_global.add_trace(go.Box(
            y=y_norm,
            name=translate_label(col),
            customdata=np.expand_dims(col_data, axis=1),
            hovertext=[hover_info]*len(col_data),
            hovertemplate='%{hovertext}<br>Original: %{customdata[0]:.3f}<extra></extra>',
            boxpoints='outliers',
            marker_color='blue'
        ))

    fig_box_global.update_layout(title="Global Census Features", **common_layout)

    # ========= 2. SELECTED FEATURES =========
    fig_box_selected = go.Figure()
    for col in cols_to_plot:
        col_data = temp_census[col].dropna()
        y_norm = normalize(col_data, input_features[col])

        q1 = np.percentile(col_data, 25)
        q2 = np.percentile(col_data, 50)
        q3 = np.percentile(col_data, 75)

        hover_info = (
            f"<b>{translate_label(col)} (selected)</b><br>"
            f"Q1: {q1:.2f}<br>"
            f"Median: {q2:.2f}<br>"
            f"Q3: {q3:.2f}<br>"
        )

        fig_box_selected.add_trace(go.Box(
            y=y_norm,
            name=translate_label(col),
            customdata=np.expand_dims(col_data, axis=1),
            hovertext=[hover_info]*len(col_data),
            hovertemplate='%{hovertext}<br>Original: %{customdata[0]:.3f}<extra></extra>',
            boxpoints='outliers',
            marker_color='red'
        ))

    fig_box_selected.update_layout(title="Selected Census Features", **common_layout)

    # ========= 3. COMPARACIÓN =========
    fig_box_combined = go.Figure()

    for i, col in enumerate(cols_to_plot):
        translated_name = translate_label(col)

        col_global = input_features[col]
        col_selected = temp_census[col].dropna()

        y_norm_global = normalize(col_global, col_global)
        y_norm_selected = normalize(col_selected, col_global)

        q1_g, q2_g, q3_g = np.percentile(col_global, [25, 50, 75])
        q1_s, q2_s, q3_s = np.percentile(col_selected, [25, 50, 75])

        hover_global = (
            f"<b>{translated_name} (Global)</b><br>"
            f"Q1: {q1_g:.2f}<br>Median: {q2_g:.2f}<br>Q3: {q3_g:.2f}<br>"
        )
        hover_selected = (
            f"<b>{translated_name} (Selected)</b><br>"
            f"Q1: {q1_s:.2f}<br>Median: {q2_s:.2f}<br>Q3: {q3_s:.2f}<br>"
        )

        x_pos = i

        # 📦 Boxplot Global
        fig_box_combined.add_trace(go.Box(
            y=y_norm_global,
            x=[x_pos]*len(y_norm_global),
            name="Global (box)",
            legendgroup="global_box",
            showlegend=(i == 0),
            marker_color='blue',
            boxpoints='outliers',
        ))

        # 🔵 Scatter Global
        x_jitter_global = x_pos - 0.2 + np.random.uniform(-0.07, 0.07, size=len(y_norm_global))
        fig_box_combined.add_trace(go.Scatter(
            y=y_norm_global,
            x=x_jitter_global,
            mode='markers',
            name="Global (points)",
            legendgroup="global_points",
            showlegend=(i == 0),
            customdata=np.expand_dims(col_global, axis=1),
            hovertext=[hover_global]*len(col_global),
            hovertemplate='%{hovertext}<br>Original: %{customdata[0]:.3f}<extra></extra>',
            marker=dict(color='blue', size=6, opacity=0.1, symbol='circle')
        ))

        # 📦 Boxplot Selected
        fig_box_combined.add_trace(go.Box(
            y=y_norm_selected,
            x=[x_pos]*len(y_norm_selected),
            name="Selected (box)",
            legendgroup="selected_box",
            showlegend=(i == 0),
            marker_color='red',
            boxpoints='outliers',
        ))

        # 🔴 Scatter Selected
        x_jitter_selected = x_pos - 0.2 + np.random.uniform(-0.07, 0.07, size=len(y_norm_selected))
        fig_box_combined.add_trace(go.Scatter(
            y=y_norm_selected,
            x=x_jitter_selected,
            mode='markers',
            name="Selected (points)",
            legendgroup="selected_points",
            showlegend=(i == 0),
            customdata=np.expand_dims(col_selected, axis=1),
            hovertext=[hover_selected]*len(col_selected),
            hovertemplate='%{hovertext}<br>Original: %{customdata[0]:.3f}<extra></extra>',
            marker=dict(color='red', size=6, opacity=0.1, symbol='circle')
        ))

    fig_box_combined.update_layout(
        title="Comparison Global vs Selected (Normalized)",
        showlegend=True,
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(cols_to_plot))),
            ticktext=[translate_label(col) for col in cols_to_plot]
        ),
        margin={'l': 0, 't': 40, 'b': 30, 'r': 0},
        height=400,
    )


    scatters = []
    for i in range(1, 6):
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=getattr(df, f'u{i}'),
            y=getattr(df, f'v{i}'),
            mode='markers',
            marker=dict(color='#00a699', size=5, opacity=0.5)
        ))
        fig.add_trace(go.Scattergl(
            x=getattr(temp, f'u{i}'),
            y=getattr(temp, f'v{i}'),
            mode='markers',
            marker=dict(color='red', size=5, opacity=0.5)
        ))
        fig.update_layout(clickmode='event+select', dragmode='select', margin={'l': 0, 't': 0, 'b': 0, 'r': 0}, showlegend=False)
        scatters.append(fig)

    crime_df_sel = input_features_og.iloc[selected_corners_indices].iloc[:, 1:]
    dates = pd.date_range(start='2006-01-01', end='2017-12-31', freq='MS')
    fig_time_series = go.Figure()
    if not crime_df_sel.empty:
        avg_values = crime_df_sel.mean()
        avg_crimes = pd.DataFrame({'Date': dates, 'Avg_value': avg_values.values})
        for i, (index, row) in enumerate(crime_df_sel.iterrows()):
            try:
                series_mean = row.values.mean()
                opacity = min(max(series_mean / 10, 0.1), 0.6)
                fig_time_series.add_trace(go.Scatter(
                    x=dates,
                    y=row.values,
                    mode='lines',
                    name='Point',
                    line=dict(color=f'rgba(0, 0, 255, {opacity})'),
                    legendgroup='group1',
                    showlegend=(i == 0),
                    hovertext=[f"Point ID: {index}<br>Date: {d.strftime('%Y-%m')}<br>Value: {v:.2f}" for d, v in zip(dates, row.values)],
                    hoverinfo='text'
                ))
            except Exception as e:
                print(f"Error processing row {index}: {e}")
        fig_time_series.add_trace(go.Scatter(x=avg_crimes['Date'], y=avg_crimes['Avg_value'], mode='lines', name='Average', line=dict(color='red')))
    fig_time_series.update_layout(height=300, margin={'l': 0, 't': 30, 'b': 0, 'r': 0}, title="Crime Time Series - Selected")

    mapfig = px.scatter_mapbox(
        temp, lat="lat", lon="lon",
        hover_name=temp.index,
        hover_data=temp.columns[:-2],
        color="category",
        color_discrete_map={"0": "#0000FF", "1": "red"},
        zoom=3, height=300,
        mapbox_style="carto-positron"
    )
    mapfig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0}, autosize=True,
                         mapbox={'center': {'lon': -46.625290, 'lat': -23.533773}, 'zoom': 10}, height=450)

    return (
        *barplot_figs,
        fig_time_series,
        fig_box_combined, fig_box_global, fig_box_selected,
        *scatters,
        mapfig,
        temp.to_json(date_format='iso', orient='split'),
        df.to_json(date_format='iso', orient='split')
    )




if __name__ == "__main__":
    #app.run_server(debug=True,port=8053)
    #app.run_server(debug=True,host='0.0.0.0',port=8070)
    app.run(host='0.0.0.0',port=8060)
    #app.run_server()