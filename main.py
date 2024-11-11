import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import skew, kurtosis, sem, t

# Загрузка данных
data = pd.read_csv('data.csv').drop(columns=['ID'])

# Предобработка данных
data = data.dropna()
numeric_cols = data.select_dtypes(include='number').columns

# KMeans кластеризация
kmeans = KMeans(n_clusters=3, random_state=0)
data['cluster'] = kmeans.fit_predict(data[numeric_cols])

from minisom import MiniSom

# Инициализация и обучение SOM
som_size = 10  # Размер карты SOM (10x10)
som = MiniSom(som_size, som_size, len(numeric_cols), sigma=1.0, learning_rate=0.5, random_seed=0)
som.train_random(data[numeric_cols].values, 100)  # 100 - число итераций

# Назначение кластера каждому объекту данных на основе SOM
som_labels = []
for sample in data[numeric_cols].values:
    winning_node = som.winner(sample)
    som_labels.append(winning_node[0] * som_size + winning_node[1])  # Преобразование узлов в кластерные метки
data['som_cluster'] = som_labels

unique_clusters = sorted(set(som_labels))
sequential_labels = {cluster: i for i, cluster in enumerate(unique_clusters)}
data['sequential_som_cluster'] = data['som_cluster'].map(sequential_labels)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

selected_columns = ['age', 'Global', 'lacunes_num', 'SVD Amended Score', 'lac_count']

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Анализ распределения и кластеризация", className="card-title"),
                    html.P("Выберите параметр для анализа его распределения и статистики, а также для кластеризации.",
                           className="card-text"),
                    dcc.Dropdown(
                        id='dropdown-parameter',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        value=numeric_cols[0],
                        clearable=False,
                        style={'margin-bottom': '10px'}
                    ),
                    dcc.Graph(id='histogram-parameter'),
                    html.Div(id='stats-output', style={'margin-top': '10px'})
                ])
            ], className="mb-4")
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Таблица данных по отобранным параметрам", className="card-title"),
                    html.P("Просмотр ключевых параметров для каждого пациента.", className="card-text"),
                    dash_table.DataTable(
                        id='table-data',
                        columns=[{"name": i, "id": i} for i in selected_columns],
                        data=data[selected_columns].to_dict('records'),
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ])
            ])
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Кластеризация K-средних"),
                    dcc.Graph(id='scatter-cluster')
                ])
            ], className="mb-4")
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Просмотр кластеров", className="card-title"),
                    html.P("Выберите кластер для анализа его характеристик.", className="card-text"),
                    dcc.Dropdown(
                        id='dropdown-cluster',
                        options=[{'label': f'Кластер {i}', 'value': i} for i in range(kmeans.n_clusters)],
                        value=0,
                        clearable=False,
                        style={'margin-top': '10px'}
                    ),
                    dcc.Graph(id='cluster-details')
                ])
            ])
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("SOM кластеризация", className="card-title"),
                    html.P("Анализ кластеров, найденных с помощью самоорганизующейся карты (SOM).",
                           className="card-text"),
                    dcc.Dropdown(
                        id='dropdown-som-cluster',
                        options=[{'label': f'Кластер {i}', 'value': i} for i in sorted(sequential_labels.values())],
                        value=0,
                        clearable=False,
                        style={'margin-top': '10px'}
                    ),
                    dcc.Graph(id='som-cluster-details')
                ])
            ])
        ], width=12)
    ])
], fluid=True)


@app.callback(
    Output('som-cluster-details', 'figure'),
    [Input('dropdown-som-cluster', 'value')]
)
def update_som_cluster_details(sequential_som_cluster):
    filtered_data = data[data['sequential_som_cluster'] == sequential_som_cluster]
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Box(y=filtered_data[col], name=col))
    fig.update_layout(title=f'Детализация для SOM кластера {sequential_som_cluster}')
    return fig


# Функция для расчета статистических характеристик
def calculate_statistics(data, parameter):
    n = len(data)
    mean = data.mean()
    std_dev = data.std()
    sem_value = sem(data)  # Ошибка среднего значения
    median = data.median()
    skewness = skew(data)
    kurt = kurtosis(data, fisher=False)

    # Доверительный интервал
    confidence = 0.95
    h = sem_value * t.ppf((1 + confidence) / 2., n - 1)
    ci_lower, ci_upper = mean - h, mean + h

    stats_dict = {
        "Среднее значение": mean,
        "Стандартное отклонение": std_dev,
        "Ошибка среднего значения": sem_value,
        "Медиана": median,
        "Асимметрия": skewness,
        "Эксцесс": kurt,
        "Доверительный интервал": f"[{ci_lower:.2f}, {ci_upper:.2f}]"
    }
    return stats_dict


# Гистограмма и статистика
@app.callback(
    [Output('histogram-parameter', 'figure'), Output('stats-output', 'children')],
    [Input('dropdown-parameter', 'value')]
)
def update_histogram(parameter):
    fig = px.histogram(data, x=parameter, title=f'Распределение {parameter}')

    stats = calculate_statistics(data[parameter], parameter)
    stats_html = html.Ul([html.Li(f'{stat}: {value}') for stat, value in stats.items()])

    return fig, stats_html


# Визуализация кластеров
@app.callback(
    Output('scatter-cluster', 'figure'),
    [Input('dropdown-parameter', 'value')]
)
def update_clusters(parameter):
    fig = px.scatter(data, x=parameter, y=numeric_cols[1], color='cluster', title='Кластеризация K-средних')
    return fig


# Детализация для выбранного кластера
@app.callback(
    Output('cluster-details', 'figure'),
    [Input('dropdown-cluster', 'value')]
)
def update_cluster_details(cluster):
    filtered_data = data[data['cluster'] == cluster]
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Box(y=filtered_data[col], name=col))
    fig.update_layout(title=f'Детализация для кластера {cluster}')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
