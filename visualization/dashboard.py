# dashboard.py - Запуск веб-дашборда для интерактивного анализа результатов
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output

# Загрузка данных из CSV файлов
saturation_dfs = {}
days = [10, 50, 100]

for day in days:
    file_path = os.path.join("..", "output", f"saturation_profile_day_{day}.csv")
    if os.path.exists(file_path):
        saturation_dfs[day] = pd.read_csv(file_path)
    else:
        print(f"Файл {file_path} не найден")

# Загрузка капиллярного давления и ОФП
cap_path = os.path.join("..", "output", "capillary_table.csv")
rel_perm_path = os.path.join("..", "output", "rel_perm_table.csv")

if os.path.exists(cap_path):
    cap_df = pd.read_csv(cap_path)
else:
    print(f"Файл {cap_path} не найден")
    cap_df = None

if os.path.exists(rel_perm_path):
    rel_perm_df = pd.read_csv(rel_perm_path)
else:
    print(f"Файл {rel_perm_path} не найден")
    rel_perm_df = None

# Создание дашборда
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Интерактивный дашборд моделирования капиллярных эффектов", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Выберите параметры отображения:"),
            dcc.Dropdown(
                id='day-selector',
                options=[{'label': f'День {day}', 'value': day} for day in days],
                value=50,
                clearable=False
            ),
            html.Br(),
            dcc.RadioItems(
                id='chart-type',
                options=[
                    {'label': 'Профили насыщенности', 'value': 'profiles'},
                    {'label': 'Разница насыщенности', 'value': 'difference'},
                    {'label': 'Капиллярное давление', 'value': 'capillary'},
                    {'label': 'Относительная проницаемость', 'value': 'relperm'}
                ],
                value='profiles'
            ),
            html.Br(),
            html.Div(id='statistics-output')
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

        html.Div([
            dcc.Graph(id='main-graph')
        ], style={'width': '75%', 'display': 'inline-block'})
    ]),

    html.Div([
        dcc.Graph(id='additional-graph')
    ])
])


@callback(
    [Output('main-graph', 'figure'),
     Output('additional-graph', 'figure'),
     Output('statistics-output', 'children')],
    [Input('day-selector', 'value'),
     Input('chart-type', 'value')]
)
def update_graphs(selected_day, chart_type):
    # Основной график
    main_fig = go.Figure()

    # Дополнительный график
    add_fig = go.Figure()

    # Статистика
    stats_output = []

    if chart_type == 'profiles' and selected_day in saturation_dfs:
        df = saturation_dfs[selected_day]

        # Основной график: профили насыщенности
        main_fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=('Без капиллярных эффектов', 'С капиллярными эффектами'))

        main_fig.add_trace(
            go.Scatter(x=df['Distance_m'], y=df['Sw_without_cap'], mode='lines', name='Без кап. эффектов'),
            row=1, col=1
        )

        main_fig.add_trace(
            go.Scatter(x=df['Distance_m'], y=df['Sw_with_cap'], mode='lines', name='С кап. эффектами'),
            row=1, col=2
        )

        main_fig.update_layout(
            title=f'Профили водонасыщенности на {selected_day}-й день',
            xaxis_title='Расстояние, м',
            yaxis_title='Водонасыщенность, д.ед.'
        )

        # Дополнительный график: сравнение профилей
        add_fig.add_trace(
            go.Scatter(x=df['Distance_m'], y=df['Sw_without_cap'], mode='lines', name='Без кап. эффектов')
        )

        add_fig.add_trace(
            go.Scatter(x=df['Distance_m'], y=df['Sw_with_cap'], mode='lines', name='С кап. эффектами')
        )

        add_fig.update_layout(
            title=f'Сравнение профилей водонасыщенности на {selected_day}-й день',
            xaxis_title='Расстояние, м',
            yaxis_title='Водонасыщенность, д.ед.'
        )

        # Статистика
        avg_without = df['Sw_without_cap'].mean()
        avg_with = df['Sw_with_cap'].mean()
        max_diff = (df['Sw_with_cap'] - df['Sw_without_cap']).abs().max()

        stats_output = [
            html.H4("Статистика:"),
            html.P(f"Средняя насыщенность (без кап.): {avg_without:.3f}"),
            html.P(f"Средняя насыщенность (с кап.): {avg_with:.3f}"),
            html.P(f"Разница средних: {avg_with - avg_without:.3f}"),
            html.P(f"Максимальная разница: {max_diff:.3f}")
        ]

    elif chart_type == 'difference' and selected_day in saturation_dfs:
        df = saturation_dfs[selected_day]
        diff = df['Sw_with_cap'] - df['Sw_without_cap']

        # Основной график: разница насыщенности
        main_fig.add_trace(
            go.Scatter(x=df['Distance_m'], y=diff, mode='lines', fill='tozeroy',
                       line=dict(color='purple'))
        )

        main_fig.update_layout(
            title=f'Разница водонасыщенности на {selected_day}-й день',
            xaxis_title='Расстояние, м',
            yaxis_title='Разница (с кап. - без кап.), д.ед.'
        )

        # Дополнительный график: тепловая карта разницы по дням
        if len(saturation_dfs) > 1:
            diff_data = []
            for day in saturation_dfs.keys():
                day_df = saturation_dfs[day]
                day_diff = day_df['Sw_with_cap'] - day_df['Sw_without_cap']
                diff_data.append(day_diff.values)

            x = saturation_dfs[selected_day]['Distance_m']
            y = list(saturation_dfs.keys())

            add_fig = px.imshow(diff_data,
                                x=x,
                                y=y,
                                labels=dict(x="Расстояние, м", y="День", color="Разница"),
                                title="Эволюция разницы водонасыщенности")

        # Статистика
        positive_diff = diff[diff > 0]
        negative_diff = diff[diff < 0]

        stats_output = [
            html.H4("Статистика разницы:"),
            html.P(f"Средняя разница: {diff.mean():.3f}"),
            html.P(f"Максимальная положительная: {diff.max():.3f}"),
            html.P(f"Максимальная отрицательная: {diff.min():.3f}"),
            html.P(f"Количество точек с положительной разницей: {len(positive_diff)}"),
            html.P(f"Количество точек с отрицательной разницей: {len(negative_diff)}")
        ]

    elif chart_type == 'capillary' and cap_df is not None:
        # Основной график: капиллярное давление
        main_fig.add_trace(
            go.Scatter(x=cap_df['Sw'], y=cap_df['Pc_MPa'], mode='lines+markers')
        )

        main_fig.update_layout(
            title='Кривая капиллярного давления',
            xaxis_title='Водонасыщенность, д.ед.',
            yaxis_title='Капиллярное давление, МПа'
        )

        # Дополнительный график: логарифмический масштаб
        valid_data = cap_df[(cap_df['Sw'] > 0.2) & (cap_df['Sw'] < 0.8) & (cap_df['Pc_MPa'] > 0)]

        add_fig.add_trace(
            go.Scatter(x=valid_data['Sw'], y=valid_data['Pc_MPa'], mode='markers',
                       marker=dict(size=10))
        )

        add_fig.update_layout(
            title='Капиллярное давление (логарифмический масштаб)',
            xaxis_title='Водонасыщенность, д.ед.',
            yaxis_title='Капиллярное давление, МПа',
            yaxis_type="log"
        )

        # Статистика
        stats_output = [
            html.H4("Статистика кап. давления:"),
            html.P(f"Максимальное давление: {cap_df['Pc_MPa'].max():.3f} МПа"),
            html.P(f"Среднее давление: {cap_df['Pc_MPa'].mean():.3f} МПа"),
            html.P(f"Давление при Sw=0.5: {cap_df.loc[cap_df['Sw'].sub(0.5).abs().idxmin(), 'Pc_MPa']:.3f} МПа")
        ]

    elif chart_type == 'relperm' and rel_perm_df is not None:
        # Основной график: относительная проницаемость
        main_fig.add_trace(
            go.Scatter(x=rel_perm_df['Sw'], y=rel_perm_df['Krw'], mode='lines+markers', name='Krw (вода)')
        )

        main_fig.add_trace(
            go.Scatter(x=rel_perm_df['Sw'], y=rel_perm_df['Kro'], mode='lines+markers', name='Kro (нефть)')
        )

        main_fig.update_layout(
            title='Кривые относительной проницаемости',
            xaxis_title='Водонасыщенность, д.ед.',
            yaxis_title='Относительная проницаемость, д.ед.'
        )

        # Дополнительный график: отношение подвижностей
        mu_w = 1.0
        mu_o = 5.0
        mobility_ratio = np.zeros(len(rel_perm_df))

        for i in range(len(rel_perm_df)):
            if rel_perm_df['Kro'].iloc[i] > 0:
                mobility_ratio[i] = (rel_perm_df['Krw'].iloc[i] / mu_w) / (rel_perm_df['Kro'].iloc[i] / mu_o)
            else:
                mobility_ratio[i] = np.nan

        add_fig.add_trace(
            go.Scatter(x=rel_perm_df['Sw'], y=mobility_ratio, mode='lines+markers')
        )

        add_fig.update_layout(
            title='Отношение подвижностей',
            xaxis_title='Водонасыщенность, д.ед.',
            yaxis_title='Отношение подвижностей',
            yaxis_type="log"
        )

        # Статистика
        crossover_idx = np.abs(rel_perm_df['Krw'] - rel_perm_df['Kro']).argmin()
        crossover_sw = rel_perm_df['Sw'].iloc[crossover_idx]

        stats_output = [
            html.H4("Статистика ОФП:"),
            html.P(f"Точка пересечения кривых: Sw = {crossover_sw:.3f}"),
            html.P(f"Krw при Sw=0.5: {rel_perm_df.loc[rel_perm_df['Sw'].sub(0.5).abs().idxmin(), 'Krw']:.3f}"),
            html.P(f"Kro при Sw=0.5: {rel_perm_df.loc[rel_perm_df['Sw'].sub(0.5).abs().idxmin(), 'Kro']:.3f}")
        ]

    return main_fig, add_fig, stats_output


if __name__ == '__main__':
    app.run_server(debug=True)