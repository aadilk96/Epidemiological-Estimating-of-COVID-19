import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_table
import plotly.graph_objects as go
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import datetime
import numpy as np
import math
from functions import *
from forecast import *
import pickle

confirmed_cases, total_infected = readRenameSumTotal('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths, total_dead = readRenameSumTotal('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_cases, total_recovered = readRenameSumTotal('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
total_removed = total_recovered + total_dead
countries = confirmed_cases["Country"].unique()
graph_dict = {"combi":"sike", "confirmed":confirmed_cases, "death":deaths, "recovered":recovered_cases}
dataframes = [confirmed_cases, deaths, recovered_cases]
covid = getDatewiseOverall(dataframes)
covid["Day Num"]=covid['Datetime']-covid['Datetime'][0]
covid["Day Num"]=covid["Day Num"].dt.days
confirmed_only = covid['Confirmed'].values.reshape(len(covid), 1)

# Load pickled models
with open(f'krr_country_model.pkl', 'rb') as f:
    krr = pickle.load(f)

with open(f'lr_country_model.pkl', 'rb') as f:
    lr = pickle.load(f)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CVD-19 Estimating"
server = app.server
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label='Overview', value='tab-1', children=[
            html.Br(),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='country_input', 
                        options=[{'label': i, "value": i} for i in countries],
                        value='Germany')
                ], className='six col'),
                html.Div([
                dcc.Dropdown(
                    id='graph_choice', 
                    options=[{'label': i, "value": i} for i in graph_dict],
                    multi=True,
                    value=['combi', 'confirmed']),
                dcc.RadioItems(
                    id='lin-log-radio-1',
                    options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                    value='Linear',
                    labelStyle={'display': 'inline-block', 'padding':'5px'})
                ], className='six col')
            ], className='row'),
            html.Div(id='output-graph')
        ]),
        dcc.Tab(label='SEIR', value='tab-2', children=[
            html.Div([
                html.H2("SEIR Model", style={'margin-left':'20px'}),
            ], className='row'),
            html.Br(),
            html.Div([
                html.Div([
                    html.H5("Infected Population:"),
                    dcc.Input(
                        id='infected_input',
                        type= "number", 
                        value=total_infected,
                        min=0,
                        max=7 * pow(10,9)
                    )
                ], className='three col'),
                html.Div([
                    html.H5("Removed Population:"),
                    dcc.Input(
                        id='removed_input',
                        type= "number", 
                        value=total_removed,
                        min=0,
                        max=7 * pow(10,9)
                    )
                ], className='three col'),
                html.Div([
                    html.H5("Meetings per Person:"),
                    dcc.Input(
                        id='meetings_input',
                        type= "number", 
                        value=25,
                        min=1,
                        max=500
                    )
                ], className='three col'),
                html.Div([
                    html.H5("Step size (time intervals):"),
                    dcc.Input(
                        id='timeperiod_input',
                        type= "number", 
                        value=1,
                        min=.5,
                        max=50)
                ], className='three col'),
            ], className='row', style={'margin-left':'50px'}),
            html.Br(),
            html.Br(),
            html.Div([
                html.Div([
                    dcc.Slider(
                        id='infectivity_range',
                        min=1,
                        max=20,
                        step=None,
                        marks=dict([(x, str(y)) for x, y in enumerate(range(21))]),
                        value=2),
                        html.H5(id='infectivity_range_out', className='text-center'),
                    dcc.RadioItems(
                        id='lin-log-radio-2',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block', 'padding':'5px', 'transform':'translate(15px, 5px)'})
                ], className='four col'),
                html.Div([
                    dcc.Slider(
                        id='latency_range',
                        min=1,
                        max=20,
                        step=None,
                        marks=dict([(x, str(y)) for x, y in enumerate(range(21))]),
                        value=5.1),
                        html.H5(id='latency_range_out', className='text-center'),
                ], className='four col'),
                html.Div([
                    dcc.Slider(
                        id='inf_period_range',
                        min=1,
                        max=20,
                        step=None,
                        marks=dict([(x, str(y)) for x, y in enumerate(range(21))]),
                        value=14),
                        html.H5(id='inf_period_range_out', className='text-center'),
                ], className='four col')
            ], className='row'),
            html.Div(id='anal_graph_1'),
                # dash_table.DataTable(
                #     id='seir_table',
                #     columns=[{"name": i, "id": i} for i in dfseir.columns],
                #     data=dfseir.to_dict('records'),
                # )
        ]),
        dcc.Tab(label='1-Day Forecasts', value='tab-3', children=[
                html.Div([
                    html.Div([
                    html.H2("Global Forecast:", style={
                        'position': 'absolute',
                        'margin': 'auto',
                        'width': '100%',
                        'padding': '20px',
                        'text-align': 'center'}),
                    html.Div(id='global_forecast', style={
                        'position': 'absolute',
                        'margin': 'auto',
                        'width': '100%',
                        'padding': '20px',
                        'text-align': 'center',
                        'transform': 'translateY(55px)'
                    }),
                    ]),
                ], className='row'),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Div([
                    html.Div([
                    html.H4("", style={
                        'position': 'absolute',
                        'right': '0px',
                        'padding': '10px',
                    })
                    ], className='six col'),
                    html.Div([
                    dcc.Dropdown(
                        id='country_input_forecast', 
                        options=[{'label': i, "value": i} for i in countries],
                        value='Germany',
                        # style={'position':'absolute'}
                        )
                    ], className='four col'),
                    html.Div([
                    ], className='two col'),
                ], className='row'),
                html.Div([
                    html.H2("Country Forecast:", style={
                        'position': 'absolute',
                        'margin': 'auto',
                        'width': '100%',
                        'padding': '20px',
                        'text-align': 'center',
                    }),
                ], className='row'),
                html.Br(),
                html.Br(),
                html.Br()
        ])
    ], className='mt-3'),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'), 
    [Input('tabs', 'value'),
    Input('country_input', 'value'),
    Input('graph_choice', 'value'),
    Input('lin-log-radio-1', 'value'),
    Input('lin-log-radio-2', 'value'),
    Input('meetings_input', 'value'),
    Input('removed_input', 'value'),
    Input('timeperiod_input', 'value'),
    Input('infected_input', 'value'),
    Input('infectivity_range', 'value'),
    Input('latency_range', 'value'),
    Input('inf_period_range', 'value'),
    Input('country_input_forecast', 'value')
    ]
    )
def graph_it(tab, inp, choice, linlog_1, linlog_2, meetings, removers, timeperiod, infected_inp, infectivity, latent_period, infectivity_period, country_input_forecast):
    date = dt.strftime(dt.now() - timedelta(1), '%d-%m-%Y')
    s_exp, e_exp, i_exp, r_exp, times_exp = seirExplicit(int(timeperiod), meetings, infectivity, removers, infected_inp, latent_period, infectivity_period)
    s_imp, e_imp, i_imp, r_imp, times_imp = seirImplicit(int(timeperiod), meetings, infectivity, removers, infected_inp, latent_period, infectivity_period)
    graphs = []

    combo_graph = html.Div([
                dcc.Graph(
                    id="test",
                    figure={
                        'data':[
                            dict(
                                y = countryGraphData("Worldwide", confirmed_cases),
                                x = countryGraphData("Worldwide", confirmed_cases).index,
                                name = "Infected",
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                marker = {
                                    'color':'red',
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = countryGraphData("Worldwide", deaths),
                                x = countryGraphData("Worldwide", deaths).index,
                                name = "Deaths",
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                marker = {
                                    'color':'grey',
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = countryGraphData("Worldwide", recovered_cases),
                                x = countryGraphData("Worldwide", recovered_cases).index,
                                name = "Recovered",
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                marker = {
                                    'color':'green',
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            )
                        ],
                        'layout': {
                            'annotations': [{
                                'x': 0.45, 'y': 1.05, 'xanchor': 'left', 'yanchor': 'bottom',
                                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                                'text': "Worldwide Overview of Cases"
                            }],
                            'yaxis' : {'title': 'Population', 'type': 'linear' if linlog_1 == 'Linear' else 'log'},
                            'xaxis' : {'title': "Date"},
                            'hovermode' : 'closest'
                        }
                    }
                )
            ])
    graphs.append(combo_graph)
    for x in choice:
        if "deat" in x:
            col = "grey"
        elif "recov" in x:
            col = "green"
        else:
            col = "red"
        if "combi" in x:
            graphs.append(html.Div([
                dcc.Graph(
                    id="test",
                    figure={
                        'data':[
                            dict(
                                y = countryGraphData(inp, confirmed_cases),
                                x = countryGraphData(inp, confirmed_cases).index,
                                name = "Infected",
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                marker = {
                                    'color':'red',
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = countryGraphData(inp, deaths),
                                x = countryGraphData(inp, deaths).index,
                                name = "Deaths",
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                marker = {
                                    'color':'grey',
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = countryGraphData(inp, recovered_cases),
                                x = countryGraphData(inp, recovered_cases).index,
                                name = "Recovered",
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                marker = {
                                    'color':'green',
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            )
                        ],
                        'layout': {
                            'annotations': [{
                                'x': 0.45, 'y': 1.05, 'xanchor': 'left', 'yanchor': 'bottom',
                                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                                'text': "Overview of Cases in " + inp
                            }],
                            'yaxis' : {'title': 'Population', 'type': 'linear' if linlog_1 == 'Linear' else 'log'},
                            'xaxis' : {'title': "Date"},
                            'hovermode' : 'closest'
                        }
                    }
                )
            ]))
        else:
            graphs.append(html.Div([
                    dcc.Graph(
                        id=x,
                        figure={
                            'data':[
                                dict(
                                    y = countryGraphData(inp, graph_dict[x]),
                                    x = countryGraphData(inp, graph_dict[x]).index,
                                    type='scatter',
                                    mode = 'lines+markers',
                                    opacity = 0.7,
                                    marker = {
                                        'color':col,
                                        'size': 7,
                                        'line': {'width': 0.5, 'color': 'black'}
                                        },
                                )
                            ],
                            'layout': {
                                'annotations': [{
                                    'x': 0.45, 'y': 1.05, 'xanchor': 'left', 'yanchor': 'bottom',
                                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                                    'text': x + ' - ' + inp
                                }],
                                'yaxis' : {'title': 'Population', 'type': 'linear' if linlog_1 == 'Linear' else 'log'},
                                'xaxis' : {'title': "Date"},
                                'hovermode' : 'closest'
                            }
                        }
                    )
                ]))
    
    seir_graph = html.Div([
                dcc.Graph(
                    id="test",
                    figure={
                        'data':[
                            dict(
                                y = s_exp,
                                x = times_exp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Susceptible",
                                marker = {
                                    'color':"blue",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = e_exp,
                                x = times_exp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Exposed",
                                marker = {
                                    'color':"yellow",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = i_exp,
                                x = times_exp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Infected",
                                marker = {
                                    'color':"red",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = r_exp,
                                x = times_exp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Removed",
                                marker = {
                                    'color':"grey",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            )
                        ],
                        'layout': {
                            'annotations': [{
                                'x': 0.45, 'y': 1.05, 'xanchor': 'left', 'yanchor': 'bottom',
                                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                                'text': "SEIR Model (Explicit)"
                            }],
                            'yaxis' : {'title': 'Population', 'type': 'linear' if linlog_2 == 'Linear' else 'log'},
                            'xaxis' : {'title': "Days"},
                            'hovermode' : 'closest'
                        }
                    }
                )
            ])

    seirImplicit_graph = html.Div([
                dcc.Graph(
                    id="test",
                    figure={
                        'data':[
                            dict(
                                y = s_imp,
                                x = times_imp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Susceptible",
                                marker = {
                                    'color':"blue",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = e_imp,
                                x = times_imp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Exposed",
                                marker = {
                                    'color':"yellow",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = i_imp,
                                x = times_imp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Infected",
                                marker = {
                                    'color':"red",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            ),
                            dict(
                                y = r_imp,
                                x = times_imp,
                                type='scatter',
                                mode = 'lines+markers',
                                opacity = 0.7,
                                name = "Removed",
                                marker = {
                                    'color':"grey",
                                    'size': 7,
                                    'line': {'width': 0.5, 'color': 'black'}
                                    },
                            )
                        ],
                        'layout': {
                            'annotations': [{
                                'x': 0.45, 'y': 1.05, 'xanchor': 'left', 'yanchor': 'bottom',
                                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                                'text': "SEIR Model (Implicit)"
                            }],
                            'yaxis' : {'title': 'Population', 'type': 'linear' if linlog_2 == 'Linear' else 'log'},
                            'xaxis' : {'title': "Days"},
                            'hovermode' : 'closest'
                        }
                    }
                )
            ])
    country_vals = getCountryPredVals(3)
    country_forecast_dic = {}
    for name in country_vals.columns:
        pred = krr.predict([country_vals[name].values])
        country_forecast_dic[name] = pred.item()
    
    if tab == 'tab-1':
        if len(graphs) > 0:
            return graphs
        else:
            return html.H4("Pick graph(s) to display")
    elif tab == 'tab-2':
        return seir_graph, seirImplicit_graph
    elif tab == 'tab-3':
        return html.H1(round(country_forecast_dic[country_input_forecast]), className="d-flex justify-content-center")

@app.callback(
    Output('global_forecast', 'children'),
    [Input('infectivity_range', 'value')])
def display_relayout_data(infection_prob):
    pred = round(lr.predict([confirmed_only[-13:].flatten()]).item())
    return html.H1(pred)

@app.callback(
    Output('infectivity_range_out', 'children'),
    [Input('infectivity_range', 'value')])
def display_relayout_data(infection_prob):
    return html.H5("Infection Prob: " + str(infection_prob))

@app.callback(
    Output('latency_range_out', 'children'),
    [Input('latency_range', 'value')])
def display_relayout_data(latency_range):
    return html.H5("Latent Period: " + str(latency_range))

@app.callback(
    Output('inf_period_range_out', 'children'),
    [Input('inf_period_range', 'value')])
def display_relayout_data(inf_period_range):
    return html.H5("Infectivity Period: " + str(inf_period_range))

if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=True)