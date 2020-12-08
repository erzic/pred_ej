import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
#import dash_enterprise_auth as auth
import dash_auth
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

VALID_USERNAME_PASSWORD_PAIRS = {
    'XXXXXX': 'XXXXXX'
}

# **************** Obteniendo datos del modelo ****************************
import pandas as pd
import joblib
try:
    del X_test, y_pred
except:
    pass


df = pd.read_pickle("data/data_joined.pkl")

X = df[["causa", "taller", "momento_reclamo", "Ejecutivo", "Importe_de_prima", "vigencia", "edad_cliente"]]
y = df["Estado_de_la_poliza"]
cat_vars = ["causa", "taller", "momento_reclamo", "Ejecutivo"]


vars_finales = ['Importe_de_prima', 'Ejecutivo_SANCHEZ PEÑA, JOSE ROBERTO', 'edad_cliente', 'Ejecutivo_QUALITY ASSURANCE CORREDORES DE SEGUROS, S.A. DE C.V.', 'Ejecutivo_ARTOLA GOMEZ, NOEL ALFREDO', 'momento_reclamo_night']
X = pd.concat([pd.get_dummies(X[cat_vars]), X[['Importe_de_prima', "edad_cliente"]]], axis=1)[['Importe_de_prima', 'Ejecutivo_SANCHEZ PEÑA, JOSE ROBERTO', 'edad_cliente', 'Ejecutivo_QUALITY ASSURANCE CORREDORES DE SEGUROS, S.A. DE C.V.', 'Ejecutivo_ARTOLA GOMEZ, NOEL ALFREDO', 'momento_reclamo_night']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

model_2 = joblib.load("rf_clf_imps.pkl")
y_pred = model_2.predict(X_test)

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(y_test, y_pred)
tn = cfm[0][0]
tp = cfm[1][1]
fn = cfm[1][0]
fp = cfm[0][1]


accuracy = (tp + tn)/(tp + tn + fp + fn)
recall = tp/(tp + fn)
precision = tp/(tp + fp)
f1 = 2*((precision*recall)/(precision+recall))



# ***************** Generado graficos ************************************
plot_estado = go.Histogram(x=list(df["Estado_de_la_poliza"]))

plot_edad = go.Histogram(x=list(df["edad_cliente"]))


# ***************** starting the app *************************************
app = dash.Dash(__name__) # ponemos el __name__ para conectarse con el folder de assets



app.layout= html.Div([
    #banner
    html.Div([
        html.H2("Modelo Predictivo - Cancelaciones ASESUISAS"),
        html.Img(src = "/assets/logo_as.png")
    ], className="banner"),

    # graph (two columns graph)
    html.Div([
        html.Div([# graph and input
            
            html.Div([ # input
                html.H1(["Visualización de los datos"], className = "text_header"),
                dcc.Input(id="variable_input", value = "edad_cliente", type="text"),
                html.Button(id ="submit-button", n_clicks=0, children="Submit variable")
            ]),

            html.Div([ #graph 1 (update)
                dcc.Graph(
                    id="graph_porc_estados",
                )
            ], className="two_columns"),
        ]),
         html.Div([ #Form
            html.H1(["Modelo predictivo"], className = "text_header"), 
            html.Form([
                html.Div([ # Primera Columna
                    html.Label([
                        "Importe de prima: "
                    ], className="header_text"),
                ]),
                html.Br(),
                html.Div([ # item 3
                    html.Label([
                        "Edad Cliente: "
                    ], className="header_text"),
                ]),
                html.Div([ # item 6
                    html.Label([
                        "Reclamo Noche: "
                    ], className="header_text"),
                ]),
                html.Br(),
                html.Div([ # item 2
                    html.Label([
                        "Ejecutivo 1: "
                    ], className="header_text"),
                ]),
                html.Br(),
                html.Div([ # item 4
                    html.Label([
                        "Ejecutivo 2: "
                    ], className="header_text"),
                ]),
                html.Br(),
                html.Div([ # item 5
                    html.Label([
                        "Ejecutivo 3: "
                    ], className="header_text"),
                ]),
                html.Br(),
                # submit button
                html.Br(),

                html.Div([
                    html.Label(["Submit_data"]),
                    html.Button(id ="submit-data", n_clicks=0, children="Submit")
                ], className="header_text"),
            ], className="two_columns_25"),


            html.Form([
                html.Div([
                    dcc.Input(
                        type="number", id="input_importe_de_prima", placeholder="Numero entero o decimal", value = 384.67
                        ),
                ]),
                html.Br(),
                html.Div([ 
                    dcc.Input(
                        type="number", id="input_ejecutivo_1", placeholder="Escribir 1 o 0", value = 0
                        ),
                ]),
                html.Br(),
                html.Div([
                    dcc.Input(
                        type="number", id="input_edad", placeholder="Numero entero o decimal", value = 35
                        ),
                ]),
                html.Br(),
                html.Div([
                    dcc.Input(
                        type="number", id="input_ejecutivo_2", placeholder="Escribir 1 o 0", value = 0
                        ),
                ]),
                html.Br(),
                html.Div([
                    dcc.Input(
                        type="number", id="input_ejecutivo_3", placeholder="Escribir 1 o 0", value = 0
                        ),
                ]),
                html.Br(),
                html.Div([
                    dcc.Input(
                        type="number", id="input_noche", placeholder="Escribir 1 o 0", value = 0
                        ),
                ]),
            ], className="two_columns_25"),

        html.Div([ # Prediccion
            html.P(["Predicción: "], className="header_text"),
            html.Div(id="predicted", className="data_text"),

        ], className="two_columns_25")           
        ], className="two_columns"),
    ], className="row"),

    html.Div([    
        html.Div([
            dash_table.DataTable(
                id="main_dataframe",
                columns = [{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
            )
        ], style={'overflow': 'scroll', "max-height": "400px"}, className="two_columns"),

        # Mostramos metricas
        html.Div([ # Cards Area, make a 2x2 grid
            html.H1(["Métricas"], className = "text_header"),
            html.Div([ #first row (cards)
                html.Div([
                    html.Div([
                        html.Div([html.H3(["Accuracy"], className="header_text")]),
                        html.Div([html.P([str(round(accuracy*100, 2)) + "%"], className="data_text")])
                    ], className = "two_columns"), #first column, first row
                    html.Div([
                        html.Div([html.H3(["Recall"], className="header_text")]),
                        html.Div([html.P([str(round(recall*100, 2)) + "%"], className="data_text")])
                    ], className = "two_columns")       
                ], className="row"),            
            ], className ="row"),

            html.Div([ #Second row (cards)
                html.Div([
                    html.Div([
                        html.Div([html.H3(["Precision"], className="header_text")]),
                        html.Div([html.P([str(round(precision*100, 2)) + "%"], className="data_text")])
                    ], className = "two_columns"), #first column, first row
                    html.Div([
                        html.Div([html.H3(["F1 Score"], className="header_text")]),
                        html.Div([html.P([str(round(f1*100,2)) + "%"], className="data_text")])
                    ], className = "two_columns")       
                ], className="row"),            
            ], className ="row")
        ], className="two_columns")
    ], className = "row"),

])

@app.callback(
    Output("graph_porc_estados", "figure"),
    [Input("submit-button", "n_clicks")],
    [State("variable_input", "value")]
)
def update_fig(n_clicks, input_value):
    df = pd.read_pickle("data/data_joined.pkl")

    data = list(df[input_value])
    plot_estado = go.Histogram(x=data)

    layout = { "Title": input_value}

    return {
        "data":[plot_estado],
        "layout":layout
    }

@app.callback(
    Output("predicted", 'children'),
    #[Input("submit-data", "n_clicks")],
    [Input("input_importe_de_prima", "value"),
    Input("input_ejecutivo_1", "value"),
    Input("input_edad", "value"),
    Input("input_ejecutivo_2", "value"),
    Input("input_ejecutivo_3", "value"),
    Input("input_noche", "value")]
    )
def update_customer(importe_de_prima, ejecutivo_1, edad, ejecutivo_2, ejecutivo_3, noche):
    temp_df = pd.DataFrame(columns=list(X.columns))
    temp_df.loc[len(df)] = [importe_de_prima, ejecutivo_1, edad, ejecutivo_2, ejecutivo_3, noche]
    return model_2.predict(temp_df)[0]

if __name__ == "__main__":
    app.run_server(debug=True)