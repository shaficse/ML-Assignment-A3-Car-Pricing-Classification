# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import os

# initialize mlflow url and  experiment for locally
# mlflow_url = "https://mlflow.cs.ait.ac.th"
# experiment_name="st124047-a3"

# Access environment variables
mlflow_url = os.environ.get("MLFLOW_URL")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
model_name = os.environ.get("MLFLOW_MODEL_NAME")
model_version = os.environ.get("MLFLOW_MODEL_VERSION")

mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment(experiment_name)

# #Load Model
# model_name = 'st124047-a3-model'
# model_version = 'Staging'

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")


# # load the scaling parameters for both model(same scaler is used for the features for both models)
scaler_path = "scaler/scaler.pkl"
loaded_scaler_params = pickle.load(open(scaler_path, 'rb'))

# # Create scaler with the loaded parameters
loaded_scaler = StandardScaler()
loaded_scaler.mean_ = loaded_scaler_params['mean']
loaded_scaler.scale_ = loaded_scaler_params['scale']


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    html.Div([
        dcc.Markdown('''
        - To predict the selling price of a car-   
           - **1. Please provide the required inputs(Year, km driven, Engine size, type of fuel and Transmission**(Blank input will be automatically filled using a imputation technique)
           - **2. Please Click "Predict" button. The app will predict the selling price category based on the provided inputs**
           - **3. Category of price has an order.** For example, 
                    - category 3 has higher selling price than category 2 
                    - category 2 has higher selling price than category 1
                    - category 1 has higher selling price than category 0 
        ''')
    ]),
    dbc.Row([
        html.Div([

            dbc.Label("Year of Car Made (eg. 2020)"),
            dbc.Input(id="year", type="number", placeholder="Enter the Car Model Year"),
       
            dbc.Label("Number of km Drived (eg. 450000 km)"),
            dbc.Input(id="km_driven", type="number", placeholder="Enter KM drived"),
       
            dbc.Label("Size of Engine(eg. 1248 CC)"),
            dbc.Input(id="engine_size", type="number", placeholder="Enter Engine size (in CC)"),
    
            dbc.Label("Type of Fuel"),
            dcc.Dropdown(['Petrol', 'Diesel'], id='fuel_dropdown'),
        
            dbc.Label("Type of Transmission"),
            dcc.Dropdown(['Manual', 'Automatic'], id='transmission_dropdown'),
            # dcc.Dropdown(['Manual', 'Automatic'], id='transmission_dropdown', disabled=False), 
            html.Br(),
            dbc.Button(id="submit", children="Predict", color="primary"),
        ],
        className="input_object"),

        html.Div(
            [
                html.Output(id="ouput_monitor", children="")
            ],
            className="output_object")
    ])

], fluid=True)


@callback(
    Output(component_id="ouput_monitor", component_property="children"),
    State(component_id="year", component_property="value"),
    State(component_id="km_driven", component_property="value"),
    State(component_id="engine_size", component_property="value"),
    State( component_id="fuel_dropdown", component_property="value"),
    State( component_id="transmission_dropdown", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)

def Predict_price_category(year, km_driven, engine_size, fuel, transmission, submit):
    print(year, km_driven, engine_size, fuel, transmission)

    if year is None:
        age = 7 #initialized by mean of age
    else:
        age = abs(2020+1 - year)  #calculating age as the same way was done in training  ( age_of_car = [ max_year_of_data_set + 1 - year_of_car_model ] )
    if km_driven is None:
        km_driven = np.log(70029) #initialized by mean of km_driven
    if engine_size is None:
        engine_size = 1463 #initialized by mean of engine_size
    if fuel is None or fuel == "Diesel":
        fuel = 0 #initialized by Diesel type if no input
    else:
        fuel = 1
    if transmission is None  or transmission == "Manual": 
        transmission = 1            #initialized by Manual type if no input
    else:
        transmission = 0

    #type casting of value in float64
    age = np.float64(age)
    km_driven = np.float64(np.log(km_driven))
    engine_size = np.float64(engine_size)
    fuel = np.float64(fuel)
    transmission = np.float64(transmission)
    # Make prediction using the model
    input_feature = np.array([[km_driven, age, engine_size,fuel,transmission]])
    # Transform the first 3 features
    input_feature[:, :3] = loaded_scaler.transform(input_feature[:, :3]) 


    # print(age, km_driven, engine_size, fuel, transmission, model_dropdown)
    intercept = np.ones((input_feature.shape[0], 1))
    input_feature    = np.concatenate((intercept, input_feature), axis=1)
    predPriceClass = loaded_model.predict(input_feature)[0]
    predictedText = f"Predicted Selling Price Category: {predPriceClass}"
    return predictedText
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
