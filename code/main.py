# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import os


mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th")
mlflow.set_experiment(experiment_name="st124047-a3")


def save(filename:str, obj:object):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename:str) -> object:
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


def load_mlflow(model_name, stage='Staging'):
    cache_path = os.path.join("models",stage)
    if(os.path.exists(cache_path) == False):
        os.makedirs(cache_path)
    
    # check if we cache the model
    path = os.path.join(cache_path,model_name)
    print()
    if(os.path.exists( path ) == False):
        # This will keep load the model again and again.
        model = mlflow.sklearn.load_model(model_uri=cache_path)
        save(filename=path, obj=model)

    model = load(path)
    return model

#Load Model
model_name = 'st124047-a3-model'
model_version = 'Staging'

model = load_mlflow(model_name, stage=model_version)

# load the scaling parameters for both model(same scaler is used for the features for both models)
scaler_path = "scaler/scaler.pkl"
loaded_scaler_params = pickle.load(open(scaler_path, 'rb'))

# Create scaler with the loaded parameters
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
def Predict_price_category(model_dropdown,year, km_driven, engine_size, fuel, transmission, submit):
    print(year, km_driven, engine_size, fuel, transmission)

    if year is None:
        age = 7.137924897668625 #initialized by mean of age
    else:
        age = abs(2020+1 - year)  #calculating age as the same way was done in training  ( age_of_car = [ max_year_of_data_set + 1 - year_of_car_model ] )
    if km_driven is None:
        km_driven = np.log(70029.87346502936) #initialized by mean of km_driven
    if engine_size is None:
        engine_size = 1463.855626715462 #initialized by mean of engine_size
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
    predPriceClass = model.predict(input_feature)[0]
    predictedText = f"Predicted Selling Price Category: {predPriceClass}"
    return predictedText
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
