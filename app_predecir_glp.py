# Importando las librerias necesarias
import streamlit as st

import pandas as pd
import numpy as np

import seaborn as sns # for visualizations (correlacion de variables)
import plotly.graph_objects as go # for visualizations
import plotly.express as px # for visualizations

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR # for building support vector regression model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ================ FUNCTIONS ================
def graficaScatter2Dinteractiva(nombreVariable1, nombreVariable2):
    # Create a scatter plot
    fig = px.scatter(df_glp, x=df_glp[nombreVariable1], y=df_glp[nombreVariable2], 
                    opacity=0.8, color_discrete_sequence=['black'])

    # Change chart background color
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                    showline=True, linewidth=1, linecolor='black')

    # Set figure title
    fig.update_layout(title_text="Scatter Plot")

    # Update marker size
    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        autosize=False,
        width=400,
        height=400
    )

    return fig


def graficaScatter3DInteractiva(variableX, variableY, variableZ):
    # Create a 3D scatter plot
    fig = px.scatter_3d(df_glp, x=df_glp[variableX], y=df_glp[variableY], z=df_glp[variableZ], 
                    opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title
    fig.update_layout(title_text="Scatter 3D Plot",
                    scene = dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='lightgrey'),
                                yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='lightgrey'
                                            ),
                                zaxis=dict(backgroundcolor='white',
                                            color='black', 
                                            gridcolor='lightgrey')))

    # Update marker size
    fig.update_traces(marker=dict(size=3))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=500
    )

    return fig


def graficaScatter3DInteractivaYPlanoPrediccion(nombreModelo):
    # ------------------------ Prepare a number of points to use for prediction (GENERAL - usado para todos los posteriores graficos)--------------------------
    # Increments between points in a meshgrid
    mesh_size = 0.01

    # Identify min and max values for input variables
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # Return evenly spaced values based on a range between min and max
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)

    # Create a meshgrid
    xx, yy = np.meshgrid(xrange, yrange)


    # ------------------------------ Use the model to predict the output ---------------------------------
    # Run model
    if nombreModelo == 'regresion lineal':
        pred_model = LR_model.predict(np.c_[xx.ravel(), yy.ravel()])
    elif nombreModelo == 'regresion polinomial':
        pred_model = SVR_model.predict(np.c_[xx.ravel(), yy.ravel()])

    pred_model = pred_model.reshape(xx.shape)


    # ------------------------------------------- Plot --------------------------------------------------
    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(df_glp, x=df_glp['Precio Mont Belvieu'], y=df_glp['Precio Dolar'], z=df_glp['Precio GLP'], 
                    opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(title_text="Gráfico 3D de dispersión con superficie de predicción",
                    scene = dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='lightgrey'),
                                yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='lightgrey'
                                            ),
                                zaxis=dict(backgroundcolor='white',
                                            color='black', 
                                            gridcolor='lightgrey')))
    # Update marker size
    fig.update_traces(marker=dict(size=3))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_model, name='LR'))

    fig.update_layout(
        autosize=False,
        width=400,
        height=400
    )

    return fig



def graficaScatter3DInteractivaYPlanoPrediccionYPrediccion(nombreModelo, value_mont_belvieu, value_precio_dolar, value_precio_glp_predict):
    # ------------------------ Prepare a number of points to use for prediction (GENERAL - usado para todos los posteriores graficos)--------------------------
    # Increments between points in a meshgrid
    mesh_size = 0.01

    # Identify min and max values for input variables
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # Return evenly spaced values based on a range between min and max
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)

    # Create a meshgrid
    xx, yy = np.meshgrid(xrange, yrange)


    # ------------------------------ Use the model to predict the output ---------------------------------
    # Run model
    if nombreModelo == 'regresion lineal':
        pred_model = LR_model.predict(np.c_[xx.ravel(), yy.ravel()])
    elif nombreModelo == 'regresion polinomial':
        pred_model = SVR_model.predict(np.c_[xx.ravel(), yy.ravel()])

    pred_model = pred_model.reshape(xx.shape)


    # ------------------------------------------- Plot --------------------------------------------------
    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(df_glp, x=df_glp['Precio Mont Belvieu'], y=df_glp['Precio Dolar'], z=df_glp['Precio GLP'], 
                    opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(title_text="Gráfico 3D de dispersión con superficie de predicción",
                    scene = dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='lightgrey'),
                                yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='lightgrey'
                                            ),
                                zaxis=dict(backgroundcolor='white',
                                            color='black', 
                                            gridcolor='lightgrey')))
    # Update marker size
    fig.update_traces(marker=dict(size=3))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_model, name='LR'))
    
    # Add prediction point
    fig.add_trace(go.Scatter3d(x=[value_mont_belvieu], y=[value_precio_dolar], z=[value_precio_glp_predict], mode='markers',
    marker=dict(
        size=7,
        color='#00f'                # set color to an array/list of desired values
    )))

    fig.update_layout(
        autosize=False,
        width=400,
        height=400
    )

    return fig
# ================ END FUNCTIONS ================ 


st.set_page_config(layout="wide")

# Sidebar
st.sidebar.image('images/logo_unap.png', width=200);
st.sidebar.markdown('**Materia:** Metodos Numericos')
st.sidebar.markdown('**Tema:** Ajuste de Curvas')
st.sidebar.markdown('**Docente:**')
st.sidebar.markdown('👩‍🏫 Zenaida Condori Apaza')
st.sidebar.markdown('**Integrantes:**')
st.sidebar.markdown('🌟 Deyvis Mamani Lacuta')
st.sidebar.markdown('🌟 Edith Irene Ticona Laura')
st.sidebar.markdown('🌟 Adiv Brander Cari Quispe')

# Main content
st.title('Modelos de Regresion para predecir el precio del GLP')
left_column, right_column = st.columns([3, 1])
left_column.header('IMPORTANCIA')
left_column.markdown('El Gas Licuado de Petróleo (GLP), es considerado en el Perú, como el energético más importante en la canasta de consumo de combustibles. La importancia del uso del GLP radica en que al ser combustible cuya combustión es completa no contamina el ambiente, es utilizado principalmente en cocinas y hornos también es utilizado, pero en menos proporción, para la iluminación, para las termas y últimamente se está utilizando como combustible para los vehículos motorizados, además al ser usado en los hogares como fuente de energía se ayuda a preservar el ambiente ya que se deja de talar árboles para la producción de leña y carbón y a la vez se deja de lado el consumo del petróleo y el kerosene los cuales contaminan el ambiente. \nLa situación que se observa en el mercado de GLP es preocupante. En los últimos meses el precio del gas licuado en el Perú se ha incrementado en mayor ritmo que el precio de importación y ello afecta considerablemente en la economía de las familias');
right_column.image('images/balon_gas.png', width = 170)
st.markdown('---')


st.header('CONJUNTO DE DATOS')
df_glp = pd.read_csv('GLP_Data.csv')
st.dataframe(df_glp)
st.markdown('---')


st.header('VISUALIZACION DE LAS VARIABLES DE ENTRADA Y SALIDA')
left_column, right_column = st.columns(2)
left_column.subheader('Visualizar Precio Mont Belvieu vs Precio GLP')
left_column.plotly_chart(graficaScatter2Dinteractiva('Precio Mont Belvieu', 'Precio GLP'), use_container_width=True)
right_column.subheader('Visualizar Precio Dolar vs Precio GLP')
right_column.plotly_chart(graficaScatter2Dinteractiva('Precio Dolar', 'Precio GLP'), use_container_width=True)

st.subheader('Visualizar el conjunto de datos')
st.plotly_chart(graficaScatter3DInteractiva('Precio Mont Belvieu', 'Precio Dolar', 'Precio GLP'))
st.markdown('---')



st.header('Generacion de los modelos de regresion (80% data)')
# Dividiendo nuestro conjunto de datos para entrenamiento y prueba
X = df_glp.iloc[:, 1:3].to_numpy()
y = df_glp.iloc[:, 4].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# >>> Generando el modelo de Regresion Lineal <<<
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)

y_pred_LR = LR_model.predict(X_test)
MSE_LR = mean_squared_error(y_test, y_pred_LR)


# >>> Generando el modelo de Regresion Polinomial <<<
SVR_model = SVR(kernel='rbf', C=700, epsilon=0.001)
SVR_model.fit(X_train, y_train)

y_pred_SVR = SVR_model.predict(X_test)
MSE_SVR = mean_squared_error(y_test, y_pred_SVR)


left_column, right_column = st.columns(2)
left_column.subheader('Modelo de Regresion Lineal')
left_column.plotly_chart(graficaScatter3DInteractivaYPlanoPrediccion('regresion lineal'), use_container_width=True)
left_column.markdown('> **Error cuadratico medio (20% data): **'+str(round(MSE_LR, 5)))

right_column.subheader('Modelo de Regresion Polinomial')
right_column.plotly_chart(graficaScatter3DInteractivaYPlanoPrediccion('regresion polinomial'), use_container_width=True)
right_column.markdown('> **Error cuadratico medio (20% data): **'+str(round(MSE_SVR, 5)))
st.markdown('---') 


st.header('Predecir el precio del GLP')
left_column, right_column = st.columns(2)
precio_mont_belvieu = left_column.number_input('Ingrese el precio del Mont Belvieu', min_value=0.000, step=0.001, format="%.3f")
precio_dolar = left_column.number_input('Ingrese el precio del Dolar', min_value=0.000, step=0.001, format="%.3f")

if left_column.button('Predecir el precio del GLP'):
    precio_glp_LR =  float(LR_model.predict([[precio_mont_belvieu, precio_dolar]]))
    precio_glp_SVR = float(SVR_model.predict([[precio_mont_belvieu, precio_dolar]]))
    left_column, right_column = st.columns(2)
    left_column.subheader('Modelo de Regresion Lineal')
    left_column.markdown('**Precio GLP: **' + str(round(precio_glp_LR, 5)))
    left_column.plotly_chart(graficaScatter3DInteractivaYPlanoPrediccionYPrediccion('regresion lineal', precio_mont_belvieu, precio_dolar, precio_glp_LR))

    right_column.subheader('Modelo de Regresion Polinomial')
    right_column.markdown('**Precio GLP: **' + str(round(precio_glp_SVR, 5)))
    right_column.plotly_chart(graficaScatter3DInteractivaYPlanoPrediccionYPrediccion('regresion polinomial', precio_mont_belvieu, precio_dolar, precio_glp_SVR))
