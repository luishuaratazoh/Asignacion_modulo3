import uvicorn
from fastapi import Depends, FastAPI, File, UploadFile,Response
from io import StringIO
import pandas as pd
from joblib import load

from sqlalchemy import create_engine, MetaData, Table,text
from sqlalchemy.orm import sessionmaker,session #esto sirve para importar las funciones necesarias de SQLAlchemy 

#para interactuar con la base de datos, como crear una conexión, definir la estructura de la base de 
# datos y gestionar las sesiones de la base de datos.

from datetime import datetime
import pytz #esto sirve para importar la biblioteca pytz, que se utiliza para manejar zonas horarias en Python, lo que permite trabajar con fechas y horas de manera precisa en diferentes regiones del mundo.

import os

# Configurar la base de datos
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:vDpBWXODHwyOZYezXqrsSVHdcurxeNhH@kodama.proxy.rlwy.net:46949/railway"

engine = create_engine(SQLALCHEMY_DATABASE_URL)#esto sirve para crear una conexión a la base de datos 
#SQLite utilizando SQLAlchemy, lo que permite interactuar con la base de datos en la aplicación FastAPI.

metadata = MetaData()#esto sirve para crear un objeto de metadatos en SQLAlchemy, que se utiliza 
#para definir la estructura de la base de datos, como las tablas y sus columnas, lo que facilita la gestión de la base de datos en la aplicación FastAPI.

session_local = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine) #esto sirve para configurar una sesión de SQLAlchemy que se utilizará para interactuar 
#con la base de datos, lo que permite realizar operaciones


app = FastAPI()

def get_db():
    db = session_local()
    try:
        yield db#esto sirve para definir una función que se utiliza para obtener una sesión de base de datos en la aplicación FastAPI, lo que permite interactuar con la base de datos de manera eficiente y segura, asegurando que las sesiones se cierren correctamente después de su uso.
    finally:
        db.close() #esto sirve para cerrar la sesión de base de datos después de su uso, lo que es importante para liberar recursos y evitar problemas de conexión en la aplicación FastAPI.

@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.get("/health", status_code=200)
def health_check(response: Response, db=Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "success", 
            "message": "Connected to the database successfully."
        }
    except Exception as e:
        response.status_code = 500
        return {
            "status": "error", 
            "message": f"Database connection failed: {str(e)}"
        }

#def health_check(db=Depends(get_db)):
#    return {"status": "OK"}#esto sirve para definir un endpoint de salud en la aplicación FastAPI, 
#lo que permite verificar si la aplicación está funcionando correctamente y responder con un mensaje de 
# #estado saludable.

@app.post("/predict")
async def predict_bancknote(file: UploadFile = File(...)):
    classifier = load("linear_regression.joblib")
    
    features_df = pd.read_csv('selected_features.csv')
    features = features_df['0'].to_list()

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df = df[features]

    predictions = classifier.predict(df)


    lima_tz = pytz.timezone('America/Lima')#esto sirve para configurar la zona horaria de Lima, lo que permite trabajar con fechas y horas de manera precisa en esa región específica.  
    now = datetime.now(lima_tz)#esto sirve para obtener la fecha y hora actual en la zona horaria de Lima, lo que es útil para registrar eventos o realizar operaciones basadas en el tiempo en esa región específica.
    
    predictions_df=pd.DataFrame({
        "file_name": file.filename,#esto sirve para crear un DataFrame de pandas que contiene el nombre del archivo cargado, las predicciones generadas por el modelo y la fecha y hora actual en la zona horaria de Lima, lo que permite almacenar y analizar los resultados de las predicciones de manera estructurada.
        "prediction": predictions,
        "created_at": now
    })


    predictions_df.to_sql(
        "predictions", 
        con=engine, 
        if_exists="append", 
        index=False
        )#esto sirve para guardar el DataFrame de predicciones en una tabla llamada "predictions" en la base de datos utilizando SQLAlchemy, lo que permite almacenar los resultados de las predicciones de manera persistente y consultable en la base de datos.

    return {
        "predictions": predictions.tolist(),
        "timestamp": now
    }
