#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class PredictorLluvia:
    def __init__(self):
        try:
            self.modelo = tf.keras.models.load_model("mejor_modelo_nn.h5")
            self.pipeline = joblib.load("preprocess.pkl")
            print("✅ Modelo y pipeline cargados exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    def preparar_datos(self, datos_input):
        if isinstance(datos_input, dict):
            df = pd.DataFrame([datos_input])
        else:
            df = datos_input.copy()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.month

        cardinal_to_angle = {
            "N": 0,
            "NNE": 22.5,
            "NE": 45,
            "ENE": 67.5,
            "E": 90,
            "ESE": 112.5,
            "SE": 135,
            "SSE": 157.5,
            "S": 180,
            "SSW": 202.5,
            "SW": 225,
            "WSW": 247.5,
            "W": 270,
            "WNW": 292.5,
            "NW": 315,
            "NNW": 337.5,
        }

        for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
            if col in df.columns:
                df[col] = df[col].map(cardinal_to_angle)

        if "WindGustDir" in df.columns:
            df["WindGustDir_Sen"] = np.sin(np.radians(df["WindGustDir"]))
            df["WindGustDir_Cos"] = np.cos(np.radians(df["WindGustDir"]))
            df = df.drop("WindGustDir", axis=1)

        if "WindDir9am" in df.columns:
            df["WindDir9am_Sen"] = np.sin(np.radians(df["WindDir9am"]))
            df["WindDir9am_Cos"] = np.cos(np.radians(df["WindDir9am"]))
            df = df.drop("WindDir9am", axis=1)

        if "WindDir3pm" in df.columns:
            df["WindDir3pm_Sen"] = np.sin(np.radians(df["WindDir3pm"]))
            df["WindDir3pm_Cos"] = np.cos(np.radians(df["WindDir3pm"]))
            df = df.drop("WindDir3pm", axis=1)

        if "Region" in df.columns:
            df = pd.get_dummies(df, columns=["Region"], drop_first=True)

        if "RainToday" in df.columns:
            df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

        expected_columns = [
            "Date",
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "Evaporation",
            "Sunshine",
            "WindGustSpeed",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp9am",
            "Temp3pm",
            "RainToday",
            "WindGustDir_Sen",
            "WindGustDir_Cos",
            "WindDir9am_Sen",
            "WindDir9am_Cos",
            "WindDir3pm_Sen",
            "WindDir3pm_Cos",
            "Region_Nueva Gales del Sur Norte",
            "Region_Queensland Central",
            "Region_Territorio Norte",
            "Region_Victoria",
        ]

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        datos_preparados = self.pipeline.transform(df)
        return datos_preparados

    def predecir(self, datos_input):
        try:
            datos_preparados = self.preparar_datos(datos_input)
            probabilidad = self.modelo.predict(datos_preparados, verbose=0)[0][0]
            prediccion = 1 if probabilidad > 0.5 else 0

            resultado = {
                "fecha_prediccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "lluvia_manana": bool(prediccion),
                "probabilidad_lluvia": float(probabilidad),
                "probabilidad_no_lluvia": float(1 - probabilidad),
                "confianza": (
                    "Alta"
                    if abs(probabilidad - 0.5) > 0.3
                    else "Media" if abs(probabilidad - 0.5) > 0.1 else "Baja"
                ),
            }
            return resultado
        except Exception as e:
            return {"error": f"Error en predicción: {e}"}


def main():
    predictor = PredictorLluvia()

    df = pd.read_csv("weatherAUS.csv")
    print("🌦️  Predictor de Lluvia - Australia")
    print("=" * 50)

    resultados = []
    num_ejemplos = 15

    for i in range(num_ejemplos):
        datos_aleatorios = df.sample(n=1).iloc[0].to_dict()

        # Guardar valor real antes de eliminarlo
        valor_real = datos_aleatorios.get("RainTomorrow", None)
        if "RainTomorrow" in datos_aleatorios:
            del datos_aleatorios["RainTomorrow"]

        if "Location" in datos_aleatorios:
            location = datos_aleatorios["Location"]
            region_mapping = {
                "Sydney": "Nueva Gales del Sur Norte",
                "Melbourne": "Victoria",
                "Brisbane": "Queensland Central",
                "Darwin": "Territorio Norte",
                "Adelaide": "Territorio Norte",
                "Perth": "Territorio Norte",
                "Hobart": "Victoria",
                "Canberra": "Nueva Gales del Sur Norte",
            }
            datos_aleatorios["Region"] = region_mapping.get(
                location, "Nueva Gales del Sur Norte"
            )
            del datos_aleatorios["Location"]

        resultado = predictor.predecir(datos_aleatorios)

        if "error" not in resultado:
            if valor_real is not None:
                valor_real_bool = valor_real == "Yes"
                resultado["valor_real"] = valor_real_bool
                resultado["acierto"] = resultado["lluvia_manana"] == valor_real_bool

            fecha = datos_aleatorios.get("Date", "N/A")
            lluvia = "🌧️ SÍ" if resultado["lluvia_manana"] else "☀️ NO"
            real = "🌧️ SÍ" if resultado.get("valor_real", False) else "☀️ NO"
            acierto = "✅" if resultado.get("acierto", False) else "❌"
            prob = resultado["probabilidad_lluvia"]

            print(
                f"📍 Ejemplo {i+1}: {fecha} - Pred: {lluvia} | Real: {real} {acierto} ({prob:.1%})"
            )
            resultados.append(resultado)

    with open("prediccion_resultado.json", "w") as f:
        json.dump(resultados, f, indent=2)

    aciertos = sum(1 for r in resultados if r.get("acierto", False))
    print(
        f"\n✅ {len(resultados)} predicciones guardadas | Accuracy: {aciertos}/{len(resultados)} ({aciertos/len(resultados):.1%})"
    )


if __name__ == "__main__":
    main()
