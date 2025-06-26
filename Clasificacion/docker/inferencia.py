#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import json
from datetime import datetime, timedelta
import warnings
import random

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


def generar_datos_sinteticos(num_ejemplos=10):
    """Genera datos sintéticos realistas para Australia"""

    datos_sinteticos = []
    direcciones = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
        "NNE",
        "ENE",
        "ESE",
        "SSE",
        "SSW",
        "WSW",
        "WNW",
        "NNW",
    ]
    regiones = [
        "Nueva Gales del Sur Norte",
        "Queensland Central",
        "Territorio Norte",
        "Victoria",
    ]

    for i in range(num_ejemplos):
        # Generar fecha aleatoria en el futuro
        fecha_base = datetime(2025, 1, 1)
        dias_aleatorios = random.randint(0, 365)
        fecha = fecha_base + timedelta(days=dias_aleatorios)

        # Estacionalidad australiana (hemisferio sur)
        mes = fecha.month
        es_verano = mes in [12, 1, 2]
        es_invierno = mes in [6, 7, 8]

        # Temperaturas basadas en estación
        if es_verano:
            min_temp = round(random.uniform(15, 25), 1)
            max_temp = round(random.uniform(25, 40), 1)
        elif es_invierno:
            min_temp = round(random.uniform(2, 15), 1)
            max_temp = round(random.uniform(12, 25), 1)
        else:
            min_temp = round(random.uniform(8, 20), 1)
            max_temp = round(random.uniform(18, 32), 1)

        # Probabilidad de lluvia mayor en verano
        prob_lluvia = 0.3 if es_verano else 0.15
        lluvia_hoy = random.random() < prob_lluvia

        # Generar humedad y presión correlacionadas
        humidity_9am = random.randint(30, 95)
        humidity_3pm = random.randint(20, 90)
        pressure_9am = round(random.uniform(990, 1025), 1)
        pressure_3pm = round(random.uniform(990, 1025), 1)
        cloud_9am = random.randint(0, 8)
        cloud_3pm = random.randint(0, 8)

        # Generar RainTomorrow sintético basado en condiciones Mayor probabilidad si: llueve hoy, alta humedad, baja
        # presión, muchas nubes
        prob_lluvia_manana = 0.2  # base

        if lluvia_hoy:
            prob_lluvia_manana += 0.3
        if humidity_3pm > 70:
            prob_lluvia_manana += 0.2
        if pressure_3pm < 1010:
            prob_lluvia_manana += 0.15
        if cloud_3pm > 5:
            prob_lluvia_manana += 0.15
        if es_verano:
            prob_lluvia_manana += 0.1

        lluvia_manana_real = random.random() < prob_lluvia_manana

        datos = {
            "Date": fecha.strftime("%Y-%m-%d"),
            "Location": f"SyntheticCity{i+1}",
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": round(random.uniform(0, 15) if lluvia_hoy else 0, 1),
            "Evaporation": round(random.uniform(2, 12), 1),
            "Sunshine": round(random.uniform(0, 14), 1),
            "WindGustDir": random.choice(direcciones),
            "WindGustSpeed": random.randint(15, 80),
            "WindDir9am": random.choice(direcciones),
            "WindDir3pm": random.choice(direcciones),
            "WindSpeed9am": random.randint(0, 30),
            "WindSpeed3pm": random.randint(0, 35),
            "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm,
            "Cloud9am": cloud_9am,
            "Cloud3pm": cloud_3pm,
            "Temp9am": round(random.uniform(min_temp, (min_temp + max_temp) / 2), 1),
            "Temp3pm": round(random.uniform((min_temp + max_temp) / 2, max_temp), 1),
            "RainToday": "Yes" if lluvia_hoy else "No",
            "RainTomorrow": (
                "Yes" if lluvia_manana_real else "No"
            ),  # Valor "real" sintético
            "Region": random.choice(regiones),
        }

        datos_sinteticos.append(datos)

    return datos_sinteticos


def main():
    predictor = PredictorLluvia()

    print("🌦️  Predictor de Lluvia - Australia")
    print("=" * 50)

    resultados = []

    # 1. Datos del dataset original (10 ejemplos)
    print("\n📊 PREDICCIONES CON DATOS DEL DATASET:")
    print("-" * 40)

    df = pd.read_csv("weatherAUS.csv")
    for i in range(10):
        datos_aleatorios = df.sample(n=1).iloc[0].to_dict()

        # Guardar valor real
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
        resultado["tipo"] = "dataset"

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
                f"📍 Dataset {i+1}: {fecha} - Pred: {lluvia} | Real: {real} {acierto} ({prob:.1%})"
            )
            resultados.append(resultado)

    # 2. Datos sintéticos (10 ejemplos)
    print("\n🧪 PREDICCIONES CON DATOS SINTÉTICOS:")
    print("-" * 40)

    datos_sinteticos = generar_datos_sinteticos(10)

    for i, datos in enumerate(datos_sinteticos):
        # Guardar valor real sintético
        valor_real = datos.get("RainTomorrow", None)
        if "RainTomorrow" in datos:
            del datos["RainTomorrow"]

        # Remover Location ya que usamos Region
        if "Location" in datos:
            del datos["Location"]

        resultado = predictor.predecir(datos)
        resultado["tipo"] = "sintetico"
        resultado["datos_entrada"] = datos

        if "error" not in resultado:
            if valor_real is not None:
                valor_real_bool = valor_real == "Yes"
                resultado["valor_real"] = valor_real_bool
                resultado["acierto"] = resultado["lluvia_manana"] == valor_real_bool

            fecha = datos.get("Date", "N/A")
            lluvia = "🌧️ SÍ" if resultado["lluvia_manana"] else "☀️ NO"
            real = "🌧️ SÍ" if resultado.get("valor_real", False) else "☀️ NO"
            acierto = "✅" if resultado.get("acierto", False) else "❌"
            prob = resultado["probabilidad_lluvia"]

            print(
                f"🧪 Sintético {i+1}: {fecha} - Pred: {lluvia} | Real: {real} {acierto} ({prob:.1%})"
            )
            resultados.append(resultado)

    # Guardar resultados
    with open("prediccion_resultado.json", "w") as f:
        json.dump(resultados, f, indent=2, default=str)

    # Estadísticas
    dataset_results = [r for r in resultados if r.get("tipo") == "dataset"]
    synthetic_results = [r for r in resultados if r.get("tipo") == "sintetico"]

    aciertos_dataset = sum(1 for r in dataset_results if r.get("acierto", False))
    aciertos_sinteticos = sum(1 for r in synthetic_results if r.get("acierto", False))

    print(f"\n📊 RESUMEN:")
    print(
        f"✅ Dataset: {aciertos_dataset}/{len(dataset_results)} aciertos ({aciertos_dataset/len(dataset_results):.1%})"
    )
    print(
        f"🧪 Sintéticos: {aciertos_sinteticos}/{len(synthetic_results)} aciertos ({aciertos_sinteticos/len(synthetic_results):.1%})"
    )
    print(
        f"📁 {len(resultados)} predicciones totales guardadas en prediccion_resultado.json"
    )

    # Distribución de confianza en datos sintéticos
    confianzas = [r["confianza"] for r in synthetic_results]
    print(
        f"🎯 Confianza sintéticos: Alta={confianzas.count('Alta')}, Media={confianzas.count('Media')}, Baja={confianzas.count('Baja')}"
    )


if __name__ == "__main__":
    main()
