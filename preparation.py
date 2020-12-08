# funciones y transformaciones basicas
import pandas as pd
import numpy as np
import joblib

def calculateAge(birthDate_str, splitter = "-"):
    from datetime import date 
    import datetime

    date_parts = birthDate_str.split(splitter)
    days_in_year = 365.2425
    birthDate = date(int(date_parts[0]),int(date_parts[1]), int(date_parts[2][:3]))
    age = int((date.today() - birthDate).days / days_in_year) 
    return age

def calculateAge_format(birthDate):
    from datetime import date

    days_in_year = 365.2425    
    age = int((date.today() - birthDate).days / days_in_year) 
    return age

def get_hist_freq(df, column, bins=10):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.hist(df[pd.notnull(df[column])][column], bins = bins)
    plt.show()


    porc = []
    unit = []
    integer = []

    for i in df[column].value_counts().index:
        porc.append(str(round((df[column].value_counts()[i]/df.shape[0])*100, 2)) + "%")
        unit.append(i)
        integer.append(str(df[column].value_counts()[i]))

    # agregando nulos
    porc.append(str(round(df[column].isna().sum()/df.shape[0]*100, 2)) + "%")
    integer.append(str(df[column].isna().sum()))
    unit.append("nulos")
    print()
    print("Total de registros: " + str(df.shape[0]))

    temp_df = pd.DataFrame()
    temp_df["Categoria"] = unit
    temp_df["registros"] = integer
    temp_df["Porcentaje"] = porc
    
    return temp_df



xls = pd.ExcelFile('Dataset Cancelaciones - XXXXXXXX.xlsx')
clientes = pd.read_excel(xls, 'Clientes')
siniestros = pd.read_excel(xls, 'Siniestros')
produccion = pd.read_excel(xls, "Producción")

# Sustituyendo espacios por nulos y valores repetidos

#clientes
clientes.replace(' ', np.NaN, inplace=True)
clientes.replace('[NULL]', np.NaN, inplace=True)
clientes["Estado_civil"].replace('CASADA', "CASADO", inplace=True)
clientes["Estado_civil"].replace('SOLTERA', "SOLTERO", inplace=True)
clientes["Estado_civil"].replace('DIVORCIADA', "DIVORCIADO", inplace=True)
clientes["Estado_civil"].replace('ACOMPAÑADA', "ACOMPAÑADO", inplace=True)
clientes["Estado_civil"].replace('VIUDA', "VIUDO", inplace=True)

clientes["Profesion_Ocupacion"].replace('AGENTE DE MIGRACION', "AGENTE", inplace=True)
clientes["Profesion_Ocupacion"].replace('AGENTE DE SEGUROS', "AGENTE", inplace=True)
clientes["Profesion_Ocupacion"].replace('AGENTE DE VENTAS', "AGENTE", inplace=True)
clientes["Profesion_Ocupacion"].replace('AGENTE VIAJERO', "AGENTE", inplace=True)
clientes["Profesion_Ocupacion"].replace('ABOGADO Y NOTARIO', "ABOGADO", inplace=True)
clientes["Profesion_Ocupacion"].replace('MEDICOS', "MEDICO", inplace=True)
clientes["Profesion_Ocupacion"].replace('EMPLEADOS', "EMPLEADO", inplace=True)
clientes["Profesion_Ocupacion"].replace('EMMPLEADA', "EMPLEADO", inplace=True)
clientes["Profesion_Ocupacion"].replace('ASISTENTE FINANCIERO', "ASISTENTE", inplace=True)
clientes["Profesion_Ocupacion"].replace('EMPRESARIOS', "EMPRESARIO", inplace=True)

# produccion
produccion.replace(' ', np.NaN, inplace=True)
produccion.replace('[NULL]', np.NaN, inplace=True)
produccion["Estado_de_la_poliza"].replace('NO RENOVADA', "NO_CANCELADA", inplace=True)
produccion["Estado_de_la_poliza"].replace('ACTIVA', "NO_CANCELADA", inplace=True)
produccion['Estado_de_la_poliza'].replace("CANCELADA", 1, inplace= True)
produccion['Estado_de_la_poliza'].replace("NO_CANCELADA", 0, inplace= True)
# Siniestros
hora = [i.hour for i in siniestros["fec_hora_reclamo"]]
hora_temp = []
for i in hora:
    if i < 12:
        hora_temp.append("morning")
        
    elif i == 12:
        hora_temp.append("mid-day")
        
    elif 12 < i <= 16:
        hora_temp.append("afternoon")
        
    elif i > 16:
        hora_temp.append("night")
        
siniestros["momento_reclamo"] = hora_temp

# Creando nuevas variables

edad_clientes = [calculateAge(str(i)) if str(i) != "nan" else np.nan for i in list(clientes["Fecha_de_nacimiento"])]
clientes["edad_cliente"] = edad_clientes
clientes["edad_cliente"].fillna(clientes["edad_cliente"].mean(), inplace=True)

siniestros["hora_reclamo"] = [i.hour for i in list(siniestros["fec_hora_reclamo"])]

produccion["vigencia"] = produccion["Fecha_vigencia_hasta"] - produccion["Fecha_vigencia_desde"]
produccion["vigencia"] = [i.days/365 for i in produccion["vigencia"]]

# filling

clientes["Lugar_de_TRabajo"].fillna("No_trabaja", inplace=True)
clientes["Profesion_Ocupacion"].fillna("No_profesion", inplace=True)
clientes["Estado_civil"].fillna("Desconocido", inplace=True)
clientes["edad_cliente"].fillna(clientes["edad_cliente"].mean(), inplace=True)
clientes["Nacionalidad"].fillna(clientes["Nacionalidad"].mode, inplace=True)

df_sp = pd.merge(siniestros, produccion,left_on="id", right_on="id")
df_spc = pd.merge(df_sp, clientes, right_on="ID_cliente", left_on="ID_cliente", how="left")

df_spc.to_pickle("data/data_joined.pkl")

X = df_spc[["causa", "taller", "momento_reclamo", "Ejecutivo", "Importe_de_prima", "vigencia", "edad_cliente"]]
y = df_spc["Estado_de_la_poliza"]
cat_vars = ["causa", "taller", "momento_reclamo", "Ejecutivo"]

X = pd.concat([pd.get_dummies(X[cat_vars]), X[['Importe_de_prima', "edad_cliente"]]], axis=1)

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

x = X[['Importe_de_prima', "edad_cliente"]].values #returns a numpy array
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
df_n = pd.DataFrame(x_scaled)
X[['Importe_de_prima', "edad_cliente"]] = df_n

# Separamos datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)


variables_imp = ['Importe_de_prima', 'Ejecutivo_SANCHEZ PEÑA, JOSE ROBERTO', 'edad_cliente', 'Ejecutivo_QUALITY ASSURANCE CORREDORES DE SEGUROS, S.A. DE C.V.', 'Ejecutivo_ARTOLA GOMEZ, NOEL ALFREDO', 'momento_reclamo_night']

model_2 = RandomForestClassifier(n_estimators=5, bootstrap = True, max_features= 6, random_state=100)

model_2.fit(X_train[variables_imp], y_train)
y_pred = model_2.predict(X_test[variables_imp])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model_2.score(X_test[variables_imp], y_test)))


# metricas
from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(y_test, y_pred)
tn = cfm[0][0]
tp = cfm[1][1]
fn = cfm[1][0]
fp = cfm[0][1]

recall = tp/(tp + fn)
precision = tp/(tp + fp)

joblib.dump(model_2, "rf_clf_imp.pkl")

