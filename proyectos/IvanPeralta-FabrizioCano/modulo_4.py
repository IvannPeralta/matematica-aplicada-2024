import numpy as np
import skfuzzy as fuzzy
import pandas as pd
import time

tiempo_total_promedio=0

#definimos los rangos del universo
x_pos = np.arange(0, 1, 0.1)
x_neg = np.arange(0, 1, 0.1)
x_out = np.arange(0, 10, 1)

# Generar las funciones de membresía difusas
pos_low = fuzzy.trimf(x_pos, [0, 0, 0.5])
pos_mid = fuzzy.trimf(x_pos, [0, 0.5, 1])
pos_high = fuzzy.trimf(x_pos, [0.5, 1, 1])

neg_low = fuzzy.trimf(x_neg, [0, 0, 0.5])
neg_mid = fuzzy.trimf(x_neg, [0, 0.5, 1])
neg_high = fuzzy.trimf(x_neg, [0.5, 1, 1])

out_neg = fuzzy.trimf(x_out, [0, 0, 5])  
out_neu = fuzzy.trimf(x_out, [0, 5, 10])
out_pos = fuzzy.trimf(x_out, [5, 10, 10])

#dataset que contiene los puntajes positivos y negativos
dataset = pd.read_csv('test_data_con_puntajes.csv')

puntaje_defuzzificado = []
categoria_defuzzificada = []
tiempos=[]

for index, tweet in dataset.iterrows():
    start = time.perf_counter()

    tweetPos = tweet['tweetPos']
    tweetNeg = tweet['tweetNeg']

    if tweetPos == 1: tweetPos = 0.90
    if tweetNeg == 1: tweetNeg = 0.90

    # Calcular el nivel de pertenencia de cada valor de entrada
    pos_level_low = fuzzy.interp_membership(x_pos, pos_low, tweetPos)
    pos_level_mid = fuzzy.interp_membership(x_pos, pos_mid, tweetPos)
    pos_level_high = fuzzy.interp_membership(x_pos, pos_high, tweetPos)

    neg_level_low = fuzzy.interp_membership(x_neg, neg_low, tweetNeg)
    neg_level_mid = fuzzy.interp_membership(x_neg, neg_mid, tweetNeg)
    neg_level_high = fuzzy.interp_membership(x_neg, neg_high, tweetNeg)

    # Definir las reglas
    reglas={}
    reglas['WR1'] = np.fmin(pos_level_low, neg_level_low)
    reglas['WR2'] = np.fmin(pos_level_mid, neg_level_low)
    reglas['WR3'] = np.fmin(pos_level_high, neg_level_low)
    reglas['WR4'] = np.fmin(pos_level_low, neg_level_mid)
    reglas['WR5'] = np.fmin(pos_level_mid, neg_level_mid)
    reglas['WR6'] = np.fmin(pos_level_high, neg_level_mid)
    reglas['WR7'] = np.fmin(pos_level_low, neg_level_high)
    reglas['WR8'] = np.fmin(pos_level_mid, neg_level_high)
    reglas['WR9'] = np.fmin(pos_level_high, neg_level_high)

    # Agregar las reglas de output
    Wneg = np.fmax(np.fmax(reglas['WR4'], reglas['WR7']), reglas['WR8'])
    Wneu = np.fmax(np.fmax(reglas['WR1'], reglas['WR5']), reglas['WR9'])
    Wpos = np.fmax(np.fmax(reglas['WR2'], reglas['WR3']), reglas['WR6'])   
   
    
    # Calcular la activación de las reglas
    op_activation_low = np.fmin(Wneg, out_neg)
    op_activation_mid = np.fmin(Wneu, out_neu)
    op_activation_high = np.fmin(Wpos, out_pos)

    # Agregar las tres funciones de membresia
    aggregated = np.fmax(op_activation_low, np.fmax(op_activation_mid, op_activation_high))

    # Calcular el resultado defusificado
    out = (fuzzy.defuzz(x_out, aggregated, 'centroid'))
    puntaje_defuzzificado.append(round(out, 2))

    # Seleccionar la categoria luego de la desfusificación
    if 0<(out)<3.33:
        categoria_defuzzificada.append("negativo")
    elif 3.34<(out)<6.66:
        categoria_defuzzificada.append("neutral")
    elif 6.67<(out)<10:
        categoria_defuzzificada.append("positivo")

    end = time.perf_counter() 
    tiempo_ejecucion = round((end - start) * 1000, 10)  
    tiempos.append(tiempo_ejecucion)
    tiempo_total_promedio += tiempo_ejecucion 

columnas_nuevas = pd.DataFrame({
    'puntaje_defuzzificado': puntaje_defuzzificado,
    'categoria_defuzzificada': categoria_defuzzificada,
    'tiempo_de_ejecucion (ms)':tiempos
})

columnas_nuevas.reset_index(drop=True, inplace=True)
dataset = pd.concat([dataset.reset_index(drop=True), columnas_nuevas], axis=1)
dataset.to_csv('test_data_con_puntajes_y_categorias.csv', index=False)
print(f'Datos guardados en el archivo csv')


#benchmarks
dataset_final=dataset.copy()
dataset_final.drop(columns=['sentiment', 'puntaje_defuzzificado'], inplace=True)

dataset_final = dataset_final.rename(
    columns={
    'sentence':'tweet_original',
    'categoria':'categoria_original',
    'tweetPos': 'puntaje_positivo',
    'tweetNeg': 'puntaje_negativo',
    'categoria_defuzzificada': 'resultado_de_inferencia',
    }
)

dataset_final.to_csv('dataset_final.csv', index=False)
print(dataset_final.head(10))
print(f'El tiempo de ejecucion total promedio es de: {tiempo_total_promedio/100} milisegundos')
