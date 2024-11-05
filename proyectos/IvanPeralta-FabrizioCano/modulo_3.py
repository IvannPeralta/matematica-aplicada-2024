import numpy as np
import skfuzzy as fuzzy
import matplotlib.pyplot as plt
import pandas as pd

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

for index, tweet in dataset.iterrows():
    tweetPos = tweet['tweetPos']
    tweetNeg = tweet['tweetNeg']

    print('\n\ntweet: '+tweet['sentence'])
    print('\ntweetPos: '+str(tweetPos))
    print('tweetNeg: '+str(tweetNeg))

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

    op0 = np.zeros_like(x_out)

    # Agregar las tres funciones de membresia
    aggregated = np.fmax(op_activation_low, np.fmax(op_activation_mid, op_activation_high))

    print("\nFiring Strength of Negative (wneg): "+str(round(Wneg,4)))
    print("Firing Strength of Neutral (wneu): "+str(round(Wneu,4)))
    print("Firing Strength of Positive (wpos): "+str(round(Wpos,4)))
    
    print("\nResultant consequents MFs:" )
    print("op_activation_low: "+str(op_activation_low))
    print("op_activation_med: "+str(op_activation_mid))
    print("op_activation_high: "+str(op_activation_high))
    
    print("\nAggregated Output: "+str(aggregated))


