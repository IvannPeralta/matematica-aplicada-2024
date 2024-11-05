import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def vader_analisis(test_data):
    #se utiliza un analizador de intensidad de sentimientos de Vader
    vader = SentimentIntensityAnalyzer()
    #arreglos que contendran los valores tweetPos, tweetNeg y TweetComp (combinado) de cada tweet
    tweetPos = []
    tweetNeg = []
    categoria_sentimientos = []    
    
    #iteramos sobre cada tweet en el dataset
    for tweet in test_data['sentence']:
        #obtenemos los puntajes de sentimiento o polaridad del tweet y asignamos cada puntaje a su respectiva categoria
        puntaje = vader.polarity_scores(tweet)
        puntaje_pos = puntaje['pos']
        puntaje_neg = puntaje['neg']

        #si el puntaje positivo es mayor en valor absoluto al negarivo, la categoria del tweet es positivo
        if puntaje_pos > abs(puntaje_neg):
            categoria_sentimientos.append('positivo')
        #si el puntaje positivo es menor en valor absoluto al negarivo, la categoria del tweet es negativo
        elif puntaje_pos < abs(puntaje_neg):
            categoria_sentimientos.append('negativo')
        #en otro caso, la categoria es neutral
        else:
            categoria_sentimientos.append('neutral')

        #agregamos la lista de puntajes a los arreglos
        tweetPos.append(round(puntaje_pos, 1))
        tweetNeg.append(round(puntaje_neg, 1))

    #creamos nuevas columnas en el dataset para almacenar los datos obtenidos anteriormente
    columnas_nuevas = pd.DataFrame({
        'tweetPos': tweetPos,
        'tweetNeg': tweetNeg,
        'categoria': categoria_sentimientos
    })

    #se resetea el index del dataset ya que se agregaron columnas nuevas
    columnas_nuevas.reset_index(drop=True, inplace=True)
    #se concatena el dataset anterior y las columnas nuevas creadas
    test_data = pd.concat([test_data.reset_index(drop=True), columnas_nuevas], axis=1)
    #se convierte el dataset a un archivo csv
    test_data.to_csv('test_data_con_puntajes.csv', index=False)
    
    print(test_data.head(10))

#Main
test_data = pd.read_csv('test_data.csv', usecols=['sentence', 'sentiment'])
vader_analisis(test_data)
