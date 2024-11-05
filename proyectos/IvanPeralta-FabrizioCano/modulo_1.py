import pandas as pd
import nltk
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import  WordNetLemmatizer

def lemm_eliminate(tweets):
    #obtener la lista de palabras de parada en ingles
    palabras_parada=stopwords.words("english")
    #lemmatixador de manera a reducir las palabras a su lexema (EJ: changed y changing a change)
    lemm = WordNetLemmatizer() 
    #separar en tokens
    tokens=word_tokenize(tweets)

    texto_limpio=[]
    for t in tokens:
        if t not in palabras_parada:
            #se agrega al arreglo la palabra que ha sido lemmatizada
            texto_limpio.append(lemm.lemmatize(t))

    return " ".join(texto_limpio) #string de palabras

pd.options.display.max_colwidth=100

test_data=pd.read_csv('test_data.csv',usecols=['sentence','sentiment'])

#preprocesar el data set
#eliminar duplicados
test_data.drop_duplicates(inplace=True)
test_data.shape #reshape el data set
test_data.reset_index(drop=True,inplace=True)

#aplicar la funcion lemm_eliminate a el dataset
test_data['sentence']=test_data.sentence.apply(lemm_eliminate)
print(test_data.head(10))