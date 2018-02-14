import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers

def predict(model, input, output):
    #read input
    df_input = pd.read_csv(input)

    #normalization, same as done during training
    df_input /= 255
    X_predict = df_input.values

    #load model
    model = keras.models.load_model(model)

    #predict
    y_predict = model.predict(X_predict).argmax(axis=1)
    
    #optional process the output, decode, etc.
     
    #write output
    df_output = pd.DataFrame(y_predict)
    df_output.to_csv(output, index=False, header=None)

if __name__ == "__main__":
    model = r'MNIST\keras.h5'
    input = r'MNIST\test.csv'
    output = r'MNIST\predict.csv'
    predict(model, input, output)
