import numpy as np
import pandas as pd
import socket
import json

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

    #predict
    y_predict = model.predict(X_predict).argmax(axis=1)
    
    #optional process the output, decode, etc.
     
    #write output
    df_output = pd.DataFrame(y_predict)
    df_output.to_csv(output, index=False, header=None)

model = r'MNIST\keras.h5'

print('Loading model')
#load model
model = keras.models.load_model(model)

host, port = '', 1234
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((host, port))
listen_socket.listen(1)
print('Serving HTTP on port', port)

while True:
    client_connection, client_address = listen_socket.accept()
    request = client_connection.recv(1024)
    request_string = request.decode()
    #print(request_string)

    body_start = request_string.find('Body')
    if body_start == -1:
        print('Bad request', request_string)
        client_connection.close()
        continue

    request_string = request_string[body_start:]
    print('New request', request_string)

    parameters = request_string.split('\r\n')
    predict(model, parameters[1], parameters[2])

    http_response = """\
HTTP/1.1 200 OK
Hello, World!
"""
    client_connection.sendall(http_response.encode())
    client_connection.close()

    print('Request done')
