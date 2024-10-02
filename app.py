import pickle 
import json


from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


import numpy  as np
import pandas as pd

app=Flask(__name__)

regmodel = pickle.load(open("regressionmodel.pkl","rb"))


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])   #prediction api , methode :post car    POST : Typiquement utilisé lorsqu'on envoie des données au serveur


def predict_api():
    data=request.json["data"]  #data = request.json["data"] a pour rôle de récupérer une valeur associée à la clé "data" dans un objet JSON envoyé via une requête POS
    #capture the result of HTTP POST request and store them in data 
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    data=np.array(list(data.values())).reshape(1,-1)
    output=regmodel.predict(data)
    return jsonify(output[0]) #jsonify(...) : Cette méthode renvoie la prédiction dans un format JSON pour que le client qui a fait la requête POST puisse la récupérer.


if __name__=="__main__" :
    app.run(debug=True)