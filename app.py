from flask import Flask ,render_template ,request,jsonify,url_for
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)


app=Flask(__name__)

@app.route('/')
@app.route('/prediction',methods=['GET','POST'])
def Prediction():
    if request.method=='POST':
        MedInc=float(request.form.get('MedInc'))
        HouseAge=float(request.form.get('HouseAge'))
        AveRooms=float(request.form.get('AveRooms'))
        AveBedrms=float(request.form.get('AveBedrms'))
        Population=float(request.form.get('Population'))
        AveOccup=float(request.form.get('AveOccup'))
        Latitude=float(request.form.get('Latitude'))
        Longitude=float(request.form.get('Longitude'))

        input=[[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]]
        result=model.predict(input)

        return render_template('index.html',results=round(result[0],3))
    else:
        return render_template('index.html')

if (__name__=='__main__'):
    app.run(debug=True)