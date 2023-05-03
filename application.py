from flask import Flask,request,jsonify,render_template
# render template--finding url of html file
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application


# Interact with ridge.pkl and scaler.pkl

ridge_model=pickle.load(open('Models/ridge (1).pkl','rb'))
standard_scaler=pickle.load(open('Models/scaler (1).pkl','rb'))


# Home Page
# route for home page
@app.route("/")
def index():
   return render_template('index.html')

# if i search for google.com i just get a page its a get request
# when i search something ie sending then its a post reques
# all input fields we will post data and FWI will be predicted
# post ---interact with ridge and get output





# when we just enter in the url /predictdata so we will only get the form...coz its a get method and in get method we are rendering home.html

# when we will enter values then and click on predict it will hit the predict button  and our url will get hit

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data)
# result will be in form of a list so write result[0]
        return render_template('home.html',results=result[0])
    # the results is used in home.html
    
    else:
        return render_template('home.html')




# mapped to local ip address as 0.0.0.0
# local ip address not publicly availaible

if __name__=="__main__":
    app.run(host="0.0.0.0")
