

# Import statements should be grouped and separated from custom functions with two blank lines.
from flask import Flask, render_template, send_from_directory, request, session, jsonify, redirect, url_for
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import holidays
from datetime import date
import pickle

# Konstaner
APP_SECRET_KEY = 'INF161ErLættis'
FINAL_MODEL_FILE = 'final_model.sav'
IMPUTE_MODEL_FILE = 'impute_model.sav'
USECOLUMNS = 'Globalstraling Solskinstid Lufttemperatur Vindretning Vindstyrke Vindkast time ukedag måned ferie day'.split()
IMPUTECOLUMNS = ['Globalstraling', 'Solskinstid', 'Lufttemperatur', 'Vindretning', 'Vindstyrke', 'Lufttrykk', 'Vindkast', 'ferie', 'time', 'måned', 'year','day']


# Create time series dummy columns
def create_hour_day_month(df):
    df_new = df.copy()
    df_new['ukedag'] = df_new['dato'].dt.day_of_week
    df_new['måned'] = df_new['dato'].dt.month
    df_new['year'] = df_new['dato'].dt.year
    df_new['day'] = df_new['dato'].dt.day
    weekdays = df_new['dato'].dt.day_of_week
    return df_new, weekdays

#Imputere X dataen
def transform_to_transform(df, impute_KNN):
    new_df, weekdays = create_hour_day_month(df)
    new_df = new_df.drop(['dato', 'ukedag'], axis=1)
    new_df['Lufttrykk'] = [np.NaN]
    new_df['Vindretning'] = [np.NaN]
    new_df['Vindstyrke'] = [np.NaN]
    df_testcolumns = pd.DataFrame(new_df[IMPUTECOLUMNS])
    all_interpolated = impute_KNN.transform(df_testcolumns)
    df_all_interpolate = pd.DataFrame(all_interpolated, columns=IMPUTECOLUMNS)
    weekdays = weekdays.reset_index(drop=True)
    df_all_interpolate['ukedag'] = weekdays 
    return df_all_interpolate


# laste inn modellen
def load_model2():
    model = pickle.load(open(FINAL_MODEL_FILE, 'rb'))
    impute_knn = pickle.load(open(IMPUTE_MODEL_FILE, 'rb'))
    return (model, impute_knn)

# Initialisere Flask app
app = Flask(__name__, template_folder='templates', static_url_path='/static')
app.secret_key = APP_SECRET_KEY

# Norske Helligdager
NORWEGIAN_HOLIDAYS = holidays.country_holidays('NO', years=[i for i in range(2010, 2030)])
MODEL, IMPUTER = load_model2()

@app.route('/', methods=['POST', 'GET'])
def index():
    if 'date' not in session:
        session['date'] = date.today().strftime('%Y-%m-%d')
    if 'time' not in session:
        session['time'] = '00:00'
    if 'sunamount' not in session:
        session['sunamount'] = '0'
    if 'temp' not in session:
        session['temp'] = '0'
    return render_template('index.html')

@app.route('/getcalendar')
def getcalendar():
    return render_template('calendar.html', today=session['date'])

@app.route('/gettime')
def gettime():
    return render_template('time.html', time = session['time'])

@app.route('/getsolskinn')
def getsolskinn():
    return render_template('solskinn.html')

@app.route('/gettemperatur')
def gettemperatur():
    return render_template('temperatur.html')


@app.route('/final', methods=['GET', 'POST'])
def getfinal():
    midtemp = request.form.get('temp')
    print('temperatur:',midtemp)
    if midtemp is not None:
        session['temp'] = midtemp
        
    sol = session['sunamount']
    temp = session['temp']
    time = session['time']
    dato = session['date']
    d = {
        'Globalstraling' : [np.NaN],
        'Solskinstid' : [int(sol)],
        'Lufttemperatur' : [int(temp)],
        'Vindkast' : [np.NaN],
        'time' : [int(time[:2])],
    }
    #lage dataframen så maskinlæringsmodellen forstår den
    df = pd.DataFrame(d)
    df['dato'] = [pd.to_datetime(dato)]
    df['ukedag'] = [df['dato'].dt.dayofweek]
    df['måned'] = [df['dato'].dt.month]

    df['day'] = [df['dato'].dt.day]

    df['ferie'] = [1 if dato in NORWEGIAN_HOLIDAYS else 0]
    new_df = transform_to_transform(df, IMPUTER)
    new_df = new_df[USECOLUMNS]
    print(new_df)
    #gjette
    ans = MODEL.predict(new_df)

    ukedag = {0: 'Mandag', 1: 'Tirsdag', 2: 'Onsdag', 3: 'Torsdag', 4: 'Fredag', 5: 'Lørdag', 6: 'Søndag'}
    maned = {1: 'Januar', 2: 'Februar', 3: 'Mars', 4: 'April', 5: 'Mai', 6: 'Juni', 7: 'Juli', 8: 'August', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}

    df['dato'] = [pd.to_datetime(dato)]

    solskinn = 'sola skinner🌞😁'
    if sol < 8 and sol > 3:
        solskinn = 'delvis overskyet ⛅︎😊'
    elif sol <= 3:
        solskinn = 'ganske overskyet ☁️🌧️'

    return render_template('siste.html', 
                           res = 0 if ans[0] < 0 else round(ans[0]),
                           tid=time[:2],
                           tid2 = str(int(time[0:2]) + 1) if len(str(int(time[0:2]) + 1)) > 1 else '0' + str(int(time[0:2]) + 1),
                           ukedag = ukedag[df.loc[0,'ukedag'].squeeze()], 
                           dato = str(df['dato'].dt.day.squeeze()) + "." + str(maned[df.loc[0,'måned'].squeeze()]),
                           ar = df['dato'].dt.year.squeeze(),
                           temperatur = temp,
                           solskinn = solskinn)


@app.route('/postdate', methods=['POST', 'GET'])
def postdate():
    try:
        data = request.get_json() 
        date = data.get('date')
        if date is not None:
            print("dato entered:",date)
            session['date'] = date
        else:
            print("Error invalid input")
    except Exception as e:
        print("Error:", str(e)) 
        return "An error occurred", 500 

    return jsonify({"message": "Success"})


@app.route('/posttime', methods=['POST', 'GET'])
def posttime():
    try:
        data = request.get_json() 
        time = data.get('time')
        if time is not None:
            
            if len(time) < 2:
                session['time'] = "0" + time + ":00"
            else:
                session['time'] = time + ":00"
            print("time chosen:", session['time'])
        else:
            print("Error invalid input")
    except Exception as e:
        print("Error:", str(e)) 
        return "An error occurred", 500 

    return jsonify({"message": "Success"})


@app.route('/postposition', methods=['POST', 'GET'])
def postposition():
    try:
        data = request.get_json() 
        pos = data.get('pos')
        if pos is not None:
            session['sunamount'] = (pos/100)*10
            print("sunamount in session:", session['sunamount'])
        else:
            print("Error invalid input")
    except Exception as e:
        print("Error:", str(e)) 
        return "An error occurred", 500 

    return jsonify({"message": "Success"})


@app.route('/posttemp', methods=['POST', 'GET'])
def posttemp():
    try:
        data = request.get_json() 
        temp = data.get('temp')
        if temp is not None:
            session['temp'] = temp
            print("temperature in session:", session['temp'])
        else:
            print("Error invalid input")
    except Exception as e:
        print("Error:", str(e)) 
        return "An error occurred", 500 

    return jsonify({"message": "Success"})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)

