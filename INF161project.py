# Modellutvalg og evaluering


#Standard Biblioteker
import pandas as pd
import numpy as np
import math
import plotly.express as px
import holidays
import pickle
import warnings
warnings.filterwarnings("ignore")

#MaskinLærings modeller
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeClassifier

#hjelpe funksjoner
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse

#imputering
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

#Konstanter
USECOLUMNS = 'Globalstraling Solskinstid Lufttemperatur Vindretning Vindstyrke Vindkast time ukedag måned ferie day'.split()
SEED = 10


######## HJELPEFUNKSJONER
# |
# |
# |
# |
# |
# |
# |
# |
# V

#make input trafic dataframe into fixed dataframe
def filter_trafikk_data(df):
    df = df.loc[:,['Dato', 'Fra tidspunkt','Trafikkmengde']][df['Felt'] == 'Totalt']
    df['Datotid'] = pd.to_datetime(df['Dato'] + ' ' + df['Fra tidspunkt'])
    df.drop(['Fra tidspunkt', 'Dato'], inplace= True, axis = 1)
    df['Datotid'] = df['Datotid'].dt.floor('H')
    return df

#make input weather dataframe into fixed dataframe
def filter_weather_data(df):
    filtered_df_weatherdata = df.drop('Relativ luftfuktighet', axis=1) #368305 manglende verdier var litt for mange, så fjerner dette
    filtered_df_weatherdata['Datotid'] = pd.to_datetime(filtered_df_weatherdata['Dato'] + ' ' + filtered_df_weatherdata['Tid'])
    filtered_df_weatherdata = filtered_df_weatherdata.drop(['Dato', 'Tid'], axis= 1)
    filtered_df_weatherdata['Datotid'] = filtered_df_weatherdata['Datotid'].dt.floor('H')
    return filtered_df_weatherdata

#Get an array with one hot encoded vacation 
def get_ferie_array(df):
    norwegian_holidays = holidays.country_holidays('NO', years=[i for i in range(2010,2025)])
    ferie_array = np.zeros(df.shape[0])
    dato_tid = df['Datotid']
    i = 0
    for datotid in dato_tid:
        if datotid in norwegian_holidays:
            ferie_array[i] = 1
        i+=1
    return ferie_array


#helper function to easier label plots x-axis, y-axis and legend
def updateplot(plot, title, x_label, y_label):
    plot.update_xaxes(title=x_label)
    plot.update_yaxes(title=y_label)
    plot.update_layout(title=title)

# Create time series dummy columns
def create_hour_day_month(df):
    df_new = df.copy()
    df_new['time'] = df_new['Datotid'].dt.hour
    df_new['ukedag'] = df_new['Datotid'].dt.day_of_week
    df_new['måned'] = df_new['Datotid'].dt.month
    df_new['year'] = df_new['Datotid'].dt.year
    df_new['day'] = df_new['Datotid'].dt.day
    weekdays = df_new['Datotid'].dt.day_of_week
    return df_new, weekdays

#Create a impute model and impute training X data
def impute_all(df):
    new_df, weekdays = create_hour_day_month(df)
    new_df = new_df.drop(['Datotid', 'ukedag'], axis=1)
    columns_to_threshold = ['Globalstraling', 'Lufttemperatur', 'Lufttrykk', 'Solskinstid', 'Vindkast', 'Vindretning', 'Vindstyrke']
    thresholds = [1000, 80, 1050, 11, 40, 366, 40]
    df_all = new_df.copy()
    for column, threshold in zip(columns_to_threshold, thresholds):
        df_all[column] = np.where(df_all[column] > threshold, np.nan, df_all[column])

    impute_KNN = KNNImputer(n_neighbors=4)
    all_interpolated = impute_KNN.fit_transform(df_all)
    df_all_interpolate = pd.DataFrame(all_interpolated, columns=list(df_all.columns))
    df_all_interpolate['ukedag'] = weekdays
    return df_all_interpolate, impute_KNN


#Transform X validation or test data that has not yet been imputed
def transform_to_transform(df, impute_KNN):
    new_df, weekdays = create_hour_day_month(df)
    new_df = new_df.drop(['Datotid', 'ukedag'], axis=1)
    columns_to_threshold = ['Globalstraling', 'Lufttemperatur', 'Lufttrykk', 'Solskinstid', 'Vindkast', 'Vindretning', 'Vindstyrke']
    thresholds = [1000, 80, 1050, 11, 40, 366, 40]
    df_all = new_df.copy()
    for column, threshold in zip(columns_to_threshold, thresholds):
        df_all[column] = np.where(df_all[column] > threshold, np.nan, df_all[column])
    all_interpolated = impute_KNN.transform(df_all)
    df_all_interpolate = pd.DataFrame(all_interpolated, columns=list(df_all.columns))
    weekdays = weekdays.reset_index(drop=True)
    df_all_interpolate['ukedag'] = weekdays 
    return df_all_interpolate


#create X traindata given merged data frame
def fix_xdata(df):
    final_df = df.copy()
    final_df['ferie'] = get_ferie_array(final_df)
    final_df.drop(['Trafikkmengde'], inplace= True, axis = 1)
    return final_df

def rmse(arr1, arr2):
    return math.sqrt(mse(arr1, arr2))

class TRAINMODELS:
    def __init__(self, models, X, y, cv = 1):
        self.models = models
        self.cv = cv
        self.SEED = 42
        self.X = X
        self.y = y
        self.predictions_val = {}
        self.errors_val = {}
        self.errors_test = {}
        self.predictions_test = {}
        self.valdatetime = None

    #Train and Test on validation 70-85% end of data
    def train_and_evaluate(self):
        self.x_train, self.x_test, self.y_train, self.y_test = tts(self.X, self.y, test_size= 0.3, shuffle=False)
        self.x_val, self.x_test, self.y_val, self.y_test = tts(self.x_test, self.y_test, test_size= 0.5, shuffle=False)
        save_train_datetime = self.x_train['Datotid']
        self.valdatetime = self.x_val['Datotid']
        self.x_train, imputemodel = impute_all(self.x_train)
        self.postimputationdf = self.x_train[['Globalstraling', 'Solskinstid', 'Lufttemperatur', 'Vindretning', 'Vindstyrke', 'Vindkast']]
        self.postimputationdf = self.postimputationdf.copy()
        self.postimputationdf['Datotid'] = save_train_datetime
        self.postimputationdf = self.postimputationdf[(self.postimputationdf['Datotid'] > '2016-09-18') & (self.postimputationdf['Datotid'] < '2016-10-01')]
        self.x_test = transform_to_transform(self.x_test, imputemodel)
        self.x_val = transform_to_transform(self.x_val, imputemodel)
        self.x_train, self.x_test, self.x_val = self.x_train[USECOLUMNS], self.x_test[USECOLUMNS], self.x_val[USECOLUMNS]
        index = 0
        for modelname, model in self.models.items():
            print(modelname,"loading ⏳")
            model.fit(self.x_train, self.y_train)
            pred_val = model.predict(self.x_val)
            pred_val = np.floor(pred_val)
            pred_val[pred_val < 0] = 0
            self.predictions_val[modelname] = pred_val
            self.errors_val[modelname] =  rmse(pred_val, self.y_val)
            pred_test = model.predict(self.x_test)
            pred_test = np.floor(pred_test)
            pred_test[pred_test < 0] = 0
            self.predictions_test[modelname] = pred_test
            self.errors_test[modelname]= rmse(pred_test, self.y_test)
            print(f"average RMSE for", modelname, "on validation data is:", self.errors_val[modelname])
            print(f"average RMSE for", modelname, "on test data is:", self.errors_test[modelname])
            print("\n")
            index+=1
            if (((len(self.models) // 10) * 10) % (index) == 0):
                if (len(self.models) // 10) != 0:
                    print(f"{round(((index) / ((len(self.models) // 10) * 10))*100, 2)}% done ✅")
                else:
                    print("Done ✅")
        
    #Train and evaluate on generalised Data, so train on first 85% and test on last 15%
    def train_and_evaluate_generalisation(self):
        self.x_train, self.x_test, self.y_train, self.y_test = tts(self.X, self.y, test_size= 0.15, shuffle=False)
        self.testdatetime = self.x_test['Datotid']
        self.x_train, imputemodel = impute_all(self.x_train)
        self.x_test = transform_to_transform(self.x_test, imputemodel)
        self.x_train, self.x_test = self.x_train[USECOLUMNS], self.x_test[USECOLUMNS]
        index = 0
        for modelname, model in self.models.items():
            print(modelname,"loading ⏳")
            model.fit(self.x_train, self.y_train)
            pred_test = model.predict(self.x_test)
            pred_test = np.floor(pred_test)
            pred_test[pred_test < 0] = 0
            self.predictions_test[modelname] = pred_test
            self.errors_test[modelname]= rmse(pred_test, self.y_test)
            print(f"average RMSE for", modelname, "on test generalisation data is:", self.errors_test[modelname])
            print("\n")
            index+=1
            if (((len(self.models) // 10) * 10) % (index) == 0):
                if (len(self.models) // 10) != 0:
                    print(f"{round(((index) / ((len(self.models) // 10) * 10))*100, 2)}% done ✅")
                else:
                    print("Done ✅")


    #Train model on entire data set and make predictions for 2023
    def fill_2023_missing_data(self, X_2023):
        self.new_X, imputemodel = impute_all(self.X)
        self.X_2023 = transform_to_transform(X_2023, imputemodel)
        self.X_2023, self.new_X = self.X_2023[USECOLUMNS], self.new_X[USECOLUMNS]
        index = 0
        for modelname, model in self.models.items():
            print(modelname,"loading ⏳")
            model.fit(self.new_X, self.y)
            pred_2023 = model.predict(self.X_2023)
            pred_2023 = np.floor(pred_2023)
            pred_2023[pred_2023 < 0] = 0
            self.predictions_2023 = pred_2023
            index+=1
            if (((len(self.models) // 10) * 10) % (index) == 0):
                if (len(self.models) // 10) != 0:
                    print(f"{round(((index) / ((len(self.models) // 10) * 10))*100, 2)}% done ✅")
                else:
                    print("Done ✅")
        
        final_model = open('server/final_model.sav', 'wb')
        pickle.dump(model, final_model)

        impute_model = open('server/impute_model.sav', 'wb')
        pickle.dump(imputemodel, impute_model)

        return pred_2023
    
    
    ####################
    # Hjelpe metode for å lagre modeller
    # med oversikt over RMSE i en seperat csv fil
    #
    def save_models(self):
        data = []
        for modelname, model in self.models.items():
            data.append({'models' : modelname, 
                            'rmse_val' : self.errors_val[modelname],
                            'rmse_test' : self.errors_test[modelname]})
        df = pd.read_csv('saved_models/models.csv', index_col=False)
        df = pd.concat([df, pd.DataFrame(data)])
        df = df.sort_values(by='rmse_val')
        df.to_csv('saved_models/models.csv', index=False)
    #
    #
    ###################


    #############
    # hjelpe funksjoner for 
    # å få interne 
    # klasse variabler
    def get_val_rmse(self, modelname):
        return self.errors_val.get(modelname, None)

    def get_test_rmse(self, modelname):
        return self.errors_test.get(modelname, None)
    
    def get_pred_test(self, modelname):
        return self.predictions_test[modelname]

    def get_pred_val(self, modelname):
        return self.predictions_val[modelname]

    def get_trained_models(self):
        return self.models

    def get_datetime_val(self):
        return self.valdatetime
    
    def get_datetime_test(self):
        return self.testdatetime
    
    def get_y_val(self):
        return self.y_val
    
    def get_y_test(self):
        return self.y_test
    
    def get_post_imputation(self):
        return self.postimputationdf
    #
    #
    ############

# ˆ
# |
# |
# |
# |
# |
# |
# |
########

######## DATA Innlesing, fiksing og imputering av manglende verdier i værdata
# |
# |
# |
# |
# |
# |
# |
# |
# V

weather_data = {
    2010:'Florida_2010-01-01_2011-01-01_1654174747.csv',
    2011:'Florida_2011-01-01_2012-01-01_1654174772.csv',
    2012:'Florida_2012-01-01_2013-01-01_1654174811.csv',
    2013:'Florida_2013-01-01_2014-01-01_1654174853.csv',
    2014:'Florida_2014-01-01_2015-01-01_1654174868.csv',
    2015:'Florida_2015-01-01_2016-01-01_1654174882.csv',
    2016:'Florida_2016-01-01_2017-01-01_1654174902.csv',
    2017:'Florida_2017-01-01_2018-01-01_1654174925.csv',
    2018:'Florida_2018-01-01_2019-01-01_1654175073.csv',
    2019:'Florida_2019-01-01_2020-01-01_1654174955.csv',
    2020:'Florida_2020-01-01_2021-01-01_1654174973.csv',
    2021:'Florida_2021-01-01_2022-01-01_1654174989.csv',
    2022:'Florida_2022-01-01_2023-01-01_1688719054.csv',
    2023:'Florida_2023-01-01_2023-07-01_1688719120.csv'
}

trafikk_data_str = 'raw_data/trafikkdata.csv'

df_trafikk = pd.read_csv(trafikk_data_str, delimiter=';')

all_weather_df = pd.DataFrame()

for key,val in weather_data.items():
    weather_data[key] = pd.read_csv('raw_data/' +val, sep=',')
    
all_weather_df = pd.concat(weather_data.values(), ignore_index=True)


#Sortere ut nødvendige linjer
filtered_df_trafikk = filter_trafikk_data(df_trafikk)


#alltrafikkplot
trafikk_mengde_display = px.line(filtered_df_trafikk, x='Datotid', y= 'Trafikkmengde', title='Antall Syklister')


# Finne 17. september og gjøre om trafikkmengden til det den var 7 dager før
verdi_på_10til17_september = filtered_df_trafikk[(filtered_df_trafikk['Datotid'] >= '2017-09-10') & (filtered_df_trafikk['Datotid'] <= '2017-09-17')]['Trafikkmengde'].values
filtered_df_trafikk.loc[(filtered_df_trafikk['Datotid'] >= '2017-09-17') & (filtered_df_trafikk['Datotid'] <= '2017-09-24'), 'Trafikkmengde'] = verdi_på_10til17_september
filtered_df_trafikk['Trafikkmengde'] = pd.to_numeric(filtered_df_trafikk['Trafikkmengde'], errors='coerce')
filtered_df_trafikk.dropna(inplace=True)

weather_df = filter_weather_data(all_weather_df)
#weather_df_imputed = impute_weather(weather_df)

#sveise sammen
merged_data = pd.merge(filtered_df_trafikk, 
                       weather_df[['Datotid','Globalstraling', 'Solskinstid', 'Lufttemperatur', 'Vindretning', 'Vindstyrke', 'Lufttrykk', 'Vindkast']], 
                       on='Datotid', how = 'left')

merged_data = merged_data.groupby('Datotid').mean().reset_index()

#correlasjonsmatrise
correlation_matrix = merged_data.corr()
confusion_matrix = px.imshow(correlation_matrix, color_continuous_scale='YlOrRd', zmin=0, zmax=1, title='Korrelasjons matrise')

#før imputering plot 
oct17_df = weather_df[(weather_df['Datotid'] > '2016-09-18') & (weather_df['Datotid'] < '2016-10-01')]
oct17plot = px.line(oct17_df, x = 'Datotid', y= [col for col in oct17_df if (col != 'Datotid' and col != 'Lufttrykk')])
updateplot(oct17plot, 'visualisere manglende verdier', 'tidspunkt', 'verdi til vær')


#tidspunkt plots
merged_data_dates = merged_data.copy()
merged_data_dates['time'] = np.array(merged_data_dates['Datotid'].dt.hour)
merged_data_dates['ukedag'] = np.array(merged_data_dates['Datotid'].dt.weekday)
merged_data_dates['måned'] = np.array(merged_data_dates['Datotid'].dt.month)



for col in merged_data_dates.columns[-3:]:
    histplot = px.histogram(merged_data_dates, x = col, y = 'Trafikkmengde')
    updateplot(histplot, f'sum av antal syklister i {col}', col, 'antall syklister')



print("\n"*6,"Lese inn data ✅")
print("\n"*6)

# ˆ
# |
# |
# |
# |
# |
# |
# |
########


######## Dele data, imputere treningsdata. meningsfull tidsavhengig oppdeling, modeltilpassinger
# |
# |
# |
# |
# |
# |
# |
# |
# V

print("trene og teste modeller 📀", "\n"*6)

#få Xdata
X = fix_xdata(merged_data)
#få ydata
y = merged_data['Trafikkmengde']

#beste model i dictionarien
models = {
    'GradientBoosterRegressor winner': GradientBoostingRegressor(
        loss='squared_error', 
        learning_rate=0.03, 
        n_estimators=140, 
        subsample=0.6, 
        max_depth=9, 
        alpha=0.4,
        criterion='squared_error', 
        min_samples_split=10, 
        min_samples_leaf=3, 
        tol=0.0001,
        random_state=SEED #for reproduksjon og sammenligning
    )
}

#trene model og teste på validerings data og test data, men ta hensyn til validering
model_trainer = TRAINMODELS(models, X, y)
model_trainer.train_and_evaluate()

print("modeller ferdig kjørt ✅")


#etter imputering plot
oct17_df = model_trainer.get_post_imputation()
oct17plot = px.line(oct17_df, x='Datotid', y=[col for col in oct17_df.columns if (col != 'Datotid' and col != 'Lufttrykk')])
updateplot(oct17plot, 'Etter påfylle NaN ved hjelp av KNNImputer', 'Tidspunkt', 'verdi av værdata')


# ˆ
# |
# |
# |
# |
# |
# |
# |
########
######## Generalisering
# |
# |
# |
# |
# |
# |
# |
# |
# V

print("generalisering....","\n"*6)
#trene ny modell med trening og validering for å teste generalisering på test data
model_trainer.train_and_evaluate_generalisation()

realvals_df = pd.DataFrame()
realvals_df['real'] = model_trainer.get_y_test()
realvals_df['preds'] = model_trainer.get_pred_test('GradientBoosterRegressor winner')
realvals_df['Datotid'] = model_trainer.get_datetime_test()

# ekte vs prediksjon plot
barchart1 = px.line(realvals_df, x='Datotid', y=['real', 'preds'])
updateplot(barchart1,'Ekte verdier vs Gjettet verdier', 'Dato og Tid','Mengde syklister')

print("generalisering✅","\n"*6)

# ˆ
# |
# |
# |
# |
# |
# |
# |
########
######## 2023 data prediksjon
# |
# |
# |
# |
# |
# |
# |
# |
# V

print("2023 predikering....","\n"*6)

#importere 2023 Trafikk data
filtered_df_trafikk = filter_trafikk_data(df_trafikk)
df_trafikk_2023 = filtered_df_trafikk[filtered_df_trafikk['Datotid'] >= '2023-01-01']
df_trafikk_2023 = df_trafikk_2023.copy()
df_trafikk_2023['Trafikkmengde'] = pd.to_numeric(df_trafikk_2023['Trafikkmengde'], errors='coerce')


#ny merge
merged_data_2023 = pd.merge(df_trafikk_2023, 
                       weather_df[['Datotid','Globalstraling', 'Solskinstid', 'Lufttemperatur', 'Vindretning', 'Vindstyrke', 'Lufttrykk', 'Vindkast']], 
                       on='Datotid', how = 'left')

merged_data_2023 = merged_data_2023.groupby('Datotid').mean().reset_index()

X_2023 = fix_xdata(merged_data_2023)

y_2023_guess = model_trainer.fill_2023_missing_data(X_2023)

#lage predictions.csv
pd.DataFrame({'Dato' : X_2023['Datotid'].dt.date,'Tid' : X_2023['Datotid'].dt.time, 'Prediksjon': y_2023_guess}).to_csv('predictions.csv', index=False)

print("2023 prediksjon ferdig ✅","\n"*6)

# ˆ
# |
# |
# |
# |
# |
# |
# |
########


