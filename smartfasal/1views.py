from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_deny
from django.views.decorators.clickjacking import xframe_options_sameorigin
#from .models import FTP_session_model
import pandas as pd
import numpy as np
import os
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['interactive'] = False


#def plot_graph(list_1, list_2,label):
    #plt.plot(list_1, list_2, label=label)
    ##plt.ylabel(label)
    #plt.xlabel('Time')
    #plt.title(label + 'of the Day ')
    #plt.legend()
    #plt.savefig('static/' + label + '.png')


from django.views.decorators.clickjacking import xframe_options_exempt

@xframe_options_exempt
def ok_to_load_in_a_frame(request):
    return HttpResponse("This page is safe to load in a frame on any site.")



class FTP_session_view:

    def index(request):
        return render(request,'home.html')

    def visual(request):
        return render(request,'visualise.html')


def Real_time_plot1(request):
    ######### Phase 1:- Data aloocating and defining
    import plotly.graph_objects as go
    import plotly
    import numpy as np
    import plotly.express as px

    ########### phase :: Accessing the downloaded file (Data1.csv)
    import os
    from smartfasal_project.settings import BASE_DIR
    STATIC_ROOT = os.path.join(BASE_DIR, 'static')
    path = STATIC_ROOT
    os.chdir(path)
    import pandas as pd
    filename = "data1.csv"
    #Local_data_smartfasal = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes'])
    #Local_data_smartfasal = pd.read_csv(filename)
    import plotly.express as px

    df = pd.read_csv('Last_rows.csv')


    x_Time = df['Time']
    y0 = df['SM10']
    y1 = df['SM45']
    y2 = df['SM80']
    y3 = df['Temp']
    y4 = df['Humd']
    y5 = df['LMNS']
    y6 = df['PRSR']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_Time, y=y0, mode='lines', name='SM @ 10 cm',
                            marker = dict(color = 'rgba(255,0,0,0.8)')))
    fig.add_trace(go.Scatter(x=x_Time, y=y1, mode='lines', name='SM @ 45 cm',
                            marker = dict(color = 'rgba(0,255,0,0.8)')))
    fig.add_trace(go.Scatter(x=x_Time, y=y2, mode='lines', name='SM @ 80 cm',
                            marker = dict(color = 'rgba((26, 102, 255,0.8)')))
    fig.add_trace(go.Scatter(x=x_Time, y=y3, mode='lines', name='Temperature',
                            marker = dict(color = 'rgba(204, 0, 204, 0.8)')))
    fig.add_trace(go.Scatter(x=x_Time, y=y4, mode='lines', name='Humidity',
                            marker = dict(color = 'rgba(0, 153, 51, 0.8)')))
    fig.add_trace(go.Scatter(x=x_Time, y=y5, mode='lines', name='Luminisity',
                            marker = dict(color = 'rgba(0, 0, 204, 0.8)')))
    fig.add_trace(go.Scatter(x=x_Time, y=y6, mode='lines', name='Pressure',
                            marker = dict(color = 'rgba(80, 26, 80, 0.8)')))
    fig.show()

    import chart_studio
    username = 'sukhi02' # your usernam
    api_key = 'VQ5pvk3TMJi50tDGdWne' # your api key - go to profile > settings > regenerate keychart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)


    import chart_studio.plotly as csp
    csp.plot(fig,   showLink= 'false');

    #Plotly.plot(divid, data, layout, {showLink: false})
    #plt_div = plot(fig, output_type='div', include_plotlyjs=False)
    #return render(request,'www.http://smartfasal.in/wp/?page_id=277')
    return HttpResponse("Prcoessed completed")







class Plots:

    #@xframe_options_exempt
    def ftp_login(request):
        url='ftp.smartfasal.in'
        username = 'testuser@smartfasal.in'
        pwd = 'fasal@thapar'
        print("Setting the URL >>  " + url)

        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)
        file_path = os.path.join(BASE_DIR, path)
        import ftplib
        ftp = ftplib.FTP(url, username, pwd)
        files = ftp.dir()
        ftp.cwd("/")
        print("Downloading the files")
        filename = 'S_AgriB.csv'
        my_file = open(filename, 'wb') # Open a local file to store the downloaded file
        ftp.retrbinary('RETR ' + filename, my_file.write) # Enter
        ftp.quit() # Terminate the FTP connection
        my_file.close() # Close the local file you had opened for do
        print("Closing the FTP connection")
        return Plots.real_plot(request)


    #@xframe_options_exempt
    def real_plot(request):
        print("Accesing the FTP")
        import pandas as pd
        import numpy as np
        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)

        print(" Step 6: Library imported")

        #Load the CSV file
        print(" Step 7: Load the CSV file")

        data_file = pd.read_csv("S_AgriB.csv")

        print(" >>>     File loaded succesfiully")
        print(data_file.head())

        ############################################################

        print("##################################################")
        print(" Part 3: Modifying the downloaded file")
        print("##################################################")

        print(" Step 8: Load the dataset")

        cols_names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','a', 'b', 'c']

        data_file.columns = cols_names

        data_file.to_csv("s.csv")
        print("         Column name upgraded")

        print("         Convert epochs to date-time")

        date_file = pd.read_csv("s.csv")
        date_file = data_file.drop(['S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','a', 'b', 'c'], axis = 1)
        cols_names=['Time']
        date_file.columns = cols_names
        date_file.to_csv("date_file.csv")

    ###########################################################
    ##########################################################
        def dateparse(time_in_secs):
           import datetime
           time_in_secs = time_in_secs
           return datetime.datetime.fromtimestamp(float(time_in_secs))

        dtype= {"Time": float, "Value":float}
        date_time = pd.read_csv("date_file.csv", dtype=dtype, parse_dates=["Time"], date_parser=dateparse)
        print(date_time)
        print("  >> Successfully converted epochs to the date_time_format")
        #######################
        #######################
        #######################
        date_time = date_time['Time']

        data_file.index = date_time
        data_file = data_file.drop(['Timestamp','Battery', 'Readings', 'a', 'b', 'c'], axis =1)
        print(" >>   Data cleaned successfully")

        ######################
        print("##################################################")
        print("    Part 4: Import 400 observations")
        print("##################################################")

        last_rows = data_file.iloc[-400:, 0:7]
        date_time = last_rows.index
        del date_file
        print("last 400 observations are succssfully imported")

        print("step 9: plot the dataset")
        #import matplotlib.pyplot as plt

        #plt.plot(last_rows, label = 'Predicted')
        #plt.xlabel('Time')
        #plt.ylabel('Lumisity')
        #plt.title('Forecasted Lum')
        #plt.legend()
        #plt.savefig('Lum ARIMA FULL')
        #plt.show()
        print("Succssfully plotted")

        del dtype

        ######################
        print("##################################################")
        print("    Part 5:  Scalling the dataset")
        print("##################################################")

        sm1 = last_rows.iloc[:, 0:1].values
        sm2 = last_rows.iloc[:, 1:2].values
        sm3 = last_rows.iloc[:, 2:3].values
        Temp= last_rows.iloc[:, 3:4].values
        Humd= last_rows.iloc[:, 4:5].values
        Prsr= last_rows.iloc[:, 5:6].values
        #Prsr = last_rows["Pressure"]
        Lmns= last_rows.iloc[:, 6:7].values

        print("\n Checking Prresure for :")
        print("OK")


        print("Converting colon to string")
        Prsr = Prsr.astype("|S5")
        print(Prsr)
        print("Strings to the array of float 64")
        Prsr = Prsr.astype(np.float)



        #for i in range(len(Prsr)):
        #    value = Prsr[i]
        #    Prsr[i] = value[:-3]
        #print("OK")

        print("\n Checked")
        sm1 = sm1/100
        sm2 = sm2/100
        sm3 = sm3/100
        Prsr= Prsr/1000
        Lmns= Lmns/100



        sm1 = pd.DataFrame(sm1, index = date_time, columns=['SM10'])
        sm2 = pd.DataFrame(sm2, index = date_time, columns=['SM45'])
        sm3 = pd.DataFrame(sm3, index = date_time, columns=['SM80'])
        Temp= pd.DataFrame(Temp, index = date_time, columns=['Temp'])
        Humd= pd.DataFrame(Humd, index = date_time, columns=['Humd'])
        Prsr= pd.DataFrame(Prsr, index = date_time, columns=['Prsr'])
        Lmns= pd.DataFrame(Lmns, index = date_time, columns=['Lmns'])

        last_rows = pd.concat([sm1, sm2, sm3, Temp, Humd, Prsr, Lmns], axis =1)
        last_rows.to_csv("Last_rows.csv")


        del sm1, sm2, sm3, Humd, Lmns, Temp, Prsr,  cols_names, data_file, date_time
        print (" Cache Cleared")
        #plt.plot(last_rows)
        #plt.xlabel('Time')
        #plt.ylabel('Lumisity')
        ##plt.title('Forecasted Lum')
        #plt.legend()
        #plt.savefig(' new Plot')
        #plt.show()
        print("Succssfully plotted")
        print(" Memory Empty")
        del last_rows
        #return HttpResponse("D O N E ")
        return Plots.Real_time_plot(request)

    #@xframe_options_deny
    def Real_time_plot(request):
        ######### Phase 1:- Data aloocating and defining
        import plotly.graph_objects as go
        import plotly
        import numpy as np
        import plotly.express as px
        from django.shortcuts import redirect
        print("Library imported of the third function")

        ########### phase :: Accessing the downloaded file (Data1.csv)
        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)
        import pandas as pd
        print("Path set")
        print("Importing the created dataset >> Lastrows.csv")
        #filename = "data1.csv"
        #Local_data_smartfasal = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes'])
        #Local_data_smartfasal = pd.read_csv(filename)
        import plotly.express as px

        df = pd.read_csv('Last_rows.csv')
        print("Succesfully loaded")
        print("##########################################")

        print("Loading the columns")

        x_Time = df['Time']
        y0 = df['SM10']
        y1 = df['SM45']
        y2 = df['SM80']
        y3 = df['Temp']
        y4 = df['Humd']
        y5 = df['Prsr']
        y6 = df['Lmns']


        print("Succesfully loaded all columns to the model")
        print("##############################################")
        print ("Making the plots")
        print("###############################################")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y0, mode='lines', name='SM @ 10 cm',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))
        print("Plot_1_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y1, mode='lines', name='SM @ 45 cm',
                                marker = dict(color = 'rgba(0,255,0,0.8)')))
        print("Plot_2_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y2, mode='lines', name='SM @ 80 cm',
                                marker = dict(color = 'rgba(26, 102, 255,0.8)')))
        print("Plot_3_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y3, mode='lines', name='Temperature',
                                marker = dict(color = 'rgba(204, 0, 204, 0.8)')))
        print("Plot_4_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y4, mode='lines', name='Humidity',
                                marker = dict(color = 'rgba(0, 153, 51, 0.8)')))
        print("Plot_5_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y5, mode='lines', name='Luminisity',
                                marker = dict(color = 'rgba(0, 0, 204, 0.8)')))
        print("Plot_6_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y6, mode='lines', name='Pressure',
                                marker = dict(color = 'rgba(80, 26, 80, 0.8)')))
        print("Plot_7_ok")
        #fig.show()
        print("Succsfully created all plots")
        print("################################################")
        print("Plotting SUccessfull")
        print("#####################")
        print("###########################################")

        print("Importing Chart Studio")

        layout1 = go.Layout(margin=go.layout.Margin(
                            l=0, #left margin
                            r=0, #right margin
                            b=0, #bottom margin
                            t=0  #top margin
                            )
                        )
        print("Fetching the path")
        #os.chdir("/")
        print( os.path.dirname(path))
        print("Changing the path")
        #os.chdir("templates")
        print("New path")
        file_path = os.path.join(BASE_DIR, 'templates')
        os.chdir(file_path)
        print("Path Finally set")

        filename='plots_of_all_readings.html'
        import os
        #os.remove(filename)
        print("Printing the plot")
        # Plot and embed in ipython notebook!
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))
        print("Finalised everything and packup")

        print("Printing the plot on the smartfasal portal")
        #import chart_studio.plotly as csp
        #csp.plot(fig, showLink=False,fileopt='overwrite', filename='RPi')
        #csp.plot(fig,  showLink= 'false');
        #plt.clf()
        #print("Process Accomplised")


        print("################################################")
        print("Soil Moisture 10 cm")
        print("#####################")
        print("###########################################")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y0, mode='lines', name='SM @ 10 cm',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))
        filename='plots_of_smonereadings.html'
        import os
        #os.remove(filename)
        print("Printing the plot")
        # Plot and embed in ipython notebook!
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))
        print("Finalised everything and packup")

        print("Printing the plot on the smartfasal portal")
        print("Process Accomplised S_M_10cm")







        print("################################################")
        print("Soil Moisture 45 cm")
        print("#####################")
        print("###########################################")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y1, mode='lines', name='SM @ 45 cm',
                                                        marker = dict(color = 'rgba(0,255,0,0.8)')))
        print("Plot_2_ok")

        filename='plots_of_smtworeadings.html'
        import os
        #os.remove(filename)
        print("Printing the plot")
        # Plot and embed in ipython notebook!
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))
        #plt.clf()
        print("SM_45 Process Accomplised")









        print("################################################")
        print("Soil Moisture 80 cm")
        print("#####################")
        print("###########################################")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y2, mode='lines', name='SM @ 80 cm',
                                marker = dict(color = 'rgba(26, 102, 255,0.8)')))
        print("Plot_3_ok")

        filename='plots_of_smthreereadings.html'
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))

        print("SM_80 Process Accomplised")

        print("################################################")
        print("Temperature")
        print("#####################")
        print("###########################################")


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y3, mode='lines', name='Temperature',
                                marker = dict(color = 'rgba(26, 102, 255,0.8)')))
        print("Plot_4_ok")

        filename='plots_of_tempreadings.html'
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))

        print("Temperature Process Accomplised")

        print("################################################")
        print("Humidity")
        print("#####################")
        print("###########################################")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y4, mode='lines', name='Humidity',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))


        filename='plots_of_humidreadings.html'
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))

        print("Pressure Humidity Process Accomplised")

        print("################################################")
        print("Pressure")
        print("#####################")
        print("###########################################")


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y5, mode='lines', name='Pressure',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))


        filename='plots_of_prsrreadings.html'
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))
        print("Pressure Printing the plot on the smartfasal portal")

        print("Pressure Process Accomplised")

        print("################################################")
        print("Lumiousity")
        print("#####################")
        print("###########################################")


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y6, mode='lines', name='Lumiousity',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))

        filename='plots_of_lmnsreadings.html'
        plotly.offline.plot(fig,  filename=filename, auto_open= False, config=dict(displayModeBar=False))


        print("Lmns Pressure Process Accomplised")
        print("\n\n redirected to the smartfasal.in")
        #return render(request,'home.html')
        return redirect("http://smartfasal.in/agriculture_waspmote/")
        #return HttpResponse("Prcoessed completed")


        #Plotly.plot(divid, data, layout, {showLink: false})
        #plt_div = plot(fig, output_type='div', include_plotlyjs=False)
        #return render(request,'www.http://smartfasal.in/wp/?page_id=277')

class ARIMA_code:
    def HEroku_ARIMA_upload(request):
        import warnings
        import pandas as pd
        from pandas import read_csv
        from statsmodels.tsa.arima_model import ARIMA
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from math import sqrt
        import os
        import numpy as np
        import joblib
        from sklearn.metrics import r2_score

        import pandas as pd
        from pandas import read_csv
        import os
        import numpy as np
        #os.chdir("D:\CSIR_Smart_Fasal\cODE\TIME Series\SmartFasal\ARIMA")
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)

        import matplotlib.pyplot as plt

        warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                                FutureWarning)
        warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                                FutureWarning)

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



        print("Loading the file")
        filename = "S_AgriB.csv"
        raw_dataset = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','Day', 'Date', 'Time'])
        print("Okay")


        print("Setting the index")
        Date_Time = raw_dataset['Timestamp']
        print("1. Timestamp variable created")

        Date_Time1 = pd.to_datetime(Date_Time, unit='s')
        print("2. Converted Epochs to the Date_Time")
        raw_dataset.index = Date_Time1
        print("3. Index passed to the raw_dataset")
        print("Okay")

        print("Drpping the extra columns")
        raw_dataset = raw_dataset.drop(['Timestamp', 'Battery','Readings','Day', 'Date', 'Time'], axis = 1)
        print("Describtions of the dataset")
        print(raw_dataset.describe())
        print("Okay")



        def fifteen_minutes(dataset):
            print("Dataset loaded")
            len_data = len(dataset)
            print("Number of rows :", len_data)
            for i in range(1,len_data, 50):
                dataset = dataset.resample('15min').mean()
            print("Datset converted")
            return dataset

        raw_dataset = fifteen_minutes(raw_dataset)
        print("Dataset Loaded")
        print("COnverted to 15 minutes")


        dataset =               raw_dataset.iloc[:, 0:8]
        dataset_10 =            raw_dataset.iloc[:, 0:1]
        dataset_45 =            raw_dataset.iloc[:, 1:2]
        dataset_80 =            raw_dataset.iloc[:, 2:3]
        dataset_Temperature =   raw_dataset.iloc[:, 3:4]
        dataset_Humidity =      raw_dataset.iloc[:, 4:5]
        dataset_Pressure =      raw_dataset.iloc[:, 5:6]
        dataset_Lum =           raw_dataset.iloc[:, 6:7]
        print("Splitted into 7 columns")
        dataset_10  = dataset_10.fillna(dataset_10.mean())
        dataset_45  = dataset_45.fillna(dataset_45.mean())
        dataset_80  = dataset_80.fillna(dataset_80.mean())
        dataset_Temperature  = dataset_Temperature.fillna(dataset_Temperature.mean())
        dataset_Humidity  = dataset_Humidity.fillna(dataset_Humidity.mean())
        dataset_Pressure  = dataset_Pressure.fillna(dataset_Pressure.mean())
        #dataset_Lum  = dataset_Lum.fillna(dataset.mean())
        print("")
        print("Dataset Splitted")
        print("")
            # split into train and test sets
        train_size = int(len(dataset) * 0.80)
        test_size= len(dataset) - train_size
        total_size = train_size+1
        print("")


        train_date_time =           Date_Time1[:train_size]
        train_data_10 =             dataset_10[:train_size].values
        train_data_45 =             dataset_45[:train_size].values
        train_data_80 =             dataset_80[:train_size].values
        train_data_Temperature =    dataset_Temperature[:train_size].values
        train_data_Humidity =       dataset_Humidity[:train_size].values
        train_data_Pressure =       dataset_Pressure[:train_size].values
        train_data_Lum =            dataset_Lum[:train_size].values
        test_data_10  =             dataset_10[train_size+1: ].values
        test_data_45  =             dataset_45[train_size+1: ].values
        test_data_80  =             dataset_80[train_size+1: ].values
        test_data_Temperature  =    dataset_Temperature[train_size+1: ].values
        test_data_Humidity  =       dataset_Humidity[train_size+1: ].values
        test_data_Pressure  =       dataset_Pressure[train_size+1: ].values
        test_data_Lum  =            dataset_Lum[train_size+1: ].values
        test_date_time =           Date_Time1[train_size+1: ]

        print("")
        print("Dataset Splitted into testing and training")
        print("")


        print("##################")
        print("########")
        print("###")
        print("10 m Training mode activated")
        print("###")
        print("#######")
        print("#################")

        history = [x for x in train_data_10]
        predictions = list()
        for t in range(len(test_data_10)):
        	model_10 = ARIMA(history, order=(1,0,6), missing='drop')
        	model_10_fit = model_10.fit(disp=0)
        	output = model_10_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_10[t]
        	history.append(obs)
        	print('10m ',t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_10  = predictions

        error_10 = mean_squared_error(test_data_10 , testPredict_10)
        print('Test MSE: %.3f' % error_10)

        MAE_10= mean_absolute_error(test_data_10 , testPredict_10)
        print('Test MAE: %.3f' % MAE_10)

        rmse_10 = sqrt(mean_squared_error(test_data_10 , testPredict_10))
        print('Test RMSE: %.3f' % rmse_10)

        MAPE_10= mean_absolute_percentage_error(test_data_10 , testPredict_10)
        print("\n\n \tMean absolute percentage error  :",MAPE_10)
        print(".......Accuracy for 80 m  ...", 100-MAPE_10)

        from sklearn.metrics import r2_score
        R2_10 = r2_score(test_data_10 , testPredict_10)
        print("R2 score = ", ((R2_10)*100))



        ################ Soil Moisture at 25 m #########################################

        print("##################")
        print("########")
        print("###")
        print("45 m Training mode activated")
        print("###")
        print("#######")
        print("#################")


        history = [x for x in train_data_45]
        predictions = list()
        for t in range(len(test_data_45)):
        	model_45 = ARIMA(history, order=(0,0,1))
        	model_45_fit = model_45.fit()
        	output = model_45_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_45[t]
        	history.append(obs)
        	print('45',t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_45  = predictions


        #trainPredict_50 = sc.inverse_transform(Y_50)
        #testPredict_50 = sc.inverse_transform(testPredict_50)
        error_45 = mean_squared_error(test_data_45 , testPredict_45)
        print('Test MSE: %.3f' % error_45)

        MAE_45= mean_absolute_error(test_data_45, testPredict_45)
        print('Test MAE: %.3f' % MAE_45)

        rmse_45 = sqrt(mean_squared_error(test_data_45, testPredict_45))
        print('Test RMSE: %.3f' % rmse_45)

        MAPE_45= mean_absolute_percentage_error(test_data_45 , testPredict_45)
        print("\n\n \tMean absolute percentage error  :",MAPE_45)
        print(".......Accuracy for 80 m  ...", 100-MAPE_45)

        R2_45 = r2_score(test_data_45 , testPredict_45)
        print("R2 score = ", ((R2_45)*100))




        ################ Soil Moisture at 80 m #########################################
        print("##################")
        print("########")
        print("###")
        print("80 m Training mode activated")
        print("###")
        print("#######")
        print("#################")


        history = [x for x in train_data_80]
        predictions = list()
        for t in range(len(test_data_80)):
        	model_80 = ARIMA(history, order=(1,0,4))
        	model_80_fit = model_80.fit(disp=0)
        	output = model_80_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_80[t]
        	history.append(obs)
        	print('80',t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_80  = predictions


        #trainPredict_50 = sc.inverse_transform(Y_50)
        #testPredict_50 = sc.inverse_transform(testPredict_50)
        error_80 = mean_squared_error(test_data_80 , testPredict_80)
        print('Test MSE: %.3f' % error_80)

        MAE_80= mean_absolute_error(test_data_80, testPredict_80)
        print('Test MAE: %.3f' % MAE_80)

        rmse_80 = sqrt(mean_squared_error(test_data_80, testPredict_80))
        print('Test RMSE: %.3f' % rmse_80)

        MAPE_80= mean_absolute_percentage_error(test_data_80 , testPredict_80)
        print("\n\n \tMean absolute percentage error  :",MAPE_80)
        print(".......Accuracy for 80 m  ...", 100-MAPE_80)


        from sklearn.metrics import r2_score
        R2_80 = r2_score(test_data_80 , testPredict_80)
        print("R2 score = ", ((R2_80)*100))



        ################ T E M P  E R A T U R E #########################################
        print("##################")
        print("########")
        print("###")
        print("Temp Training mode activated")
        print("###")
        print("#######")
        print("#################")


        history = [x for x in train_data_Temperature]
        predictions = list()
        for t in range(len(test_data_Temperature)):
        	model_Temp = ARIMA(history, order=(1,0,6))
        	model_Temp_fit = model_Temp.fit(disp=0)
        	output = model_Temp_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_Temperature[t]
        	history.append(obs)
        	print('Temperature' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_Temperature  = predictions


        #trainPredict_80 = sc.inverse_transform(Y_80)
        #testPredict_80 = sc.inverse_transform(testPredict_80)


        error_Temperature = mean_squared_error(test_data_Temperature , testPredict_Temperature)
        print('Test MSE: %.3f' % error_Temperature)


        MAE_Temperature = mean_absolute_error(test_data_Temperature, testPredict_Temperature)
        print('Test MAE: %.3f' % MAE_Temperature)


        rmse_Temperature = sqrt(mean_squared_error(test_data_Temperature, testPredict_Temperature))
        print('Test RMSE: %.3f' % rmse_Temperature)

        MAPE_Temperature = mean_absolute_percentage_error(test_data_Temperature , testPredict_Temperature)
        print("\n\n \tMean absolute percentage error  :",MAPE_Temperature)
        print(".......Accuracy for 80 m  ...", 100-MAPE_Temperature)

        R2_Temperarure = r2_score(test_data_Temperature , testPredict_Temperature)
        print("R2 score = ", ((R2_Temperarure)*100))


        ################ Humidity  #########################################


        print("##################")
        print("########")
        print("###")
        print("Humidity Training mode activated")
        print("###")
        print("#######")
        print("#################")

        history = [x for x in train_data_Humidity]
        predictions = list()
        for t in range(len(test_data_Humidity)):
        	model_Humd = ARIMA(history, order=(1,0,6))
        	model_Humd_fit = model_Humd.fit(disp=0)
        	output = model_Humd_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_Humidity[t]
        	history.append(obs)
        	print('Humidity' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_Humidity  = predictions


        #trainPredict_80 = sc.inverse_transform(Y_80)
        #testPredict_80 = sc.inverse_transform(testPredict_80)

        error_Humidity = mean_squared_error(test_data_Humidity , testPredict_Humidity)
        print('Test MSE: %.3f' % error_Humidity)

        MAE_Humidity= mean_absolute_error(test_data_Humidity, testPredict_Humidity)
        print('Test MAE: %.3f' % MAE_Humidity)

        from math import sqrt
        # calculate RMSE
        rmse_Humidity = sqrt(mean_squared_error(test_data_Humidity, testPredict_Humidity))
        print('Test RMSE: %.3f' % rmse_Humidity)

        MAPE_Humidity = mean_absolute_percentage_error(test_data_Humidity, testPredict_Humidity)
        print("\n\n \tMean absolute percentage error  :",MAPE_Humidity)
        print(".......Accuracy for 80 m  ...", 100-MAPE_Humidity)

        R2_Humidity = r2_score(test_data_Humidity , testPredict_Humidity)
        print("R2 score = ", ((R2_Humidity)*100))

        ################ Pressure  #########################################

        print("##################")
        print("########")
        print("###")
        print("Pressure Training mode activated")
        print("###")
        print("#######")
        print("#################")


        history = [x for x in train_data_Pressure]
        predictions = list()
        for t in range(len(test_data_Pressure)):
        	model_Prsr = ARIMA(history, order=(1,0,10))
        	model_Prsr_fit = model_Prsr.fit(disp=0)
        	output = model_Prsr_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_Pressure[t]
        	history.append(obs)
        	print('Pressure' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_Pressure  = predictions


        #trainPredict_80 = sc.inverse_transform(Y_80)
        #testPredict_80 = sc.inverse_transform(testPredict_80)

        error_Pressure = mean_squared_error(test_data_Pressure , testPredict_Pressure)
        print('Test MSE: %.3f' % error_Pressure)

        MAE_Pressure= mean_absolute_error(test_data_Pressure, testPredict_Pressure)
        print('Test MAE: %.3f' % MAE_Pressure)
        # calculate RMSE
        rmse_Pressure = sqrt(mean_squared_error(test_data_Pressure, testPredict_Pressure))
        print('Test RMSE: %.3f' % rmse_Pressure)

        MAPE_Pressure = mean_absolute_percentage_error(test_data_Pressure, testPredict_Pressure)
        print("\n\n \tMean absolute percentage error  :",MAPE_Pressure)
        print(".......Accuracy for 80 m  ...", 100-MAPE_Pressure)

        R2_Pressure = r2_score(test_data_Pressure , testPredict_Pressure)
        print("R2 score = ", ((R2_Pressure)*100))


        ################ Luminisity  #########################################


        history = [x for x in train_data_Lum]
        predictions = list()
        for t in range(len(test_data_Pressure)):
        	model_Lmns = ARIMA(history, order=(1,0,10))
        	model_Lmns_fit = model_Lmns.fit(disp=0)
        	output = model_Lmns_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test_data_Lum[t]
        	history.append(obs)
        	print('Lum' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
        testPredict_Lum  = predictions


        #trainPredict_80 = sc.inverse_transform(Y_80)
        #testPredict_80 = sc.inverse_transform(testPredict_80)

        error_Lum = mean_squared_error(test_data_Lum , testPredict_Lum)
        print('Test MSE: %.3f' % error_Lum)

        MAE_Lum = mean_absolute_error(test_data_Lum, testPredict_Lum)
        print('Test MAE: %.3f' % MAE_Lum)

        # calculate RMSE
        rmse_Lum = sqrt(mean_squared_error(test_data_Lum, testPredict_Lum))
        print('Test RMSE: %.3f' % rmse_Lum)


        MAPE_Lum = mean_absolute_percentage_error(test_data_Lum, testPredict_Lum)
        #print("\n\n \tMean absolute percentage error  :",MAPE_Lum)
        #print(".......Accuracy for 80 m  ...", 100-MAPE_Lum)

        R2_Lums =  r2_score(test_data_Lum , testPredict_Lum)
        print("R2 score = ", ((R2_Lums)*100))




        ###################################

        ############# sAVING THE MODEL
        Name = ['MAE', 'MSE', 'RMSE', 'MAPE']

        # List2
        All_error_10 = [error_10, MAE_10, rmse_10, MAPE_10 ]
        All_error_45 = [error_45, MAE_45, rmse_45, MAPE_45]
        All_error_80 = [error_80, MAE_80, rmse_80, MAPE_80]
        All_error_Temperature = [error_Temperature, MAE_Temperature,
                                 rmse_Temperature, MAPE_Temperature]
        All_error_Humidity = [error_Humidity, MAE_Humidity, rmse_Humidity, MAPE_Humidity]
        All_error_Pressure = [error_Pressure, MAE_Pressure, rmse_Pressure, MAPE_Pressure]
        All_error_Luminisity = [error_Lum, MAE_Lum, rmse_Lum, MAPE_Lum]

        Total_error_rate =     [All_error_10, All_error_45, All_error_80, All_error_Temperature , All_error_Humidity,  All_error_Pressure , All_error_Luminisity ]

        Total_error_rate_DF = pd.DataFrame(Total_error_rate, columns = Name)
        print("Results acquired")


        Total_error_rate_DF.rename(index={0:'Sm_10_cm',
                                          1:'SM_45_cm',
                                          2:'SM_80_cm',
                                          3:'Temperature',
                                          4:'Humidity',
                                          5:'Pressure',
                                          6:'Luminisity'}, inplace=True)

        Total_error_rate_DF.to_csv("ARIMA_Error_rate.csv")

        print("Results stored to CSV")

        print("Uploading to the FTP")


        ########################### P R E D I C TI O N S


        start_index = 1
        start_index_date = test_date_time.iloc[-1]

        n_length = 289

        end_index = n_length + start_index

        import datetime
        start_date = datetime.datetime.today()
        from datetime import timedelta
        a = (timedelta(hours = (n_length-1)))
        end_date = start_date + a


        import pandas as pd


        forecast_index = pd.date_range((start_date), periods= (n_length+1), freq='15 min')



        forecast_10 = model_10_fit.predict(start=start_index, end=end_index)
        forecast_10 = np.floor(forecast_10)
        forecast_10

        forecast_45 = model_45_fit.predict(start=start_index, end=end_index)
        forecast_45 = np.floor(forecast_45)
        forecast_45

        forecast_80 = model_80_fit.predict(start=start_index, end=end_index)
        forecast_80 = np.floor(forecast_80)
        forecast_80

        forecast_Temp = model_Temp_fit.predict(start=start_index, end=end_index)
        forecast_Temp = np.floor(forecast_Temp)
        forecast_Temp

        forecast_Humd = model_Humd_fit.predict(start=start_index, end=end_index)
        forecast_Humd = np.floor(forecast_Humd)
        forecast_Humd

        forecast_Prsr = model_Prsr_fit.predict(start=start_index, end=end_index)
        forecast_Prsr = np.floor(forecast_Prsr)
        forecast_Prsr

        forecast_Lmns = model_Lmns_fit.predict(start=start_index, end=end_index)
        forecast_Lmns = np.floor(forecast_Lmns)
        forecast_Lmns

        Forecasting = pd.DataFrame([forecast_10, forecast_45, forecast_80, forecast_Temp ,
                        forecast_Humd,  forecast_Prsr , forecast_Lmns])

        Forecasting = Forecasting.T
        Forecasting.index = forecast_index

        Forecasting.columns=('Sm_10_cm','SM_45_cm', 'SM_80_cm','Temperature',
                                          'Humidity','Pressure','Luminisity')


        Forecasting.to_csv("ARIMA_Forecast.csv")





        url='ftp.smartfasal.in'
        username = 'sukhwinder@smartfasal.in'
        pwd = 'Thapar@123'
        print("Setting the URL >>  " + url)
        import os
        import ftplib
        ftp = ftplib.FTP(url, username, pwd)
        #files = ftp.dir()
        path_ftp = "/"
        ftp.cwd(path_ftp)
        print("Uploading the file 1")

        # local file name you want to upload
        filename_1 = "ARIMA_Forecast.csv"
        with open(filename_1, "rb") as file:
            # use FTP's STOR command to upload the file
            ftp.storbinary(f"STOR {filename_1}", file)

        print("Uploading the file 2")
        # local file name you want to upload
        filename_2 = "ARIMA_Error_rate.csv"
        with open(filename_2, "rb") as file:
            # use FTP's STOR command to upload the file
            ftp.storbinary(f"STOR {filename_2}", file)

        print("list current files & directories")


        ftp.dir()

        print("quit and close the connection")
        ftp.quit()
        print("P  R O C E S S     ->   c O M P L E T E D - >")















#############################################################################

class upload_readings:
    def allreadings(request):
        return render(request,'plots_of_all_readings.html')

    def smonereadings(request):
        return render(request,'plots_of_smonereadings.html')

    def smtworeadings(request):
        return render(request,'plots_of_smtworeadings.html')

    def smthreereadings(request):
        return render(request,'plots_of_smthreereadings.html')

    def tempreadings(request):
        return render(request,'plots_of_tempreadings.html')

    def humidreadings(request):
        return render(request,'plots_of_humidreadings.html')

    def prsrreadings(request):
        return render(request,'plots_of_prsrreadings.html')

    def lmnsreadings(request):
        return render(request,'plots_of_lmnsreadings.html')
