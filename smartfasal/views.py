from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_deny
from django.views.decorators.clickjacking import xframe_options_sameorigin
#from .models import FTP_session_model
import pandas as pd
import numpy as np
import os


from django.views.decorators.clickjacking import xframe_options_exempt

@xframe_options_exempt
def ok_to_load_in_a_frame(request):
    return HttpResponse("This page is safe to load in a frame on any site.")



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
        filename = 'S_AgriA.csv'
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

        data_file = pd.read_csv("S_AgriA.csv")

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
        fig.add_trace(go.Scatter(x=x_Time, y=y0, mode='lines', name='SM - 10 cm',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))
        print("Plot_1_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y1, mode='lines', name='SM - 45 cm',
                                marker = dict(color = 'rgba(0,255,0,0.8)')))
        print("Plot_2_ok")
        #fig.add_trace(go.Scatter(x=x_Time, y=y2, mode='lines', name='SM - 80 cm',
        #                        marker = dict(color = 'rgba(26, 102, 255,0.8)')))
        #print("Plot_3_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y3, mode='lines', name='Temperature',
                                marker = dict(color = 'rgba(204, 0, 204, 0.8)')))
        print("Plot_4_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y4, mode='lines', name='Humidity',
                                marker = dict(color = 'rgba(0, 153, 51, 0.8)')))
        print("Plot_5_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y5, mode='lines', name='Pressure',
                                marker = dict(color = 'rgba(0, 0, 204, 0.8)')))
        print("Plot_6_ok")
        fig.add_trace(go.Scatter(x=x_Time, y=y6, mode='lines', name='Luminosity',
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
