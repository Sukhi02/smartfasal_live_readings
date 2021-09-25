\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\print("Accesing the FTP")
import pandas as pd
import numpy as np
import os
#from smartfasal_project.settings import BASE_DIR
#STATIC_ROOT = os.path.join(BASE_DIR, 'static')
#path = STATIC_ROOT
#os.chdir(path)

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
#del date_file
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


print("Converting colon to string")
Prsr = Prsr.astype("|S5")
print(Prsr)
print("Strings to the array of float 64")
Prsr = Prsr.astype(np.float)










Prsr1  = Prsr

arr_of_strings = Prsr1.astype("|S5")
print(arr_of_strings)

Prsr2 = arr_of_strings.astype(np.float)




Prsr  = Prsr

Prsr = Prsr.astype("|S5")
print(Prsr)

Prsr = Prsr.astype(np.float)



index_Prsr = Prsr.index

new_index_Prsr = []

new_index_Prsr = list(range(len(Prsr)))

Prsr.index = new_index_Prsr













x = Prsr.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
vals = [','.join(ele.split()) for ele in x]
print(vals)

vals[0]



vals.isdigit()


Prsr1 = Prsr.to_string()



s='100.:'

type(s)

ss = np.array_str(value[0])
print ss
print type(ss)



ts = Prsr.tostring()
print(np.fromstring(ts, dtype=int))










s=s[:s.index('.')]
s


DataFrame.to_string



vals[0]


Formatted = "{:d}"
print(Formatted.format(7000))


Formatted = "{:,d}"
print(Formatted.format(7000))
Formatted = "{:^15,d}"
print(Formatted.format(7000))
Formatted = "{:*^15,d}"
print(Formatted.format(7000))
Formatted = "{:*^15.2f}"
print(Formatted.format(7000))
Formatted = "{:*>15X}"
print(Formatted.format(7000))
Formatted = "{:*<#15x}"
print(Formatted.format(7000))
Formatted = "A {0} {1} and a {0} {2}."
print(Formatted.format("blue", "car", "truck"))





















Prsr

index_Prsr = Prsr.index

new_index_Prsr = []

new_index_Prsr = list(range(len(Prsr)))

Prsr.index = new_index_Prsr

for i in range(len(Prsr)):
    
    print( Prsr[i])
    

value

value.isdigit()
















floatAsStr = "2:5"
floatAsStr = floatAsStr.replace(",", ".");
myFloat = float(floatAsStr)







value = value(del l[-1])
    
Prsr = Prsr.values.tolist()
value = Prsr[0]

value


floatAsStr = "2:5"
floatAsStr = floatAsStr.replace(",", ".");
myFloat = float(floatAsStr)





DecimalFormat_df = new DecimalFormat("#.##")
amount = Double.valueOf(df.format(amount));
String newFormat = String.valueOf(amount).replace('.',':');



