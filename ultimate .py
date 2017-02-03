
# coding: utf-8

# In[3]:

import xlrd 
file_location = "C:\Users\Kowshik\Desktop\sand prediction and paper\data\country codes .xlsx"
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)
countrycodes = {}
countryrevcodes = {}(sheet.nrows):
    countrycodes[sheet.cell_value(i,0)] = sheet.cell_value(i,1)
for j in range(sheet.nrows):
    countryrevcodes[sheet.cell_value(j,1)] =  sheet.cell_value(j,0)
    
import numpy as np
import xlrd 
file_location = "C:\Users\Kowshik\Desktop\sand prediction and paper\precise data\Finaldatasheet.xlsx"
workbook_1 = xlrd.open_workbook(file_location)
sheet_1 = workbook_1.sheet_by_index(0)
countries = []
for i in range(36):
    countries.append(sheet_1.cell_value(i+1,0))
     

for i in range
ccode = []
for i in range(36):
    ccode.append(countrycodes[countries[i]])
    
    
import xlrd 
file_location = "C:\Users\Kowshik\Desktop\sand prediction and paper\precise data\Finaldatasheet.xlsx"
workbook = xlrd.open_workbook(file_location)

#CEMENT 
cement_data = workbook.sheet_by_index(0)
cement = {}
for i in range(36):
    cement[countrycodes[cement_data.cell_value(i+1,0)]] = {}
    for j in range(19):
        cement[countrycodes[cement_data.cell_value(i+1,0)]][1996+j] = cement_data.cell_value(i+1,19-j)
        
    
#GDP CONSTRUCTION
GDP_data = workbook.sheet_by_index(1)
GDP_CONS= {}

for i in range(GDP_data.nrows-1):
    if GDP_data.cell_value(i+1,1) in ccode:
        GDP_CONS[GDP_data.cell_value(i+1,1)] = {}
        for j in range(19):
            GDP_CONS[GDP_data.cell_value(i+1,1)][1996+j] = GDP_data.cell_value(i+1,j+2)
#GDP GROWTH
GDPG_data = workbook.sheet_by_index(2)
GDP_GROWTH= {}

for i in range(GDPG_data.nrows-1):
    if GDPG_data.cell_value(i+1,1) in ccode:
        GDP_GROWTH[GDPG_data.cell_value(i+1,1)] = {}
        for j in range(19):
            GDP_GROWTH[GDPG_data.cell_value(i+1,1)][1996+j] = GDPG_data.cell_value(i+1,j+2)
#URBAN PERSENTAGE
URBAN_PER_data =  workbook.sheet_by_index(3)
URBAN_PER = {}

for i in range(URBAN_PER_data.nrows-1):
    if URBAN_PER_data.cell_value(i+1,1) in ccode:
        URBAN_PER[URBAN_PER_data.cell_value(i+1,1)] = {}
        for j in range(19):
            URBAN_PER[URBAN_PER_data.cell_value(i+1,1)][1996+j] = URBAN_PER_data.cell_value(i+1,j+2)
#URBAN POPULATION 
URBAN_POP_data =  workbook.sheet_by_index(4)
URBAN_POP = {}

for i in range(URBAN_POP_data.nrows-1):
    if URBAN_POP_data.cell_value(i+1,1) in ccode:
        URBAN_POP[URBAN_POP_data.cell_value(i+1,1)] = {}
        for j in range(19):
            URBAN_POP[URBAN_POP_data.cell_value(i+1,1)][1996+j] = URBAN_POP_data.cell_value(i+1,j+2)
#URBAN POPULATION GROWTH
URBAN_POPG_data =  workbook.sheet_by_index(5)
URBAN_POPG = {}

for i in range(URBAN_POPG_data.nrows-1):
    if URBAN_POPG_data.cell_value(i+1,1) in ccode:
        URBAN_POPG[URBAN_POPG_data.cell_value(i+1,1)] = {}
        for j in range(19):
            URBAN_POPG[URBAN_POPG_data.cell_value(i+1,1)][1996+j] = URBAN_POPG_data.cell_value(i+1,j+2)
#POPULATIONGROWTH
POPUL_GROWTH_data =  workbook.sheet_by_index(6)
POPUL_GROWTH= {}

for i in range(POPUL_GROWTH_data.nrows-1):
    if POPUL_GROWTH_data.cell_value(i+1,1) in ccode:
        POPUL_GROWTH[POPUL_GROWTH_data.cell_value(i+1,1)] = {}
        for j in range(19):
            POPUL_GROWTH[POPUL_GROWTH_data.cell_value(i+1,1)][1996+j] = POPUL_GROWTH_data.cell_value(i+1,j+2)
#Energy compsumtion
ENER_COMP_data =  workbook.sheet_by_index(7)
ENER_COMP = {}

for i in range(ENER_COMP_data.nrows-1):
    if ENER_COMP_data.cell_value(i+1,1) in ccode:
        ENER_COMP[ENER_COMP_data.cell_value(i+1,1)] = {}
        for j in range(19):
            ENER_COMP[ENER_COMP_data.cell_value(i+1,1)][1996+j] = ENER_COMP_data.cell_value(i+1,j+2)

            
            
#SIZE
SIZE_data =  workbook.sheet_by_index(8)
SIZE= {}

for i in range(SIZE_data.nrows-1):
    if SIZE_data.cell_value(i+1,1) in ccode:
        SIZE[SIZE_data.cell_value(i+1,1)] = {}
        for j in range(19):
            SIZE[SIZE_data.cell_value(i+1,1)][1996+j] = ENER_COMP_data.cell_value(i+1,2)

            
            
            
#POPULATION
POPUL_data =  workbook.sheet_by_index(9)
POPUL= {}

for i in range(POPUL_data.nrows-1):
    if POPUL_data.cell_value(i+1,1) in ccode:
        POPUL[POPUL_data.cell_value(i+1,1)] = {}
        for j in range(19):
            POPUL[POPUL_data.cell_value(i+1,1)][1996+j] = POPUL_data.cell_value(i+1,j+2)

            
#SAND
cement_data = workbook.sheet_by_index(0)
sand = {}

for i in range(36):
    sand[countrycodes[cement_data.cell_value(i+1,0)]] = {}
    for j in range(19):
        sand[countrycodes[cement_data.cell_value(i+1,0)]][1996+j] = 2*cement_data.cell_value(i+1,19-j)

#  cement[i][1996+j],GDP_CONS[i][1996+j],GDP_GROWTH[i][1996+j],URBAN_PER[i][1996+j],URBAN_POP[i][1996+j],
# URBAN_POPG[i][1996+j],ENER_COMP[i][1996+j],SIZE[i][1996+j],POPUL[i][1996+j],sand[i][1996+j]


data = {}   #STORING MY all DATA country wise and yearwise in the above order 
for i in ccode:
    data[i] = {}
    for j in range(17):
        data[i][1996+j] = [cement[i][1996+j],GDP_CONS[i][1996+j],GDP_GROWTH[i][1996+j],URBAN_PER[i][1996+j],URBAN_POP[i][1996+j],
                          URBAN_POPG[i][1996+j],ENER_COMP[i][1996+j],SIZE[i][1996+j],POPUL[i][1996+j],sand[i][1996+j] ]
data_2 = {}

for i in ccode:
    data_2[i] = {}
    for j in range(17):
        data_2[i][1996+j] = [GDP_CONS[i][1996+j],GDP_GROWTH[i][1996+j],URBAN_PER[i][1996+j],URBAN_POP[i][1996+j],
                          URBAN_POPG[i][1996+j],ENER_COMP[i][1996+j],SIZE[i][1996+j],POPUL[i][1996+j]]

x = []  #Saving my data in a label form 
y = []

for i in ccode:
    for j in range(17):
        x.append(data_2[i][1996+j])
        y.append(sand[i][1996+j])
    

import numpy as np
Y_CON = np.array(y)
X_CON = np.array(x)

##########################################################################
#############################################################
#####################################Country wise top 14
#India 
x_IND = []
y_IND = []


for j in range(17):
    x_IND.append(data_2['IND'][1996+j])
    y_IND.append(sand['IND'][1996+j])
    
    
Y_IND = np.array(y_IND)
X_IND = np.array(x_IND)


from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_IND,Y_IND)

feature_IND = reg.feature_importances_*100

factor_importance_IND =  [ round(elem, 2) for elem in feature_IND ]

IND_2015 = [33,7.9,32,419938867,5.61,810,2973190,1311050527]
IND_2016 = [34,8,32.4,439802441,1.5,850,2973190,1326801576]
#CHINA
x_CHN = []
y_CHN = []
for j in range(17):
    x_CHN.append(data_2['CHN'][1996+j])
    y_CHN.append(sand['CHN'][1996+j])
    
    
Y_CHN = np.array(y_CHN)
X_CHN = np.array(x_CHN)

from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_CHN,Y_CHN)
feature_CHN = reg.feature_importances_*100
factor_importance_CHN =  [ round(elem, 2) for elem in feature_CHN ]
factor_importance_CHN


CHN_2015 = [50.5,7,56.6,779478624,4.4,3741,9388230,1376048943]
CHN_2016 = [52,6.9,57.9,799964410,2.45,3802,9388230,1382323332]
#IRAN
x_IRN = []
y_IRN = []


for j in range(17):
    x_IRN.append(data_2['IRN'][1996+j])
    y_IRN.append(sand['IRN'][1996+j])
    
    
Y_IRN = np.array(y_CHN)
X_IRN = np.array(x_CHN)
#TURKEY
x_TUR = []
y_TUR = []


for j in range(17):
    x_TUR.append(data_2['TUR'][1996+j])
    y_TUR.append(sand['TUR'][1996+j])
    
    
Y_TUR = np.array(y_CHN)
X_TUR = np.array(x_CHN)
#VIETNAM
x_VNM = []
y_VNM = []


for j in range(17):
    x_TUR.append(data_2['VNM'][1996+j])
    y_TUR.append(sand['VNM'][1996+j])
Y_VNM = np.array(y_VNM)
X_VNM = np.array(x_VNM)


#RUSSIA
x_RUS = []
y_RUS = []


for j in range(17):
    x_RUS.append(data_2['RUS'][1996+j])
    y_RUS.append(sand['RUS'][1996+j])
    
    
Y_RUS = np.array(y_RUS)
X_RUS = np.array(x_RUS)

from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_RUS,Y_RUS)

feature_RUS = reg.feature_importances_*100
factor_importance_RUS =  [ round(elem, 2) for elem in feature_RUS ]
factor_importance_RUS


RUS_2015 = [41.1,-2.7,73.3,105163529,0.18,6900,16377960,143456918]
RUS_2016 = [43.3,0.7,73.2,105827920,-0.14,7200,16377960,143439832]

#Japan
x_JPN = []
y_JPN  = []


for j in range(17):
    x_JPN.append(data_2['JPN'][1996+j])
    y_JPN.append(sand['JPN'][1996+j])
    
    
Y_JPN  = np.array(y_JPN )
X_JPN  = np.array(x_JPN )


from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)

reg.fit(X_JPN ,Y_JPN )
feature_JPN  = reg.feature_importances_*100
factor_importance_JPN  =  [ round(elem, 2) for elem in feature_JPN  ]
factor_importance_JPN 

JPN_2015 = [27.2,1.1,93.7,118572468,3.54,7900,364500,126573481]
JPN_2016 = [28,1.7,94.1,118911665,0.42,8100,364500,126323715]
#saudhi Arabia
x_SAU = []
y_SAU  = []


for j in range(17):
    x_SAU.append(data_2['SAU'][1996+j])
    y_SAU.append(sand['SAU'][1996+j])
    
    
Y_SAU  = np.array(y_SAU )
X_SAU  = np.array(x_SAU )

#THailand
x_THA = []
y_THA  = []


for j in range(17):
    x_THA.append(data_2['THA'][1996+j])
    y_THA.append(sand['THA'][1996+j])
    
    
Y_THA  = np.array(y_THA )
X_THA  = np.array(x_THA )


from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)


reg.fit(X_THA ,Y_THA )
feature_THA = reg.feature_importances_*100
factor_importance_THA =  [ round(elem, 2) for elem in feature_THA  ]
factor_importance_THA


THA_2015 = [38,3.5,50,33952234,5.89,2800,510890,67959359]
THA_2016 = [39.2,4,51.1,34810313,2.2,3000,510890,68146609]
#Germany
x_DEU = []
y_DEU  = []


for j in range(17):
    x_DEU.append(data_2['DEU'][1996+j])
    y_DEU.append(sand['DEU'][1996+j])
    
    
Y_DEU  = np.array(y_DEU )
X_DEU  = np.array(x_DEU )


from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)


reg.fit(X_DEU ,Y_DEU )
feature_DEU  = reg.feature_importances_*100
factor_importance_DEU  =  [ round(elem, 2) for elem in feature_DEU  ]
factor_importance_DEU


DEU_2015 = [36.97,1.552,77 ,62170091,0.4,7375,349090,80688545]
DEU_2016 = [37.1,1.81,77.2 ,62260626,0.25,7440,349090,80682351]
#ITaly
x_ITA = []
y_ITA  = []


for j in range(17):
    x_ITA.append(data_2['ITA'][1996+j])
    y_ITA.append(sand['ITA'][1996+j])
    
    
Y_ITA  = np.array(y_ITA )
X_ITA  = np.array(x_ITA )


from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_ITA ,Y_ITA )


feature_ITA  = reg.feature_importances_*100
factor_importance_ITA  =  [ round(elem, 2) for elem in feature_ITA  ]
factor_importance_ITA


ITA_2015 = [22.7,0.68,70.5 ,42166069,0.78,5500,294110,59797685]
ITA_2016 = [22.1,0.73,70.7 ,42306608,0.28,5600,294110,59801004]

#Pakistan
x_PAK = []
y_PAK  = []
for j in range(17):
    x_PAK.append(data_2['PAK'][1996+j])
    y_PAK.append(sand['PAK'][1996+j])
    
    
Y_PAK  = np.array(y_PAK )
X_PAK  = np.array(x_PAK )
#INDonesia
x_IDN = []
y_IDN  = []
for j in range(17):
    x_IDN.append(data_2['IDN'][1996+j])
    y_IDN.append(sand['IDN'][1996+j])
    
    
Y_IDN  = np.array(y_IDN )
X_IDN  = np.array(x_IDN )


from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_IDN ,Y_IDN )


feature_IDN = reg.feature_importances_*100
factor_importance_IDN =  [ round(elem, 2) for elem in feature_IDN  ]
factor_importance_IDN


IDN_2015 = [42.2,4.7,53.4 ,137422002,4.45,780,1811570,257563815]
IDN_2016 = [42.4,5.5,54 ,140824151,1.12,810,1811570,260581100]
###################################################################### Feature importance of each 
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)

reg.fit(X_CON,Y_CON)

feature = reg.feature_importances_*100

factor_importance =  [ round(elem, 2) for elem in feature ]
#############################################################################Feature importance Graphing 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance

colors = ['#b4c9c0', '#326163', '#ba8c00', '#756287','#db7d69','#1f2358','#c55073','#9a456b']

explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
 # Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)

plt.axis('equal')

plt.show() #------------------------------------------------------------------------------------------> Pie plot Asia and Europe 
#################################################################################
##################################################################################Country wise 
################################################################################## 
#1) CHINA 
## Factor Importance 

import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_CHN


colors = ['#8ce794', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)

plt.show()           #------------------------------------------------------>plot china
##Sand predictions 

sand['CHN'][2015] = 4559824.98660055
#sand['CHN'][2016] = 4647231.96256603
## SUBSTITUTES

country_alarm_CHN = 500#--------------------------------------> The extra dredging which concerns government 
Year = 2011              #---------------------------------------> The Year of alarm
y_ = Year - 1996

y_CHN_1 = [[y_CHN[i]] for i in range(0, len(y_CHN), 1)]
lis = np.arange(17)

lis[0:y_] = 0
lis[y_:] = 1

from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt

X = np.array(y_CHN_1)
Y = lis
clf = RidgeClassifier().fit(X,Y)

for j in range(17):
    if clf.predict(y_CHN[j]) == 0 and clf.predict(y_CHN[j+1]) ==1:
        for i in range(int(y_CHN[j]/1000),int(y_CHN[j+1]/1000)):
            if  clf.predict([i*1000]) == 0 and clf.predict([(i+1)*1000]) == 1:
                CW_CHN =  i*1000
                
                
TH_CHN = CW_CHN - country_alarm_CHN

## VISUALS
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['CHN'].values() + [4647231.96256603]

canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)
sp1.patch.set_facecolor('white') #------> background colour


 
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour



sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')



sp1.set_title('China Sand Consumptions', color='Black')

sp1.set_xlabel('Year', color='black')      #Colour of title and lables 

sp1.set_ylabel('Sand consumption,Thousand tons', color='black')

plt.plot(x, y)

plt.tight_layout()

plt.axhline( y = TH_CHN )

plt.show()
######################################
#1) INDIA
## Factor Importance 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_IND

colors = ['#8ce794', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)


plt.show()           #------------------------------------------------------>plot IND

## SUBSTITUTES
country_alarm_IND = 50#--------------------------------------> The extra dredging which concerns government 
Year = 2011             #---------------------------------------> The Year of alarm
y_ = Year - 1996


y_IND_1 = [[y_IND[i]] for i in range(0, len(y_IND), 1)]


lis = np.arange(17)
lis[0:y_] = 0
lis[y_:] = 1


from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt

X = np.array(y_IND_1)
Y = lis


clf = RidgeClassifier().fit(X,Y)

for j in range(17):
    if clf.predict(y_IND[j]) == 0 and clf.predict(y_IND[j+1]) ==1:
        for i in range(int(y_IND[j]/1000),int(y_IND[j+1]/1000)):
            if  clf.predict([i*1000]) == 0 and clf.predict([(i+1)*1000]) == 1:
                CW_IND =  i*1000
                
                
TH_IND = CW_IND - country_alarm_IND
##Sand predictions 
#sand['IND'][2015] = 928726.37442971
#sand['IND'][2016] =  960376.44747593
## VISUALS


import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['IND'].values() +  [928726.37442971,960376.44747593]


canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')


sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)
sp1.patch.set_facecolor('white') #------> background colour


 
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour



sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')



sp1.set_title('India Sand Consumptions', color='Black')

sp1.set_xlabel('Year', color='black')      #Colour of title and lables

sp1.set_ylabel('Sand consumption,Thousand tons', color='black')

plt.plot(x, y)

plt.tight_layout()

plt.axhline( y = TH_IND )

plt.show()
####################################################################3
#1) Russia
## Factor Importance 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_RUS
colors = ['#8ce794', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)
plt.show()           #------------------------------------------------------

## SUBSTITUTES
country_alarm_RUS = 50#--------------------------------------> The extra dredging which concerns government 
Year = 2009             #---------------------------------------> The Year of alarm
y_ = Year - 1996
y_RUS_1 = [[y_RUS[i]] for i in range(0, len(y_RUS), 1)]

lis = np.arange(17)

lis[0:y_] = 0
lis[y_:] = 1


from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt

X = np.array(y_RUS_1)
Y = lis
clf = RidgeClassifier().fit(X,Y)


for j in range(17):
    if clf.predict(y_RUS[j]) == 0 and clf.predict(y_RUS[j+1]) ==1:
        for i in range(int(y_RUS[j]/1000),int(y_RUS[j+1]/1000)):
            if  clf.predict([i*1000]) == 0 and clf.predict([(i+1)*1000]) == 1:
                CW_RUS =  i*1000
                
                
TH_RUS = CW_RUS - country_alarm_RUS
##Sand predictions 
#sand['RUS'][2015] = 112372.25090733
#sand['RUS'][2016] =  104068.40102278
## VISUALS


import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['RUS'].values() +  [112372.25090733,104068.40102278]


canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)


sp1.patch.set_facecolor('white') #------> background colour
 
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour



sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')



sp1.set_title('Russia Sand Consumptions', color='Black')

sp1.set_xlabel('Year', color='black')      #Colour of title and lables

sp1.set_ylabel('Sand consumption,Thousand tons', color='black')

plt.plot(x, y)

plt.tight_layout()

plt.axhline( y = TH_RUS)
plt.show()
#########################################################
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_CON,Y_CON)
reg.predict(RUS_2015)
#######################################
#1) Japan
## Factor Importance 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_JPN
colors = ['#8ce794', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)
plt.show()           #------------------------------------------------------

##Sand predictions 
#sand['JPN'][2015] = 124558.61173253
#sand['JPN'][2016] =  116584.99758184
## VISUALS
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['JPN'].values() +  [124558.61173253,116584.99758184]
canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)
sp1.patch.set_facecolor('white') #------> background colour
 
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour

sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')

sp1.set_title('Japan Sand Consumptions', color='Black')
sp1.set_xlabel('Year', color='black')      #Colour of title and lables 
sp1.set_ylabel('Sand consumption,Thousand tons', color='black')
plt.plot(x, y)
plt.tight_layout()
plt.show()
##################################
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1200)
reg.fit(X_CON,Y_CON)
reg.predict(JPN_2016)
##################################################
#1) Thailand
## Factor Importance 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_THA
colors = ['#8ce794', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)
plt.show()           #------------------------------------------------------

## SUBSTITUTES
country_alarm_THA = 50#--------------------------------------> The extra dredging which concerns government 
Year = 2008             #---------------------------------------> The Year of alarm
y_ = Year - 1996
y_THA_1 = [[y_THA[i]] for i in range(0, len(y_THA), 1)]
lis = np.arange(17)
lis[0:y_] = 0
lis[y_:] = 1
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt

X = np.array(y_THA_1)
Y = lis
clf = RidgeClassifier().fit(X,Y)

for j in range(17):
    if clf.predict(y_THA[j]) == 0 and clf.predict(y_THA[j+1]) ==1:
        for i in range(int(y_THA[j]/1000),int(y_THA[j+1]/1000)):
            if  clf.predict([i*1000]) == 0 and clf.predict([(i+1)*1000]) == 1:
                CW_THA =  i*1000
TH_THA = CW_THA - country_alarm_THA
##Sand predictions 
#sand['THA'][2015] =  85788.22147293
#sand['THA'][2016] =   85788.22147293
## VISUALS
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['THA'].values() +  [ 85788.22147293, 85788.22147293]
canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)
sp1.patch.set_facecolor('white') #------> background colour
 
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour

sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')

sp1.set_title('Thailand Sand Consumptions', color='Black')
sp1.set_xlabel('Year', color='black')      #Colour of title and lables 
sp1.set_ylabel('Sand consumption,Thousand tons', color='black')
plt.plot(x, y)
plt.tight_layout()
plt.axhline( y = TH_THA)
plt.show()
###############################################
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=40)
reg.fit(X_CON,Y_CON)
reg.predict(THA_2016)
###############################################################
#1) Germany
## Factor Importance 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_DEU
colors = ['#8ce794', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)
plt.show()           #------------------------------------------------------

## SUBSTITUTES
country_alarm_DEU = 50#--------------------------------------> The extra dredging which concerns government 
Year = 2001            #---------------------------------------> The Year of alarm
y_ = Year - 1996
y_DEU_1 = [[y_DEU[i]] for i in range(0, len(y_DEU), 1)]
lis = np.arange(17)
lis[0:y_] = 0
lis[y_:] = 1
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt

X = np.array(y_DEU_1)
Y = lis
clf = RidgeClassifier().fit(X,Y)

for j in range(16):
    if clf.predict(y_DEU[j]) == 0 and clf.predict(y_DEU[j+1]) ==1:
        for i in range(int(y_DEU[j]/1000),int(y_DEU[j+1]/1000),-1):
            if  clf.predict([i*1000]) == 1 and clf.predict([(i+1)*1000]) == 0:
                CW_DEU = i*1000   

TH_DEU = CW_DEU - country_alarm_DEU
##Sand predictions 
#sand['THA'][2015] =  62977.93434871
#sand['THA'][2016] =   59643.39369492
## VISUALS
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['DEU'].values() +  [ 62977.93434871,59643.39369492]
canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)
sp1.patch.set_facecolor('white') #------> background colour
 
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour

sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')

sp1.set_title('Germany Sand Consumptions', color='Black')
sp1.set_xlabel('Year', color='black')      #Colour of title and lables 
sp1.set_ylabel('Sand consumption,Thousand tons', color='black')
plt.plot(x, y)
plt.tight_layout()
plt.axhline( y = TH_DEU)
plt.show()
################################3
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(learning_rate=0.01,n_estimators=400)
reg.fit(X_CON,Y_CON)
reg.predict(DEU_2016)
############################################
#1) Italy 
## Factor Importance 
import matplotlib.pyplot as plt
labels = 'GDP from construction ', 'GDP growth', 'Urban Population Percentage ', 'Urban population','Urban population Growth','Energy consumption','SIZE of country','Population '
sizes = factor_importance_ITA
colors = ['#8ce794
import matplotlib.pyplot as plt
x = np.arange(1996, 2017, 1)
y = sand['ITA'].values() + [42638.8137616,44347.08015294]


canvas = plt.figure()

rect = canvas.patch
rect.set_facecolor('white')
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
sp1.plot(x, y, 'red', linewidth=2)
sp1.patch.set_facecolor('white') #------> background colour
 
    
    
sp1.tick_params(axis='x', colors='black') #---------> X-value colour
sp1.tick_params(axis='y', colors='black') #----------> y-value colour

', '#7bbca8', '#7081a2', '#82548f','#884d4d','#1f2358','#c55073','#9a456b']
explode = (0.01, 0, 0, 0.1,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=60)
plt.show()           #------------------------------------------------------
##Sand predictions 
#sand['ITA'][2015] = 42638.8137616
#sand['ITA'][2016] = 44347.08015294
## SUBSTITUTES
## VISUALS
import numpy as np
sp1.spines['bottom'].set_color('black')        #Colour of spines 
sp1.spines['top'].set_visible('False')
sp1.spines['left'].set_color('black')
sp1.spines['right'].set_visible('False')

sp1.set_title('Italy Sand Consumptions', color='Black')
sp1.set_xlabel('Year', color='black')      #Colour of title and lables 
sp1.set_ylabel('Sand consumption,Thousand tons', color='black')

plt.plot(x, y)

plt.tight_layout()

plt.show()



# In[ ]:



