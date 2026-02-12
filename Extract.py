# exercise 1.5.1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the Iris csv data using the Pandas library
filename = "Project\carz.csv"
df = pd.read_csv(filename)

raw_data = df.values  
rows,cols=df.shape
cols = range(0, cols) 

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])
CarMaker = raw_data[:,0]
Country = raw_data[:,1]

CarMakernames = np.unique(CarMaker)
Countrynames = np.unique(Country)

Carmakerdict = dict(zip(CarMakernames,range(len(CarMakernames))))
Countrydict = dict(zip(Countrynames,range(len(Countrynames))))
for i in range(rows):
    df.loc[i,"Car Make"]= Carmakerdict[df.loc[i,"Car Make"]]
for i in range(rows):
    df.loc[i,"Country "]= Countrydict[df.loc[i,"Country "]]


change_type_cols= df.columns[df.dtypes.eq('object')]
df[change_type_cols] = df[change_type_cols].apply(pd.to_numeric, errors='coerce')

raw_data = df.values  
X = raw_data[:, cols]
print (df.columns)
