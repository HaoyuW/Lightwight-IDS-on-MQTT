import pandas as pd
import numpy as np
import scipy as sp
import os
import argparse
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from scipy import stats, optimize, interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type = int, default = 0)
args = parser.parse_args()

dataset = pd.read_csv('dataset.csv')
dataset.drop(columns = dataset.columns[0], inplace = True)

#drop 24 columns in total
Thresh_5 =[dataset.columns[1],dataset.columns[3],dataset.columns[5],dataset.columns[7],dataset.columns[11],dataset.columns[12],dataset.columns[13],
             dataset.columns[14],dataset.columns[15],dataset.columns[16],dataset.columns[17],dataset.columns[18],dataset.columns[19],dataset.columns[20],
             dataset.columns[21],dataset.columns[23],dataset.columns[24],dataset.columns[26],dataset.columns[27],dataset.columns[28],dataset.columns[29],
             dataset.columns[30],dataset.columns[31],dataset.columns[34],]
			 
#drop 20 columns in total			 
Thresh_7 =[dataset.columns[1],dataset.columns[8],dataset.columns[11],dataset.columns[12],dataset.columns[13],dataset.columns[14],dataset.columns[15],
           dataset.columns[16],dataset.columns[17],dataset.columns[18],dataset.columns[19],dataset.columns[20],dataset.columns[21],dataset.columns[24],
           dataset.columns[27],dataset.columns[28],dataset.columns[29],dataset.columns[30],dataset.columns[31],dataset.columns[34],]

#drop 18 columns in total			 
Thresh_9 =[dataset.columns[1],dataset.columns[8],dataset.columns[12],dataset.columns[13],dataset.columns[14],dataset.columns[15],
           dataset.columns[16],dataset.columns[17],dataset.columns[18],dataset.columns[19],dataset.columns[20],dataset.columns[21],
           dataset.columns[27],dataset.columns[28],dataset.columns[29],dataset.columns[30],dataset.columns[31],dataset.columns[34],]



if args.threshold == 1:
     print('drop packet with threshold 0.5 ')
     dataset.drop(columns = Thresh_5, inplace = True)
     output = '0_5_result.csv'
elif args.threshold == 2:
     print('drop packet with threshold 0.7 ')
     dataset.drop(columns = Thresh_7, inplace = True)
     output = '0_7_result.csv'
elif args.threshold == 3:
     print('drop packet with threshold 0.9 ')
     dataset.drop(columns = Thresh_9, inplace = True)
     output = '0_9_result.csv'
elif args.threshold == 0:
     print('Without feature selection') 
     output = 'result.csv'

   
X =  dataset.drop(['type'], axis=1)
Y = dataset['type']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

print('start scaling')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
cols = X.columns
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


print('start training')

#classifier = LinearSVC()

classifier = SVC(kernel = 'rbf', C =100, gamma=15)

classifier.fit(X_train,Y_train)

pred = classifier.predict(X_test)


pd.DataFrame(classification_report(Y_test, pred, output_dict = True)).transpose().to_csv(output)

print('Model accuracy score with rbf kernel  : {0:0.4f}'. format(accuracy_score(Y_test, pred)))