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


dataset = pd.read_csv('Dos.csv')
dataset2 = pd.read_csv('Intrusion.csv')
dataset3 = pd.read_csv('MitM.csv')
Data = pd.concat([dataset,dataset2,dataset3])

#drop columns with all null
columns_allnull = [Data.columns[3], Data.columns[12],Data.columns[13],Data.columns[14],Data.columns[15],
                   Data.columns[17],Data.columns[19],Data.columns[20],Data.columns[21],Data.columns[23],
                   Data.columns[25],Data.columns[49],Data.columns[50],Data.columns[59],Data.columns[60],
                   Data.columns[62],Data.columns[63], Data.columns[64], Data.columns[65]]
				   
				
Data.drop(columns = columns_allnull, inplace = True)

#drop unnecessary columns
columns_to_drop = ['ip.src','ip.dst','eth.src','eth.dst','mqtt.clientid','mqtt.topic','mqtt.msg','mqtt.topic_len','frame.encap_type','frame.ignored','frame.marked','frame.offset_shift']
Data.drop(columns = columns_to_drop, inplace = True)

replace_conack= {'mqtt.conack.flags':
 {

    '0x00000000':0,
    '0x00000061':97,
    '0x00000036':54,
    '0x0000006f':111,
    '0x00000065':101, 
    '0x00000030':48, 
    '0x00000069':105, 
    '0x00000034':52,
    '0x00000035':53
 }
}


replace_conflags = {'mqtt.conflags':
 {
    '0x00000002':2,
 }
}


replace_hdrflags = {'mqtt.hdrflags':
 {

       '0x000000c0':192, '0x000000d0':208, '0x00000030':48, '0x00000010':16,
       '0x00000020':32, '0x00000001':1, '0x00000035':53, '0x00000066':102,
       '0x00000046':70, '0x000000e0':224, '0x00000040':64, '0x00000036':54,
       '0x00000033':51, '0x0000002d':45, '0x00000061':97, '0x0000002f':47,
       '0x00000065':101, '0x00000074':116, '0x00000043':67, '0x00000044':68,
       '0x0000006d':109, '0x00000045':69, '0x00000072':114, '0x00000038':56,
       '0x00000062':98, '0x00000064':100, '0x0000006c':108, '0x00000063':99,
       '0x00000031':49, '0x00000082':130, '0x00000090':144, '0x0000006f':111,
       '0x00000039':57, '0x00000032':50, '0x00000037':55
 }
}


replace_type= {'type':
 {
    'normal':0,
     'DoS':1,
     'intrusion':2,
     'mitm':3
 }
}


replace_protoname = {'mqtt.protoname':
 {
    'MQTT':1, 
     'MQIsdp':2
 }
}

Data.replace(replace_hdrflags,inplace = True)
Data.replace(replace_protoname,inplace = True)
Data.replace(replace_type,inplace = True)
Data.replace(replace_conack,inplace = True)
Data.replace(replace_conflags,inplace = True)

Data = Data.fillna(-1)
Data.to_csv('dataset.csv')













