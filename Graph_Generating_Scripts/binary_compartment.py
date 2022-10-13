"""
Author: Rekkab Gill
Date: October 7 2022
Details: Program generates a chart to visualize 2 sets of discretized chromosomal compartments
"""


import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
import sys 

file_path1 = sys.argv[1]    #first argument passed
file_path2 = sys.argv[2]    #second argument
cell_name1 = sys.argv[3]    #third
cell_name2 = sys.argv[4]    #fourth

dataframe1 = pd.read_csv(file_path1, sep = "\t", header = None, comment = '#')
dataframe1.columns = ['chromosome', 'start', 'end', 'datavalue']
dataframe1_chr = dataframe1[dataframe1['chromosome'] == 'chr1'].copy() #the copy is necessary to prevent the settingwithcopy warning (it flags confusing chained assignments. i.e. are you referencing original or new dataframe)
dataframe2 = pd.read_csv(file_path2, sep = "\t", header = None, comment = '#')
dataframe2.columns = ['chromosome', 'start', 'end', 'datavalue']
dataframe2_chr = dataframe2[dataframe2['chromosome'] == 'chr1'].copy()


#CHANGE THE START AND END CHROM
dataframe1_chr['start'] = dataframe1_chr['start'] + 1 #we do this because Chris says genome units start at 1 not 0, so push everyting up 1
dataframe1_chr['end'] = dataframe1_chr['end'] + 1
dataframe2_chr['start'] = dataframe2_chr['start'] + 1
dataframe2_chr['end'] = dataframe2_chr['end'] + 1

#Below are the integer indices of the headers in the dataframe (i.e. START is where the chromosome bin starts, END is where it ends, etc.)
START_CHROM_COLUMN = 1
END_CHROM_COLUMN = 2
BINARY_COLUMN = 3
index = 0
difference_list = []
the_diff = 0

#replace the NAN values with 0
dataframe1_chr['datavalue'] = dataframe1_chr['datavalue'].fillna(0)
dataframe2_chr['datavalue'] = dataframe2_chr['datavalue'].fillna(0)

#here we print some values to see the difference between the two dataframes
print(dataframe1_chr.iloc[0,1], dataframe2_chr.iloc[0,1])
print(dataframe1_chr.iloc[0,2], dataframe2_chr.iloc[0,2])
print(dataframe1_chr.iloc[0,3], dataframe2_chr.iloc[0,3])
print(len(dataframe1_chr), len(dataframe2_chr))
print('---------------------------------')

binaryframe1_chr = dataframe1_chr[['chromosome','start','end']].copy()
binaryframe1_chr['binary_val'] = np.sign(dataframe1_chr[['datavalue']])
binaryframe2_chr = dataframe2_chr[['chromosome','start','end']].copy()
binaryframe2_chr['binary_val'] = np.sign(dataframe2_chr[['datavalue']])

#Here we get the difference of the data values in both dataframes, storing the difference in a list which will later be used as dataframe3
while(index < len(binaryframe1_chr) and index < len(binaryframe2_chr) ):

    if( (binaryframe1_chr.iloc[index,START_CHROM_COLUMN] == binaryframe2_chr.iloc[index,START_CHROM_COLUMN]) and (binaryframe1_chr.iloc[index,END_CHROM_COLUMN] == binaryframe2_chr.iloc[index,END_CHROM_COLUMN]) ):
        
        the_diff = binaryframe1_chr.iloc[index,BINARY_COLUMN] - binaryframe2_chr.iloc[index,BINARY_COLUMN]
        difference_list.append(the_diff)

    index = index + 1


#build the new 3rd dataframe
binaryframe3_chr = dataframe1_chr[['chromosome','start','end']].copy()
binaryframe3_chr['Difference Between Bf1 and BF2'] = difference_list
#ALTERNATIVE LINE TO USE:
#dataframe3 = pd.DataFrame(difference_list,columns=['Difference Between Bf1 and BF2'])


'''LETS DROP THE NA VALUES FOR ALL DATAFRAMES'''
#binaryframe1_chr.dropna(inplace=True)
#binaryframe2_chr.dropna(inplace=True)
#binaryframe3_chr.dropna(inplace=True)

print('-------AFTER ALL PROCESSING:------')
print('The length of the 1st binary frame is: ',len(binaryframe1_chr))
print('The length of the 2nd binary frame is: ',len(binaryframe2_chr))
print('The length of the 3rd binary frame is: ',len(binaryframe3_chr))
print('----------------------------------')

#get the row-index values of the dataframes, which also represent the 250kb block for each datavalue between -1 and 1
df1_indices = binaryframe1_chr.index.values.tolist()
df2_indices = binaryframe2_chr.index.values.tolist()
df3_indices = binaryframe3_chr.index.values.tolist()

#Overall Chart
fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col')
fig.set_figheight(13.90625) #1335 pixels
fig.set_figwidth(26.666666667) #2560 pixels
fig.tight_layout(pad = 10)
plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
x_axis1 = binaryframe1_chr['start'].div(1000000)
x_axis2 = binaryframe2_chr['start'].div(1000000)
x_axis3 = binaryframe3_chr['start'].div(1000000)
chart_padding = 0.1


#The First Chart
ax[0].set_title(cell_name1, loc = 'left', fontsize = 28)
#ax[0].set_xlabel('250kb bin', fontsize = 16,labelpad= 25)
ax[0].set_ylabel('PC1', fontsize = 32,labelpad= 12)
ax[0].set_ylim([-1 - chart_padding, 1 + chart_padding])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].tick_params(axis = 'x', labelsize = 32, length = 0)
ax[0].tick_params(axis = 'y', labelsize = 32)
l1 = ax[0].plot(x_axis1,binaryframe1_chr['binary_val'],'b', label = (cell_name1 + ' compartments'), linewidth = 0)
ax0_blue = ax[0].fill_between(x_axis1,0,binaryframe1_chr['binary_val'],where = binaryframe1_chr['binary_val'] >= 0,interpolate = True,facecolor = 'Navy')
ax0_gold = ax[0].fill_between(x_axis1,0,binaryframe1_chr['binary_val'],where = binaryframe1_chr['binary_val'] <= 0,interpolate = True,facecolor = 'Gold')
ax[0].axhline(y=0, color = 'black')
ax[0].margins(0)

#The Second Chart
ax[1].set_title( (cell_name2), loc = 'left', fontsize = 28)
#ax[1].set_xlabel('250kb bin', fontsize = 16,labelpad= 25)
ax[1].set_ylabel('PC1', fontsize = 32,labelpad= 12.0)
ax[1].set_ylim([-1 - chart_padding, 1 + chart_padding ])
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].tick_params(axis = 'x', labelsize = 32, length = 0)
ax[1].tick_params(axis = 'y', labelsize = 32)
l2 = ax[1].plot(x_axis2,binaryframe2_chr['binary_val'],'r', label = (cell_name2 + ' compartments'), linewidth = 0)
ax[1].fill_between(x_axis2,0,binaryframe2_chr['binary_val'],where = binaryframe2_chr['binary_val'] >= 0,interpolate = True,facecolor = 'Navy')
ax[1].fill_between(x_axis2,0,binaryframe2_chr['binary_val'],where = binaryframe2_chr['binary_val'] <= 0,interpolate = True,facecolor = 'Gold')
ax[1].axhline(y=0, color = 'black')
ax[1].margins(0)


#The Third Chart
ax[2].set_title(cell_name1 + ' - '+ cell_name2, y= 1, loc = 'left',fontsize = 28)
#ax[2].set_xlabel('chrom1:1-248956422', fontsize = 36,labelpad= 12) #HUMAN
ax[2].set_xlabel('Genomic Position: Chr1 (Mb)', fontsize = 32,labelpad= 12) #MOUSE
ax[2].set_ylabel('Difference', fontsize = 32,labelpad= 12)
ax[2].set_ylim([-2 - chart_padding, 2 + chart_padding])
ax[2].set_yticks([-2,-1,0,1,2])
#ax[2].set_xticks([1,25,50,75,100,125,150,175,200,225,250]) #HUMAN
#ax[2].set_xticks([1,25,50,75,100,125,150,175]) #MOUSE
x_ticks = np.arange(15,210,15)
x_ticks = np.insert(x_ticks,0,1) #parameters = array, index, value
ax[2].set_xticks(x_ticks)
ax[2].tick_params(axis = 'x', labelsize = 32)
ax[2].tick_params(axis = 'y', labelsize = 32)
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
l3 = ax[2].plot(x_axis3,binaryframe3_chr['Difference Between Bf1 and BF2'],'green', label = 'Difference between ' + cell_name1 + ' and ' + cell_name2 ,linewidth = 0)
ax[2].fill_between(x_axis3,0,binaryframe3_chr['Difference Between Bf1 and BF2'],where = binaryframe3_chr['Difference Between Bf1 and BF2'] >= 0,interpolate = True,facecolor = 'Navy')
ax[2].fill_between(x_axis3,0,binaryframe3_chr['Difference Between Bf1 and BF2'],where = binaryframe3_chr['Difference Between Bf1 and BF2'] <= 0,interpolate = True,facecolor = 'Gold')
ax[2].axhline(y=0, color = 'black')
ax[2].margins(0)


#End of chart processing
fig.legend([ax0_blue,ax0_gold], ['A Compartment', 'B Compartment'], loc = 'upper center', prop = {'size': 24}, frameon = False, ncol = 2, bbox_to_anchor = (0.5,0.944))
ax[0].grid(alpha = 0.4)
ax[1].grid(alpha = 0.4)
ax[2].grid(alpha = 0.4)
plt.show()