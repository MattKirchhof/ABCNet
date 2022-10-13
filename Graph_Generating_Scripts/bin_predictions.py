"""
Author: Rekkab Gill
Date: October 7 2022
Details: A graph to visualize the chromosomal compartment predictions and targets of the ABCnet algorithm by Matthew Kirchhoff, in a discretized manner. 
"""


import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
import sys 

file_path1 = sys.argv[1]    #file path as first argument
cell_name1 = sys.argv[2]    #the name of your predictions 
cell_name2 = sys.argv[3]    #the name of your targets 

#Below are the integer indices of the headers in the dataframe (i.e. START is where the chromosome bin starts, END is where it ends, etc.)
#CONSTANTS:
PREDICTIONS_COL = 0
GROUND_COL = 1
index = 0
difference_list = []
pred_list = []
truth_list = []
the_diff = 0
difference_title = 'Estimates - Targets'

bin_dataframe1 = pd.read_csv(file_path1, sep = ",", header = None, comment = '#')
bin_dataframe1.columns = ['predictions', 'groundtruth']
dataframe1 = pd.DataFrame()

for index in range (0,len(bin_dataframe1)):

    if bin_dataframe1.iloc[index,PREDICTIONS_COL] > 0.5:
        pred_list.append(1)
    
    elif bin_dataframe1.iloc[index,PREDICTIONS_COL] < 0.5:
        pred_list.append(-1)
    
    else: 
        pred_list.append(0)

    if bin_dataframe1.iloc[index,GROUND_COL] > 0.5:
        truth_list.append(1)
    
    elif bin_dataframe1.iloc[index,GROUND_COL] < 0.5:
        truth_list.append(-1)
    
    else: 
        truth_list.append(0)
    
dataframe1['predictions'] = pred_list
dataframe1['groundtruth'] = truth_list


for index in range(0, len(dataframe1)):

    the_diff = dataframe1.iloc[index,PREDICTIONS_COL] - dataframe1.iloc[index,GROUND_COL] #prediction - ground truth gives you difference
    difference_list.append(the_diff)


#add it to the dataframe
dataframe1['difference'] = difference_list


#get the row-index values of the dataframes, which also represent the 250kb block for each datavalue between -1 and 1
pre_df1_indices = dataframe1.index.values.tolist()
df1_indices = [x / 10 for x in pre_df1_indices]

#Overall Chart
fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col')
fig.tight_layout(pad = 2)

#x_ticks = [1,25,50,75,100,125,150,175]
#y_ticks = [-2, -1, 0, 1, 2]

print(dataframe1.loc[0:10])


"""
IMPORTANT NOTE:
    WE USE INTERPOLATE BELOW TO FILL IN THE REGIONS BETWEEN THE POINTS THAT DIFFER ABOVE THE 0 Line and Below the 0 Line. 
    THIS IS NOT APPROPRIATE FOR COMPARTMENTS BECAUSE WHEN X IS A FRACTIONAL VALUE IT ACTUALLY MEANS SOMETHING, IN REGARDS TO THE
    BIN RESOLUTION. HOWEVER HERE WE ARE JUST USING INTEGERS TO COUNT THE NUMBER OF PREDICTIONS, SO WE CAN FILL BETWEEN. IF WE USED 
    FILL BETWEEN WHEN COMPARTMENTS CROSSED OVER LIKE IN THE OTHER SCRIPTS IT WOULD INDICATE A VALUE FOR PARTS OF A 250 KILOBASE BIN 
    WHICH WE DON'T WANT TO DO. 
"""

plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
#The First Chart
ax[0].set_title(cell_name1, y= 1, loc = 'left', fontsize = 28)
#ax[0].set_xlabel('250kb bin', fontsize = 16,labelpad= 25)
ax[0].set_ylabel('PC1', fontsize = 32,labelpad= 12.0)
ax[0].set_ylim([-1.1,1.1])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].tick_params(axis = 'x', labelsize = 32, length = 0)
ax[0].tick_params(axis = 'y', labelsize = 32)
#ax[0].set_yticks(np.arange(min(y_ticks), max(y_ticks) + 0.25, 1))
l1 = ax[0].plot(df1_indices,dataframe1['predictions'],'b', label = (cell_name1 + ' compartments'), linewidth = 0)
ax0_blue = ax[0].fill_between(df1_indices,0,dataframe1['predictions'],where = dataframe1['predictions'] > 0,interpolate = True,facecolor = 'Navy')
ax0_gold = ax[0].fill_between(df1_indices,0,dataframe1['predictions'],where = dataframe1['predictions'] < 0,interpolate = True,facecolor = 'Gold')
ax[0].axhline(y=0, color = 'black', linewidth = 0.5)
ax[0].margins(0)

#The Second Chart
ax[1].set_title(cell_name2, y= 1, loc = 'left', fontsize = 28)
#ax[1].set_xlabel('250kb bin', fontsize = 16,labelpad= 25)
ax[1].set_ylim([-1.1,1.1])
ax[1].set_ylabel('PC1', fontsize = 32,labelpad= 12)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].tick_params(axis = 'x', labelsize = 32, length = 0)
ax[1].tick_params(axis = 'y', labelsize = 32)
#ax[1].set_yticks(np.arange(min(y_ticks), max(y_ticks) + 0.25, 1))
l2 = ax[1].plot(df1_indices, dataframe1['groundtruth'],'r', label = (cell_name2 + ' compartments'), linewidth = 0)
ax[1].fill_between(df1_indices,0,dataframe1['groundtruth'],where = dataframe1['groundtruth'] > 0,interpolate = True,facecolor = 'Navy')
ax[1].fill_between(df1_indices,0,dataframe1['groundtruth'],where = dataframe1['groundtruth'] < 0,interpolate = True,facecolor = 'Gold')
ax[1].axhline(y=0, color = 'black', linewidth = 0.5)
ax[1].margins(0)


#The Third Chart
#ax[2].set_title(cell_name1 + ' - '+ cell_name2, y= 1, loc = 'left',fontsize = 30)
ax[2].set_title(difference_title, y= 1, loc = 'left',fontsize = 28)
ax[2].set_xlabel('Genomic Position: Chr1 (Mb)', fontsize = 32,labelpad= 12) #MOUSE
#ax[2].set_xlabel('chrom1:1-248956422', fontsize = 36,labelpad= 12) #HUMAN
ax[2].set_ylabel('Difference', fontsize = 32,labelpad= 25)
ax[2].set_ylim([-2.1,2.1])
ax[2].tick_params(axis = 'x', labelsize = 32)
ax[2].tick_params(axis = 'y', labelsize = 32)
#ax[2].set_yticks(np.arange(min(y_ticks), max(y_ticks) + 0.25, 1))
#ax[2].set_xticks(x_ticks) #MOUSE
#ax[2].set_xticks([1,25,50,75,100,125,150,175])
x_ticks = np.arange(15,210,15)
x_ticks = np.insert(x_ticks,0,1) #parameters = array, index, value
ax[2].set_xticks(x_ticks)
ax[2].set_xlim([0,200])
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
l3 = ax[2].plot(df1_indices,dataframe1['difference'],'green', label = 'Difference between ' + cell_name1 + ' and ' + cell_name2 ,linewidth = 0)
ax[2].fill_between(df1_indices,0,dataframe1['difference'],where = dataframe1['difference'] > 0,interpolate = True,facecolor = 'Navy')
ax[2].fill_between(df1_indices,0,dataframe1['difference'],where = dataframe1['difference'] < 0,interpolate = True,facecolor = 'Gold')
ax[2].axhline(y=0, color = 'black', linewidth = 0.5)
ax[2].margins(0)


'''
#TEST:
for x in range(700,751):
    print('---------------------')
    print(dataframe1_chr.iloc[x])
    print(dataframe2_chr.iloc[x])
    print(dataframe3_chr.iloc[x])
    print('----------------------')
'''

#End of chart processing
fig.legend([ax0_blue,ax0_gold], ['A Compartment', 'B Compartment'], loc = 'upper center', prop = {'size': 24}, frameon = False, ncol = 2, bbox_to_anchor = (0.5,0.98))
ax[0].grid(alpha = 0.4)
ax[1].grid(alpha = 0.4)
ax[2].grid(alpha = 0.4)
plt.show()

#OPTIONAL: Output the bin x bin scatter plot
#RAW SCORES:
#binPlot(dataframe1_chr['datavalue'], dataframe2_chr['datavalue'],cell_name1,cell_name2)