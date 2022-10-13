"""
Author: Rekkab Gill
Date: October 7 2022
Details: Program generates a chart to visualize chromosomal compartments, both continuous and discretized. 
"""


from cProfile import label
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
import sys 

file_path1 = sys.argv[1]    #first arguement, the path to the compartment file in .txt format
cell_name1 = sys.argv[2]    #the name of the cell


dataframe1 = pd.read_csv(file_path1, sep = "\t", header = None, comment = '#')
dataframe1.columns = ['chromosome', 'start', 'end', 'datavalue']
dataframe1_chr = dataframe1[dataframe1['chromosome'] == 'chr1'].copy() #the copy is necessary to prevent the settingwithcopy warning (it flags confusing chained assignments. i.e. are you referencing original or new dataframe)


#CHANGE THE START AND END CHROM
#dataframe1_chr['start'] = dataframe1_chr['start'] + 1 #we do this because Chris says genome units start at 1 not 0, so push everyting up 1
#dataframe1_chr['end'] = dataframe1_chr['end'] + 1

#Below are the integer indices of the headers in the dataframe (i.e. START is where the chromosome bin starts, END is where it ends, etc.)
#CONSTANTS:
START_CHROM_COLUMN = 1
END_CHROM_COLUMN = 2
DATAVAL_COLUMN = 3
SCALED_DATAVAL_COLUMN = 4
index = 0
difference_list = []
the_diff = 0

#replace the NAN values with 0
dataframe1_chr['datavalue'] = dataframe1_chr['datavalue'].fillna(0)
binaryframe1_chr = dataframe1_chr[['chromosome','start','end']].copy()
binaryframe1_chr['binary_val'] = np.sign(dataframe1_chr[['datavalue']])

#Overall Chart
fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
fig.set_figheight(13.90625) #1335 pixels
fig.set_figwidth(26.666666667) #2560 pixels
fig.tight_layout(pad = 10)
x_axis1 = dataframe1_chr['start'].div(1000000)
x_axis2 = binaryframe1_chr['start'].div(1000000)
tick_list = x_axis2.to_list()
#x_ticks = ['1.0','25.0','50.0','75.0','100.0','125.0','150.0','175.0']
#x_ticks = [1,25,50,75,100,125,150,175]
x_ticks = np.arange(15,210,15)
x_ticks = np.insert(x_ticks,0,1) #parameters = array, index, value


plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
#The First Chart
ax[0].set_title(cell_name1, y= 1, loc = 'left', fontsize = 28)
#ax[0].set_xlabel('250kb bin', fontsize = 16,labelpad= 25)
ax[0].set_ylabel('Continuous PC1', fontsize = 32,labelpad= 12)
ax0_max_range = np.max([np.max(dataframe1_chr['datavalue']), np.min(dataframe1_chr['datavalue']) * (-1)]) # np.max returns the maximum along a given axis 
ax[0].set_ylim([ax0_max_range * -1.1, ax0_max_range * 1.1])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].tick_params(axis = 'x', labelsize = 32, length = 0) #length changes of the size of those little black ticks
ax[0].tick_params(axis = 'y', labelsize = 32)
l1 = ax[0].plot(x_axis1,dataframe1_chr['datavalue'],'b', label = (cell_name1 + ' compartments'), linewidth = 0)
ax1_blue = ax[0].fill_between(x_axis1,0,dataframe1_chr['datavalue'],where = dataframe1_chr['datavalue'] >= 0,interpolate = True,facecolor = 'Navy')
ax1_gold = ax[0].fill_between(x_axis1,0,dataframe1_chr['datavalue'],where = dataframe1_chr['datavalue'] <= 0,interpolate = True,facecolor = 'Gold')
ax[0].axhline(y=0, color = 'black', linewidth = 0.5)
ax[0].margins(0)

"""
#Make the x_axis labels
xtick_list = []
for value in x_axis2:
    if value in x_ticks:
        xtick_list.append(value)
    else:
        xtick_list.append(' ')
"""

#The Second Chart
ax[1].set_title(cell_name1, y= 1, loc = 'left', fontsize = 28)
ax[1].set_xlabel('Genomic Position: Chr1 (Mb)', fontsize = 32,labelpad= 12) #MOUSE
ax[1].set_ylim([-1.1,1.1])
ax[1].set_yticks([-1,0,1])
ax[1].set_ylabel('Discretized PC1', fontsize = 32,labelpad= 12)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(True)
ax[1].tick_params(axis = 'x', labelsize = 32)
ax[1].tick_params(axis = 'y', labelsize = 32)
#uses these 2 for all ticks but some labels: #ax[1].set_xticks(x_axis2) #MOUSE
                                             #ax[1].set_xticklabels(xtick_list)
ax[1].set_xticks(x_ticks)
l2 = ax[1].plot(x_axis2, binaryframe1_chr['binary_val'],'r', label = (cell_name1 + ' compartments'), linewidth = 0)
ax[1].fill_between(x_axis2,0,binaryframe1_chr['binary_val'],where = binaryframe1_chr['binary_val'] >= 0,interpolate = True,facecolor = 'Navy')
ax[1].fill_between(x_axis2,0,binaryframe1_chr['binary_val'],where = binaryframe1_chr['binary_val'] <= 0,interpolate = True,facecolor = 'Gold')
ax[1].axhline(y=0, color = 'black', linewidth = 0.5)
ax[1].margins(0)




#End of chart processing
fig.legend([ax1_blue,ax1_gold], ['A Compartment', 'B Compartment'], loc = 'upper center', prop = {'size': 24}, frameon = False, ncol = 2, bbox_to_anchor = (0.5,0.944))
ax[0].grid(alpha = 0.4)
ax[1].grid(alpha = 0.4)
plt.show()
