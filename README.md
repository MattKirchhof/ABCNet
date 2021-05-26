# ABCNet

ABCNet is a convolutional neural network for the prediction of A/B compartments directly from a reference genome.

## Environment

ABCNet has limited requirements. Only the following packages are required:

1) Python 3.9.4 or later -> Earlier versions of python3 should also work just fine
2) Numpy -> Install using pip with "pip install numpy"
3) PyTorch -> Install by visiting https://pytorch.org/get-started/locally/

## Running ABCNet yourself

To run ABCNet, we suggest the following steps:

1) Clone this repository
2) Download a copy of the mm10 house mouse reference genome (you can find all 19 chromosomes here: http://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/)
3) Extract the downloaded chromosome files into the "../ABCNet/Data/Genome_DataMM10" folder
4) Use "python DataPreprocessing.py" to run the data preprocessor and prepare the reference genome for training
5) Once the data preprocessing is complete, run "python ABCModelHarness.py"

Notes:
- It is suggested you run ABCNet on a system with 32gb of RAM or more
- During training, ABCModelHarness will:
    - create txt files within the "Data/Results" directory.
    - periodically print current training status, including its current epoch, current percent complete of that epoch, current loss, and the last output value vs its ground truth.
    - Within the output Data/Results/X,X,TrainingLog.txt, you will find the results of each validation, and the models final test accuracy.

