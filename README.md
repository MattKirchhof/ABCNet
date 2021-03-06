# ABCNet and DEFNet

ABCNet is a convolutional neural network for the prediction of A/B compartments directly from a reference genome.

## Environment

ABCNet has limited requirements. Only the following packages are required:

1) Python 3.9.4 or later -> Earlier versions of python3 should also work just fine
2) Numpy -> Install using pip with "pip install numpy"
3) PyTorch -> Install by visiting https://pytorch.org/get-started/locally/

## Running ABCNet yourself

### To run ABCNet, we suggest the following steps:

1) Clone this repository
2) Download a copy of the mm10 house mouse reference genome (you can find all 19 chromosomes here: http://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/)
3) Extract the downloaded chromosome files into the "../ABCNet/Data/Genome_Data" folder
4) Use "python DataPreprocessing.py" to run the data preprocessor and prepare the reference genome for training
5) Once the data preprocessing is complete, run "python ABCModelHarness.py"

### Notes:
- It is suggested you run ABCNet on a system with 32gb of RAM or more
- During training, ABCModelHarness will:
    - create txt files within the "Data/Results" directory.
    - periodically print current training status, including its current epoch, current percent complete of that epoch, current loss, and the last output value vs its ground truth.
    - Within the output Data/Results/X,X,TrainingLog.txt, you will find the results of each validation, and the models final test accuracy.

## Running DEFNet yourself

The DEFNet model runs on top of the ABCNet model output. Therefore, you must first complete the training of an ABCNet model before running DEFNet.

### To run DEFNet, we suggest the following steps:

1) Use "python DEFModelHarness.py" to train and test 5 randomly initialized instances of DEFNet on top of each saved ABCNet model within the "./Data/ModelCkpts" directory

### Notes:
- ABCNet will output a saved ABCNet model into "./Data/ModelCkpts" automatically once training completes
- DEFNet will first run ABCNet on all 19 chromosomes to gather ABCNet predictions (which are used as input to DEFNet during training). This step may take some time to complete.
- DEFNet will then Train on these ABCNet predictions, using the true compartment annotations as ground truth
- DEFnet results will be output to the "./Data/DEFResults" directory

## Citing ABC/DEFNet
If ABCNet was used in your analysis, please cite:

M. Kirchhof, C. J. Cameron and S. C. Kremer, "End-to-end chromosomal compartment prediction from reference genomes," _2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_, 2021, pp. 50-57, doi: [10.1109/BIBM52615.2021.9669521](https://doi.org/10.1109/BIBM52615.2021.9669521)
