import random 
import numpy as np
import sys
import os
import multiprocessing as mp
import DNADataLoader

####
# An object containing a variety of functions for preprocessing data into parseable files
####
class PreprocessingCSV():
    globalCurrentBin = 0 #Keeps track of the range of bins belonging to each chromosome
    global_basepath = './Data/'
    compartment_file = 'compartment_file.txt'

    def __init__(self,base_path, file_name):
        self.global_basepath = './Data/' + base_path + '/'
        self.compartment_file = file_name

        if not os.path.exists(self.global_basepath):
            os.mkdir(self.global_basepath)

        if not os.path.exists(self.global_basepath + 'Processed/'):
            os.mkdir(self.global_basepath + 'Processed/')

    #Loads in the information and writes contents as "expected output, 100k character bin" per line
    def process100kChunks(self, chrom):
        processedDataFile = open(self.global_basepath + 'Processed/' + chrom + 'rawPCA.fa', 'w')
        
        #First line of each processed chrom file looks like:
        # chr1,30,0,1954,0.486666
        # chr2,30,1955,3774,0.486666
        processedDataFile.write(chrom + ',' + str(self.getStartingBinValue(chrom)) + ',' + str(self.globalCurrentBin) + ',' + str(self.globalCurrentBin + self.getEndingBinNumber(chrom)) + '\n')

        self.globalCurrentBin += self.getEndingBinNumber(chrom) +1

        count = 0 # 100k chunks of data equals 2000 lines
        chunkNum = 0 # To keep track of our progress
        chunk100k = "" # Stores each bin of characters
            
        with open('./Data/Genome_Data/' + chrom + '.fa') as rawFile:
            next(rawFile) # Skip first line (labels)
            for line in rawFile:
                if (count < 1999): # collect all the data into our bin
                    chunk100k += line.strip()
                    count += 1
                    
                else:  # Add bin as line to our file
                    chunk100k += line.strip()
                    count = 0
                    chunkNum += 1
                    expectedOut = self.getPCA(chunkNum, chrom)

                    #print(chunkNum, expectedOut)
                    processedDataFile.write(expectedOut + "," + chunk100k + '\n')

                    chunk100k = ""
        
    #Loads in the information and writes contents as "expected output, Custom character bin" per line
    def processCustomChunks(self, chrom, resolution_size, single_file):
        processedDataFile = open(self.global_basepath + 'Processed/' + chrom + 'rawPCA.fa', 'w')
        
        #First line of each processed chrom file looks like:
        # chr1,30,0,1954,0.486666
        # chr2,30,1955,3774,0.486666
        processedDataFile.write(chrom + ',' + str(self.getStartingBinValue_custom(chrom, resolution_size, single_file)) + ',' + str(self.globalCurrentBin) + ',' + str(self.globalCurrentBin + self.getEndingBinNumber_custom(chrom,resolution_size,single_file)) + '\n')

        self.globalCurrentBin += self.getEndingBinNumber_custom(chrom,resolution_size,single_file) +1

        count = 0 # Custom chunks of data equals 2000 lines
        chunkNum = 0 # To keep track of our progress
        chunkCustom = "" # Stores each bin of characters
        bin_size = (resolution_size / 50) - 1
            
        with open('./Data/Genome_Data/' + chrom + '.fa') as rawFile:
            next(rawFile) # Skip first line (labels)
            for line in rawFile:
                if (count < bin_size): # collect all the data into our bin
                    chunkCustom += line.strip()
                    count += 1
                    
                else:  # Add bin as line to our file
                    chunkCustom += line.strip()
                    count = 0
                    chunkNum += 1
                    expectedOut = self.getPCA_custom(chunkNum, chrom, resolution_size, single_file)

                    #print(chunkNum, expectedOut)
                    processedDataFile.write(expectedOut + "," + chunkCustom + '\n')

                    chunkCustom = ""


    # Gets the PCA value belonging to the current bin
    def getPCA(self, chunkNum, chrom):
        with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res100000.PC1.txt') as compFile:
            compFile.readline() # skip first line

            for line in compFile:
                data = line.split()

                if (chunkNum*100000 >= int(data[2]) and chunkNum*100000 <= int(data[3])):
                    PCA = data[5]
                    return PCA
            return '0'
    
    # Gets the PCA value belonging to the current bin
    def getPCA_custom(self, chunkNum, chrom, resolution_size,single_file = False):

        if single_file == False:
            with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res100000.PC1.txt') as compFile:
                compFile.readline() # skip first line

                for line in compFile:
                    data = line.split()
                    
                    if (chunkNum*resolution_size >= int(data[1]) and chunkNum*resolution_size <= int(data[2])):
                        #check if the PCA is a nan value:
                        if data[3] == 'nan':
                            return '0'
                        else:
                            PCA = data[3]
                            return PCA
                return '0'

        else: #the compartment file is a single file
            with open(self.global_basepath + 'Compartment_Data/' + self.compartment_file) as compFile:
                compFile.readline() # skip first line

                for line in compFile:
                    data = line.split()
                    
                    if chrom == data[0]: #only process for the appropriate chromosome 
                        if (chunkNum*resolution_size >= int(data[1]) and chunkNum*resolution_size <= int(data[2])):
                            #check if the PCA is a nan value:
                            if data[3] == 'nan':
                                return '0'
                            else:
                                PCA = data[3]
                                return PCA
                return '0'



    #Gets the location of the first evaluated bin in the data (could range from bin 30 to bin 32)
    def getStartingBinValue(self, chrom):
        with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res100000.PC1.txt') as binFile:
            binFile.readline() # skip first line
            data = binFile.readline().split()

            firstbin = int( int(data[2])/100000 )
            return firstbin

    #Gets the total number of bins in the current dataset
    def getEndingBinNumber(self, chrom):
        with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res100000.PC1.txt') as binFile:
            elements = binFile.read().split()
            last_element = elements[len(elements)-3]
            last100kBin = int( int(last_element)/100000) 

            return last100kBin
    
    #Gets the location of the first evaluated bin in the data (could range from bin 30 to bin 32)
    def getStartingBinValue_custom(self, chrom, resolution_size, single_file):

        if single_file == False:
            with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res100000.PC1.txt') as binFile:
                binFile.readline() # skip first line
                data = binFile.readline().split()

                firstbin = int( int(data[1])/resolution_size )
                return firstbin
        
        else: #its a single file 
            with open(self.global_basepath + 'Compartment_Data/' + self.compartment_file) as binFile:
                binFile.readline() # skip first line
                data = binFile.readline().split()

                firstbin = int( int(data[1])/resolution_size )
                return firstbin


    #Gets the total number of bins in the current dataset
    def getEndingBinNumber_custom(self, chrom, resolution_size, single_file):

        if single_file == False:
            with open(self.global_basepath + 'Compartment_Data/mESC-' + chrom + '-pcaOut-res100000.PC1.txt') as binFile:
                elements = binFile.read().split()
                last_element = elements[len(elements)-3]
                last100kBin = int( int(last_element)/100000) 

                return last100kBin

        else: #it is a single file

            with open(self.global_basepath + 'Compartment_Data/' + self.compartment_file) as binFile:

                for line in binFile: 
                    data = line.split()
                    chrom_label = data[0]

                    if chrom_label == '#bedGraph':
                        current_chrom = data[2].split(':')

                        if current_chrom[0] == chrom:
                            last_binVal = data[2].split('-')
                            last_bin = int (int(last_binVal[1]) / resolution_size)
                            return last_bin


    # Reprocesses all bins of data
    def reprocess(self,total_chrom, resolution_size, single_file):
        for label in range(total_chrom):
            print("Current Chrom: chr" + str(label+1))
            self.processCustomChunks("chr" + str(label+1), resolution_size, single_file)
        print("Reprocessing Completed...")


    def NormalizeFiles(self):
        #Go through all 19 preprocessed files and find max and min
        minPCA = 0.0
        maxPCA = 0.0

        for chrom in range(1, 20):

            chromPCAs = []
            with open(self.global_basepath + 'Processed/chr' + str(chrom) + 'rawPCA.fa', 'r') as tempf:
                details = tempf.readline().split(',')
                # skip the first X unused bins
                for x in range (int(details[1])):
                    tempf.readline()

                #Read the rest into memory
                for x in range (int(details[3]) - int(details[2]) - int(details[1])):
                    d = tempf.readline().strip('\n').split(',')

                    if len(d) > 1:
                        chromPCAs.append(float(d[0]))

            #Get min and max of all 19 chroms
            for curr_pca in chromPCAs:
                if float(curr_pca) < minPCA:
                    minPCA = curr_pca
                if float(curr_pca) > maxPCA:
                    maxPCA = curr_pca

        zero_norm = self.normalizedEven(minPCA, maxPCA, 0)

        print("ZERO NORMALIZED for GENOME: ", zero_norm)
        for chrom in range(1, 20):

            chromPCAs, chromSequences = [], []
            with open(self.global_basepath + 'Processed/chr' + str(chrom) + 'rawPCA.fa', 'r') as chromFile:
                details = chromFile.readline().strip('\n').split(',')

                #Read the rest into memory
                for x in range (int(details[3]) - int(details[2]) - int(details[1])):
                    d = chromFile.readline().strip('\n').split(',')

                    if len(d) > 1:
                        chromPCAs.append(float(d[0]))
                        chromSequences.append(d[1])

                with open(self.global_basepath + 'Processed/chr' + str(chrom) + 'rawPCANormalized.fa', 'w') as NormalizedDataFile: # New normalized data
                    print("Normalizing chrom file:", chrom)
                    NormalizedDataFile.write(details[0] + "," + details[1] + "," + details[2] + "," + details[3].strip('\n') + ',' + str(zero_norm) + '\n')

                    # Rewrite all lines of the chrom file with normalized PCA
                    for x in range (int(details[3]) - int(details[2]) - int(details[1])):
                        NormalizedDataFile.write(self.normalizedEven(minPCA, maxPCA, chromPCAs[x]) + "," + chromSequences[x] + '\n')

    
        print("Normalization Completed...")


    def normalized(self, minVal, maxVal, val):
        return str( (float(val)-minVal)/(maxVal-minVal) )

 
    def normalizedEven(self, min_val, max_val, val):
        '''
        Normalizes values similar to above, but forces a 0.5 zero
        '''
        if val >= 0 :
            result = val * (0.5 / max_val) + 0.5
        else:
            result = val * (0.5 / (-min_val)) + 0.5

        return str(float(result))
       

    def remove_centroids(self):
        '''
        Goes through each normalized file, and removes any bins with a PCA matching the defined normalized 0 value.
        Also adjusts the first line details values to reflect the removed bins
        '''
        global_removed = 0

        for chrom in range(1, 20):
            local_removed = 0
            original_details = []
            local_lines = []
            
            with open(self.global_basepath + 'Processed/chr' + str(chrom) + 'rawPCANormalized.fa', 'r') as originalNormFile:
                original_details = originalNormFile.readline().strip('\n').split(',')
                # Check each line for a 0 PCA
                for x in range (int(original_details[3]) - int(original_details[2]) - int(original_details[1])):
                    curr_line_str = originalNormFile.readline()
                    curr_line_split = curr_line_str.strip('\n').split(',')
                    
                    if float(curr_line_split[0]) == float(original_details[4]):
                        # centroid detected
                        local_removed += 1
                    else:
                        local_lines.append(curr_line_str)
                
            # Parsed all lines, now rewriting only non centroids with updated file details
            with open(self.global_basepath + 'Processed/chr' + str(chrom) + 'rawPCANormalized.fa', 'w') as newNormFile:
                new_start = int(original_details[2]) - global_removed
                new_end = int(original_details[3]) - global_removed - local_removed 
                new_details = "chr" + str(chrom) + ",0," + str(new_start) + "," + str(new_end) + "," + str(original_details[4]) +  "\n"
                print("Chrom " + str(chrom) + " New details line: " + new_details)
                newNormFile.write(new_details)

                for line in local_lines:
                    newNormFile.write(line)

            global_removed += local_removed
            print("REMOVED " + str(local_removed) + " local bins. Global removed: " + str(global_removed) + "\n\n")
        
    
    # Helper function to convert chars to one hot
    def convertCharToOneHot(self, c):
        char = c.upper()
        
        if char == 'A': return ['0','0','0','1']
        elif char == 'T': return ['0','0','1','0']
        elif char == 'C': return ['0','1','0','0']
        elif char == 'G': return ['1','0','0','0']
        else: return ['0','0','0','0']


    def writeToOneHotFiles(self, chromSpecs):
        chromIdx = chromSpecs[0]
        folder_name = chromSpecs[2]
        print(self.global_basepath + folder_name + '/chr' + str(chromIdx) + "rawPCANormalizedOneHot.fa")

        with open(self.global_basepath + folder_name + '/chr' + str(chromIdx) + "rawPCANormalizedOneHot.fa", 'w') as newF:
            with open(self.global_basepath + 'Processed/chr' + str(chromIdx) + 'rawPCANormalized.fa', 'r') as currF:
                details = currF.readline()
                print(details)
                newF.write(details)
                details = details.split(',')

                for lineIdx in range(int(details[1])):
                    currF.readline()

                for lineIdx in range(int(details[3]) - int(details[2]) - int(details[1])):
                    te = currF.readline().split(",")
                    if len(te) > 1:
                        newF.write(te[0])

                        channels = [[] for i in range(4)]

                        for char in te[1].strip():
                            cv = self.convertCharToOneHot(char)
                            for channel in range(4):
                                channels[channel].append(cv[channel])

                        for channel in range(4):
                            channels[channel] = ''.join(channels[channel])

                        write_str = ""
                        for channel in range(4):
                            write_str += "," + channels[channel]
                        write_str += '\n'
                        newF.write(write_str)

                        if lineIdx % 200 == 0:
                            print("Line: " + str(lineIdx) + " of Chrom: " + str(chromIdx))


    def mpOnehotEncoding(self, extra_names = ""):
        p = mp.Pool(6)
        
        # Init the folder outside of MP to avoid issues
        norm_type_code = 'g'
        folder_name = 'ProcessedOneHot_4c_' + norm_type_code + 'n'
        folder_name = folder_name + extra_names
        if not os.path.exists(os.path.join(self.global_basepath,folder_name)):
            os.mkdir(os.path.join(self.global_basepath,folder_name))

        chromSpecs = [[i, 4, folder_name] for i in range (1,20)]

        print("Starting pooled conversions..")
        p.map(self.writeToOneHotFiles, chromSpecs)
        print("One Hot Encoding Completed...")


    #Gets the PCA value belonging to the current bin
    def getABs(self, chrom):
        A = 0
        B = 0
        with open(self.global_basepath + 'Compartment_Data/IndividualBins/mESC-chr' + str(chrom+1) + '-pcaOut-res100000.PC1.txt') as compFile:
            compFile.readline() # skip first line

            for line in compFile:
                data = line.split()

                if float(data[5].strip()) > 0.:
                    B += 1
                else:
                    A +=1
        return A, B


    def getTotalDistribution(self):
        A = 0
        B = 0
        for i in range(19):
            a,b = self.getABs(i)
            A += a
            B += b

        print("Total A's", A)
        print("Total B's", B)
        print("Total overall", A+B)


if __name__ == '__main__':

    base_path = sys.argv[1]
    file_name = sys.argv[2]

    p = PreprocessingCSV(base_path, file_name)
    extra_names = "_mESC" # An extra string to attach to the resulting file folders

    total_chrom = 19
    resolution_size = 250000
    single_file = True
    p.reprocess(total_chrom, resolution_size, single_file)
    p.NormalizeFiles()
    p.remove_centroids()
    p.mpOnehotEncoding(extra_names)
    #p.getTotalDistribution()
