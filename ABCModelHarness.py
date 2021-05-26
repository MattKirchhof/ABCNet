import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from multiprocessing import Pool
import random
import sys
import os

import DNADataLoader
import DEFDataLoader
import DEFModelHarness
from DNADataLoader import DNADataset
from ABCNet import ABCNET, DEFNet

def printToNetLog(args, n_batches):
    if not os.path.exists("./Data/Results"):
        os.mkdir("./Data/Results")
    f = open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainingLog.txt"), "a")
    f.write("\n")
    f.write("------- NEW RUN PARAMETERS -------\n")
    f.write("batch_size=" + str(args["batch_size"]) + '\n')
    f.write("epochs=" + str(args["epochs"]) + '\n')
    f.write("learning_rate=" + str(args["learning_rate"]) + '\n')
    f.close()

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", args["batch_size"])
    print("epochs=", args["epochs"])
    print("learning_rate=", args["learning_rate"])
    print("NUMBER OF BATCHES PER EPOCH: ", n_batches)
    print("=" * 30)


def storeTrainAccuracies(args, pred, labels, zero_norms, train_loss = -1.0):
    TP,FP,TN,FN = 0,0,0,0

    for i,p in enumerate(pred):
        #B Compartment
        if ( p[0] <= zero_norms[i] ):
            if (labels[i] <= zero_norms[i]):
                TN += 1
            else:
                FN += 1
        #A Compartment
        else:
            if ( labels[i] > zero_norms[i] ):
                TP += 1
            else:
                FP += 1

    with open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainAcc.txt"), "a") as trainf:
        trainf.write(str((TP+TN)/(TP+TN+FP+FN))+"\n")
    
    if train_loss != -1.0:
        with open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainLoss.txt"), "a") as trainl:
            trainl.write(str(train_loss) + "\n")


def getOptimizer(model):
    if args["optim"] == "SGD" :
        return torch.optim.SGD(model.parameters(), lr=args["learning_rate"], momentum=args["momentum"], nesterov=True)
    elif args["optim"] == "AdamW" :
        return torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    else:
        raise Exception("Unknown Optimizer specified")
    

def getLoss():
    if args['loss'] == 'MSE':
        return torch.nn.MSELoss()
    elif args['loss'] == 'L1':
        return torch.nn.L1Loss()
    else:
        raise Exception("Unknown Loss specified")


def runABCTest(testType, model, dataLoader, epoch, args, return_preds=False):
    print("\nRunning " + testType + "...")

    model.eval() # Notify layers we are in test mode

    total_loss = [[]]
    predictions = []
    expectedOuts = []

    #Stats
    TP, TN, FP, FN = 0,0,0,0

    for labels, inputs, zero_norm in dataLoader:
        # Move values to the GPU
        if args["use_cuda"]:
            inputs = inputs.cuda()
            labels = labels.cuda()
            zero_norm = zero_norm.cuda()

        with torch.no_grad():
            classifications = model(inputs)
            val_loss = model.loss(classifications, labels)

            total_loss[0].append(val_loss)

            # Gather statistics
            for i,classification in enumerate(classifications):
                if 'pred' in args['loss']:
                    classification = classification[0]

                #B Compartment
                if ( classification <= zero_norm[i] ):
                    if (labels[i] <= zero_norm[i]):
                        TN += 1
                    else:
                        FN += 1
                #A Compartment
                else:
                    if ( labels[i] > zero_norm[i] ):
                        TP += 1
                    else:
                        FP += 1

                if testType == "Testing":
                    predictions.append(classification.item())
                    expectedOuts.append(labels[i].item())

    total_A_Comp = FN + TP
    total_B_Comp = FP + TN
    for c in range(len(total_loss)):
        total_loss[c] = sum(total_loss[c]) / len(total_loss[c])

    print(testType + " accuracy: ", str((TP+TN)/(total_A_Comp+total_B_Comp)) )
    print(testType + " loss: ", total_loss)

    # Text file logging
    if not return_preds:
        f = open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TrainingLog.txt"), "a")
        f.write("\n")
        f.write("****" + testType + " FOR EPOCH(train)/CHROMOSOME(test): " + str(epoch) + "\n")
        f.write("TRUE POSITIVES: " +  str(TP) + "\n")
        f.write("TRUE NEGATIVES: " +  str(TN) + "\n")
        f.write("FALSE POSITIVES: " +  str(FP) + "\n")
        f.write("FALSE NEGATIVES: " +  str(FN) + "\n")
        f.write("Total A compartment examples:" +  str(total_A_Comp) + "\n")
        f.write("Total B compartment examples:" +  str(total_B_Comp) + "\n")
        f.write(testType + " losses: " +  str(total_loss) + "\n")
        f.write(testType + " accurracy: " +  str((TP+TN)/(total_A_Comp+total_B_Comp)) + "\n")
        f.close()

        if testType == "Validation":
            with open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "ValAcc.txt"), "a") as valf:
                valf.write(str((TP+TN)/(total_A_Comp+total_B_Comp))+"\n")

        if testType == "Testing":
            with open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + "TestAcc.txt"), "a") as testf:
                testf.write(str(args["instance_num"]) + "," + str((TP+TN)/(total_A_Comp+total_B_Comp)) + "\n")

        # Testing set never shuffled, so we record the predictions made for comparison to ground truth
        if testType == "Testing":
            with open(os.path.join("./Data/Results", str(args["model_chromosome"]) + "," + str(args["instance_num"]) + "," + "TestPredictions.txt"), "w") as testf:

                for p in range(len(predictions)):
                    testf.write(str(predictions[p]) + "," + str(expectedOuts[p]) + "\n")

    if return_preds:
        return predictions, expectedOuts
    return ((TP+TN)/(total_A_Comp+total_B_Comp))


def trainABCNet(model, dloader_train, dloader_val, dloader_test, args):
    n_batches = len(dloader_train.dataset)/args["batch_size"]
    printToNetLog(args, n_batches)

    # Init the loss and optimizer functions
    model.loss = getLoss()
    model.optimizer = getOptimizer(model)

    print("Performing initial Validation Test")
    acc = runABCTest("Validation", model, dloader_val, 0, args)
    max_val_acc = 0.0 # Used for early stopping and LR reduction

    # Loop over each epoch of training
    for epoch in range(args["epochs"]):
        print("Running Training...")
        model.train() # Notify layers we are in train mode

        running_predictions = []
        running_labels = []
        running_zero_norms = []
        total_train_loss = [[]]
        print_every = int(1200/args["batch_size"]) #per training examples
        i = 0

        for labels, inputs, zero_norm in dloader_train:
            i += 1
            # Move values to the GPU
            if args["use_cuda"]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            # Reset our gradients to zero
            model.optimizer.zero_grad()

            # Forward pass, backward pass and optimize
            outputs = model(inputs)
            running_predictions.extend(outputs.to('cpu'))
            running_labels.extend(labels.to('cpu'))
            running_zero_norms.extend(zero_norm)
            train_loss = model.loss(outputs,labels)
            train_loss.backward()
            model.optimizer.step()

            # Keep track of loss for statistics
            total_train_loss[0].append(train_loss.item())

            # print network status on occasion
            if (i + 1) % (print_every) == 0:
                num_batches = int(len(dloader_train.sampler) / args['batch_size'])
                print_train_loss = []
                for c in range(len(total_train_loss)):
                    print_train_loss.append([])
                    print_train_loss[c] = sum(total_train_loss[c]) / len(total_train_loss[c])
                print("Epoch {}, {:d}% \t train_loss: {:.4f}".format( epoch+1, int( 100 * (i+1) / num_batches ), print_train_loss[0]) )
                print("Last Output:", outputs[0][0], '\n', "Desired Output:", labels[0])

        training_loss = np.average(total_train_loss[0])
        storeTrainAccuracies(args, running_predictions, running_labels, running_zero_norms, training_loss)

        #VALIDATION RUN AFTER EPOCH
        acc = runABCTest("Validation", model, dloader_val, epoch+1, args)
        
        if (args["use_early_stopping"]):
            if (acc > max_val_acc):
                max_val_acc = acc
                stagnant_val_acc_count = 0
                print("New Best Validation Acc " + str(max_val_acc))
            else:
                stagnant_val_acc_count += 1
                print("Validation stagnant: " + str(stagnant_val_acc_count))

            if (stagnant_val_acc_count >= args["stop_training_after"]):
                print("EARLY STOPPING REACHED! Epoch: " + str(epoch))
                print("    Epochs since last validation accuracy reduction: " + str(stagnant_val_acc_count))
                print("    Maximum validation accuracy: " + str(max_val_acc))
                print("    Current validation accuracy: " + str(acc))
                break
            
    print("-"*30 + "\nTraining finished \nRunning Final Test...")

    # Run a test for withheld chromosome to evaluate its performance
    runABCTest("Testing", model, dloader_test, i, args)
    return model


def saveModel(model, args, PATH):
    '''
    Saves a completed network to disk
    '''
    torch.save([model.state_dict(), args], PATH)


def loadModel(model, PATH):
    '''
    Loads a network saved to disk
    '''
    ckpt = torch.load(PATH)
    model.load_state_dict(ckpt[0])
    return model, ckpt[1]


def loadArgs(args):
    # DEFAULTS
    args["epochs"] = 24
    args["optim"] = "SGD"
    args["loss"] = 'MSE'
    args["learning_rate"] = 0.005
    args["momentum"] = 0.8
    args["use_early_stopping"] = False
    args["use_variable_LR"] = False
    args["batch_size"] = 32
    args["data_loader_workers"] = 16
    args['drop_last'] = True
    args["zeroNormalized"] = 0.4855277624356476 # This we calculated as the "0" value normalized 
    args["use_cuda"] = True
    args["reduce_LR_after"] = 0
    args["stop_training_after"] = 0

    argsfile = "modelargs.txt"
    if os.path.exists(argsfile):
        with open(argsfile, 'r') as f:
            values = f.readlines()
            for line in values:
                value = line.split()
                if value[1].strip() == "True": # parse as true
                    args[value[0].strip()] = True
                elif value[1].strip() == "False": # parse as false
                    args[value[0].strip()] = False
                else:
                    try: # parse as int
                        args[value[0].strip()] = int(value[1].strip())
                    except:
                        try: # parse as float
                            args[value[0].strip()] = float(value[1].strip())
                        except: # parse as string
                            args[value[0].strip()] = value[1].strip()
    else:
        print("Args file not found.") 
    print(args)
    return args

#################################
########## MAIN  CODE ###########
#################################
if __name__ == '__main__':
    use_seed = False
    args = {}
    model_chromosome = 0 # We will use chr1 as our test set
    instance_num = 1 # Model instance number. Increment this if you want to train a second model with different logs tracked

    args["model_chromosome"] = model_chromosome
    args["instance_num"] = instance_num

    if torch.cuda.is_available():
        print("Cuda Available:", torch.cuda.get_device_name(0))
        cuda = torch.device('cuda')
        args["use_cuda"] = True
    else:
        print("Cuda UNAVAILABLE")
        args["use_cuda"] = False

    args = loadArgs(args)

    #Make the randomization of the network static. All we are changing is which chromosome is withheld from the training/val set
    if (use_seed == True):
        manualSeed = 123
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print ("CHROMOSOME WITHHELD: ", str(model_chromosome))
    dloader_train, dloader_val, dloader_test = DNADataLoader.getLoaders(model_chromosome, args)

    ABCModel = ABCNET()
    if args["use_cuda"]:
        ABCModel = ABCModel.to('cuda')

    print("\nBeginning model training...")
    ABCModelTrained = trainABCNet(ABCModel, dloader_train, dloader_val, dloader_test, args)
    if not os.path.exists("model_ckpts"):
        os.mkdir("model_ckpts")
    saveModel(ABCModelTrained, args, "./Data/ModelCkpts/ABCNet-" + args['model_name'] + "-Chr" + str(model_chromosome) + ".pt")
