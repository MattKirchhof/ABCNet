import ABCNet
import ABCModelHarness
import DNADataLoader
import DEFDataLoader
import ABCNet as NetworkArchitectures
import torch, os, glob
import numpy as np
import matplotlib.pyplot as plt
import time

def get_predictions(ABCModel, args, return_zero_inpts=False, return_loss = False, return_gc_count = False):
    '''
    Uses the ABCModel, creates a new dataloader to gather model predictions IN ORDER (no shuffle) for
    all chromosomes
    '''
    predictions = []
    labels = []
    losses = []
    zero_inpts, zero_inpts_bad  = [], []
    zero_norms = []
    gc_counts = []
    ABCModel.eval()
    
    with torch.no_grad():
        for c in range(19):
            predictions.append([])
            labels.append([])
            gc_counts.append([])
            zero_inpts.append([])
            zero_inpts_bad.append([])
            
            _, _, tloader = DNADataLoader.getLoaders(c, args, chrom_limiter=-1, shuffle_data = False, valid_size = 0.0)

            for label, inpt, zero_norm in tloader:

                if args["use_cuda"]:
                    inpt = inpt.to('cuda')
                    label = label.to('cuda')

                pred = ABCModel(inpt)
                loss = ABCModel.loss(pred,label)

                if args["use_cuda"]:
                    inpt.to('cpu')
                    pred.to('cpu')
                    label.to('cpu')
                
                gc_counts[c].append(inpt[0][0].tolist().count(1) + inpt[0][1].tolist().count(1))
                predictions[c].append(pred.item())
                labels[c].append(label.item())
                losses.append(loss.item())

                if len(zero_norms) <= c:
                    zero_norms.append(zero_norm.item())
                
                if return_zero_inpts and label.item() == zero_norms[c] and pred.item() >= label.item()-0.02 and pred.item() <= label.item()+0.02:
                    print("SAVING CORRECT: LABEL - " + str(label.item()) + "  PRED - " + str(pred.item()))
                    zero_inpts[c].append(inpt)
                
                if return_zero_inpts and label.item() == zero_norms[c] and (pred.item() < label.item()-0.02 or pred.item() > label.item()+0.02):
                    print("SAVING INCORRECT: LABEL - " + str(label.item()) + "  PRED - " + str(pred.item()))
                    zero_inpts_bad[c].append(inpt)


    if return_zero_inpts and return_loss and return_gc_count:
        return predictions, labels, zero_norms, zero_inpts, zero_inpts_bad, losses, gc_counts
    return predictions, labels, zero_norms


def get_ABC_acc_Fast(predictions, labels, zero_norms, args):
    '''
    Uses the previously calculated preds and labels from the dset generation step to evaluate model accuracy
    '''
    final_preds, final_labels = [], []
    TN, TP, FN, FP = 0,0,0,0

    c = args['model_chromosome']
    for j, pred in enumerate(predictions[c]):
        # Evaluate prediction
        if ( pred <= zero_norms[c] ):
            if (labels[c][j] <= zero_norms[c]):
                TN += 1
            else:
                FN += 1
        else:
            if (labels[c][j] > zero_norms[c] ):
                TP += 1
            else:
                FP += 1
        final_preds.append(pred)
        final_labels.append(labels[c][j])

    accuracy = (TP+TN)/(TP+TN+FP+FN)

    # Log the results
    with open(os.path.join( args['dir'], 'DEFResults/DEFPreds' + args['model_name'] + '.txt'), 'w+') as f:
        for lab in final_labels:
            f.write(str(lab) + ',')
        f.write('\n')
        for pred in final_preds:
            f.write(str(pred) + ',')
        f.write('\n')
    with open(os.path.join( args['dir'], 'DEFResults/DEFTestAccuracy' + args['model_name'] + '.txt'), 'w+') as f:
        f.write('ABCNet original Accuracy: ' + str(accuracy) + '\n')
    
    print("ABCNet model accuracy on withheld test chrom: ", accuracy)
    return accuracy


def generate_dset(predictions, labels, zero_norms, args):
    '''
    Accepts a set of predictions, and labels.
    Generates 19 training examples, returns a set of dataloaders:
    train - 16 chromosomes for training
    val - 2 chroms for validation
    test - the ABCModels original withheld test chrom
    '''
    dset = []
    for c in range(len(predictions)):
        example = []
        annotation = []

        for j,pred in enumerate(predictions[c]):
            example.append(pred)
            annotation.append(labels[c][j])
        dset.append((annotation, example))

    DEFtrain_loader, DEFval_loader, DEFtest_loader = DEFDataLoader.getLoaders(args, dset, zero_norms)

    return DEFtrain_loader, DEFval_loader, DEFtest_loader


def train_DEFNet(DEFModel, DEFloader_train, DEFloader_val, args):
    '''
    Trains the DEFNet given the training and validation loaders
    Stops once validation does not improve for X epochs
    '''    
    DEFModel.loss = ABCModelHarness.getLoss()
    DEFModel.optimizer = ABCModelHarness.getOptimizer(DEFModel)

    # Train DEF until valid decreases
    best_DEF_val_acc, curr_DEF_val_acc = 0.0, 0.01
    stale_count = 0

    while stale_count < 10:
        if best_DEF_val_acc >= curr_DEF_val_acc:
            stale_count += 1
        else:
            best_DEF_val_acc = curr_DEF_val_acc
            ABCModelHarness.saveModel(DEFModel, args, 'bestDEFModel.pt')
            stale_count = 0
        train_loss_avg = 0.0

        # Training of the epoch
        for labels, inputs, _ in DEFloader_train:
            # Move values to the GPU
            if args["use_cuda"]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            
            DEFModel.optimizer.zero_grad()
            outputs = DEFModel(inputs.unsqueeze(0))
            train_loss = DEFModel.loss(outputs, labels)
            train_loss_avg += train_loss.item()
            train_loss.backward()
            DEFModel.optimizer.step()
        
        print("\nTrain Loss: ", train_loss_avg/len(DEFloader_train), "\n")
        
        # Validation of the epoch
        TN, TP, FN, FP = 0,0,0,0
        for labels, inputs, zero_norms in DEFloader_val:
            if args["use_cuda"]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            outputs = DEFModel(inputs.unsqueeze(0))
            val_loss = DEFModel.loss(outputs, labels)
            preds = outputs.squeeze().to('cpu').tolist()
            labels = labels.squeeze().to('cpu').tolist()
            zero_norm = zero_norms.item()

            for i in range(len(preds)):
                new_pred = preds[i]
                annotation = labels[i]

                #B Compartment
                if ( new_pred <= zero_norm ):
                    if (annotation <= zero_norm):
                        TN += 1
                    else:
                        FN += 1
                #A Compartment
                else:
                    if (annotation > zero_norm ):
                        TP += 1
                    else:
                        FP += 1

        curr_DEF_val_acc = (TP+TN)/(TP+TN+FP+FN)
        print("Validation accuracy: ", curr_DEF_val_acc, "\t Current best:", best_DEF_val_acc)
        print("Validation loss:", val_loss.item())
    
    # Record the final best validation accuracy, to choose what we think will be our best instance when
    # calculating test accuracy
    with open(os.path.join( args['dir'], 'DEFResults/DEFValAccuracy' + args['model_name'] + '.txt'), 'a') as f:
        f.write(str(best_DEF_val_acc) + '\n')
    
    return DEFModel


def testDEFNet(DEFModel, DEFtest_loader, args):
    #Load in the best val acc ckpt
    DEFModel, ABCargs = ABCModelHarness.loadModel(DEFModel, 'bestDEFModel.pt')

    final_preds = []
    TN, TP, FN, FP = 0,0,0,0
    for labels, inputs, zero_norm in DEFtest_loader:
        args['zeroNormalized'] = zero_norm #For use in the final plotting steps

        if args["use_cuda"]:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

        outputs = DEFModel(inputs.unsqueeze(0))
        test_loss = DEFModel.loss(outputs, labels)
        preds = outputs.squeeze().to('cpu').tolist()
        labels = labels.squeeze().to('cpu').tolist()

        for i in range(len(preds)):
            new_pred = preds[i]
            annotation = labels[i]

            final_preds.append(new_pred)

            #B Compartment
            if ( new_pred <= zero_norm ):
                if (annotation <= zero_norm):
                    TN += 1
                else:
                    FN += 1
            #A Compartment
            else:
                if (annotation > zero_norm ):
                    TP += 1
                else:
                    FP += 1

        curr_DEF_test_acc = (TP+TN)/(TP+TN+FP+FN)
        print("Test accuracy: ", curr_DEF_test_acc)
        print("Test loss:", test_loss.item())
    
    with open(os.path.join( args['dir'], 'DEFResults/DEFTestAccuracy' + args['model_name'] + '.txt'), 'a') as f:
        f.write(str(curr_DEF_test_acc) + '\n')

    with open(os.path.join( args['dir'], 'DEFResults/DEFPreds' + args['model_name'] + '.txt'), 'a') as f:
        for pred in final_preds:
            f.write(str(pred) + ',')
        f.write('\n')


def plotPredsBeforeAfter(args):
    '''
    Plots ABCnet, along with DEFNet smoothing and the GT annotation onto a line plot
    '''
    acc_src = os.path.join( args['dir'], 'DEFResults/DEFTestAccuracy' + args['model_name'] + '.txt')
    preds_src = os.path.join( args['dir'], 'DEFResults/DEFPreds' + args['model_name'] + '.txt')

    labels, abc_preds = [], []
    abc_acc = 0.0
    def_preds = []
    def_accs = []
    with open(preds_src, 'r') as f:
        lines = f.readlines()
        labels.append(np.array(lines[0].strip().split(',')[:-1]).astype(np.float))
        abc_preds.append(np.array(lines[1].strip().split(',')[:-1]).astype(np.float))
        for line in lines[2:]:
            def_preds.append(np.array(line.strip().split(',')[:-1]).astype(np.float))

    with open(acc_src, 'r') as f:
        lines = f.readlines()
        abc_acc = lines[0].strip()
        for line in lines[1:]:
            def_accs.append(line.strip())
    
    bin_nums = list(range(len(abc_preds[0])))

    fig, axes = plt.subplots(nrows=len(def_preds), ncols=1, figsize=(24, 5*len(def_preds)))
    for i in range(1, len(def_preds)+1):
        axes[i-1].plot(bin_nums, labels[0], '-', c='k', label="Ground Truth")
        axes[i-1].plot(bin_nums, abc_preds[0], '-', c='b', label="ABCNet predictions")
        axes[i-1].plot(bin_nums, def_preds[i-1], '-', c='r', label="DEFNet adjusted predictions")
        axes[i-1].axhline(args['zeroNormalized'], c='k')
        axes[i-1].set_xlabel('bin')
        axes[i-1].set_ylabel('PCA Pred')
        axes[i-1].set_title(args['model_name'] + ' - DEFNet:' + str(i) + ' - Acc:' + str(def_accs[i-1]))
        if i == 1:
            axes[i-1].legend()

    plt.tight_layout()
    plt.savefig('DEFResults/DEFPredictions' + args['model_name'] + '.png')


def train_and_test(ABCNet, args):
    '''
    Can be called from the main model harness
    Accepts a trained ABCNet and trains a DEFNet on top of it using the ABCNet's output
    '''
    models_src = args['models_src']
    ckptfiles = glob.glob(os.path.join(models_src, '*.pt'))
    print("ABCNet checkpoints: \n", ckptfiles)

    if not os.path.exists(os.path.join( args['dir'], 'DEFResults')):
        os.mkdir(os.path.join( args['dir'], 'DEFResults'))

    for i,ckptfile in enumerate(ckptfiles):
        args['model_name'] = ckptfile.split('/')[-1][:-3]
        args['model_chromosome'] = int(args['model_name'].split('-')[-1][3:])
        print("\nGathering ABCNET predictions for: " + args['model_name'])
        print('  Withheld Chromosome: ', args['model_chromosome'])

        ABCModel = NetworkArchitectures.ABCNET()
        ABCModel, ABCargs = ABCModelHarness.loadModel(ABCModel, ckptfile)
        ABCModel.loss = ABCModelHarness.getLoss(ABCargs)
        if args['use_cuda']:
            ABCModel = ABCModel.to('cuda')
        args['data_path'] = ABCargs['data_path']

        predictions, labels, zero_norms = get_predictions(ABCModel, args)

        print("Evaluating ABCNet accuracy..")
        ABCaccFast = get_ABC_acc_Fast(predictions, labels, zero_norms, args)

        print("Generating dataset and initializing DEF loaders")
        DEFtrain_loader, DEFval_loader, DEFtest_loader = generate_dset(predictions, labels, zero_norms, args)

        for i in range(5):
            print("Training DEFNet instance: " + str(i))
            DEFModel = NetworkArchitectures.DEFNet()
            if args['use_cuda']:
                DEFModel.to('cuda')
            DEFModel = train_DEFNet(DEFModel, DEFtrain_loader, DEFval_loader, args)

            print("Testing DEFNet instance: " + str(i))
            testDEFNet(DEFModel, DEFtest_loader, args)

        plotPredsBeforeAfter(args)


if __name__ == '__main__':
    '''
    Uses a pretrained ABCNet and runs as normal
    '''
    args = {'use_cuda':False,
            'zeroNormalized':0.5,
            'DEF_batch_size':1,
            'dir':'./Data',
            'data_path':'',
            'models_src':'./Data/ModelCkpts',
            'data_loader_workers':os.cpu_count(),
            'transitional_bounds':0.0,
            'batch_size':1,
            'model_chromosome':0,
            'model_name':'',
            'loss':'L1',
            'optim':'SGD',
            'momentum':0.9,
            'learning_rate':0.005,
            'drop_last':False,
            'DEF_drop_last':False}

    train_and_test(None, args)
