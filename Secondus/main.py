
# internal

from util.conversions import onehot_from_fen, fen_from_filename, fen_from_64
from util.models import CNN_BatchNormLessFiltersLastLayer, CNN_BatchNormLessFilters, CNN_NoDropout, CNN_Dropout, CNN_Dropout_BatchNorm, CNN_BatchNorm, FullyConnected, LogisticRegression, CNN_LessFilters, CNN_BatchNormLessFilters
from util.models import save_model

# external 
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import time, datetime
from tqdm import tqdm

# maybe outdated, can't find it
# from torchsummary import summary

from random import randint

from PIL import Image
from pathlib import Path
from random import shuffle
import os
import re
import glob
import torch.optim as optim


device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# needs to be small because 
BATCH_SIZE = 10


# set baseline accuracy
def calculateNaiveAcc(transform=None,
                      root='train_full'):

        pathlist = list(Path(root).glob('**/*.*'))
        n_files = len(pathlist)
        _data = glob.glob(f"{root}/*.*")
        dataset_size = len(os.listdir(root))
        labels = []

        for idx in range(dataset_size):
            img = _data[:dataset_size][idx]
            img_label = re.sub(r'[\_][0-9]+', '',img) # remove underscores for dups
            try:
                label = onehot_from_fen(fen_from_filename(img_label))
                labels.append(label)
            except:
                print(img)
                raise
        flat = np.array(labels[:][:][:]).astype(int).flatten()
        
        counts = np.bincount(flat)
        most_common = np.argmax(counts)
        naive_acc = np.mean( flat == most_common )
        print (f'Total Number of labeled Spaces: {np.shape(flat)[0]}\n'
               f'Most Common Element: {most_common} '
               f'\nAccuracy of Guessing that Every Time: {naive_acc}')
        return naive_acc

naive_acc = calculateNaiveAcc(root='data/chess-dataset/labeled_preprocessed') 



from typing import Tuple

def train_model(model: nn.Module, 
                log_dir: str,
                train_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.modules.Module,
                optimizer: torch.optim.Optimizer,
                num_epochs: int=1,
                log_freq: int=5,
                print_guess=False,
                print_guess_freq=50,
                test_model_after_each_epoch=False,
                val_loader: torch.utils.data.DataLoader=None,
                test_loader: torch.utils.data.DataLoader=None,
                suppress_output: bool=True,
                disable_tqdm: bool=False,
            ) -> Tuple[nn.Module, str]:
    ''' A messy training loop with lots of extraneous logging functionality '''
    
    # Create Logging Directory for Tensorboard
    # now = time.mktime(datetime.datetime.now().timetuple()) - 1550000000
    # log_dir = f'{log_dir} ({now})/'
    # logger = Logger(log_dir)
    # print(f'Training model. Logging to: "{log_dir}"\n')

    model = model.to(device) # Send model to GPU if possible
    model.train() # Set model to training mode
    
    # for drawing predictions to images
    # renderer = DrawChessPosition(delimiter='-')
    
    def validate_model(model, overall_step, loader=None, val=False):
        accu = test_model(model, loader, criterion, 
                                       print_guess=False, 
                                       disable_tqdm=disable_tqdm)
        if not val: return accu
        else:
            info = { 'validation_accuracy': accu }
            # for key, value in info.items():
            #     logger.scalar_summary(key, value, overall_step)
        return accu

    total_step = len(train_loader)
    validate_model(model, overall_step=0, loader=val_loader, val=True)
    for epoch in range(num_epochs):
        if print_guess: print(f'Epoch {epoch+1}')
        running_loss = 0
        
        
        # Tqdm will create a progress bar
        with tqdm(total=len(train_loader), 
                  desc=f'Epoch {epoch+1}', 
                  unit=' minibatches',
                  disable=(print_guess or disable_tqdm)) as pbar:
            
            # Iterate through minibatches
            for step, (images, labels, original_imgs) in enumerate(train_loader):
                images, labels = images.to(device), labels.long().to(device)

                output = model(images).to(device)
                _,class_labels = torch.max(labels,2) 
                _, argmax = torch.max(output, 2)

                accuracy = float((class_labels == 
                                  argmax.squeeze()).float().mean().cpu())

                loss = criterion(output.reshape(10*64,13).float(),
                                 class_labels.reshape(10*64))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())

                pbar.set_postfix(training_accuracy=accuracy, loss=loss.item(), refresh=True)
                pbar.update(1)

                if step % log_freq == 0:
                    overall_step = epoch*total_step + step
                    
                    info = { 'loss': loss.item(), 'accuracy': accuracy }

                    for key, value in info.items():
                        logger.scalar_summary(key, value, overall_step)
                    

                    info = { f'{fen_from_64(argmax.cpu()[0])}': 
                                    [original_imgs[0].cpu()]}

                    for tag, images in info.items():
                        logger.image_summary(tag, images, overall_step)

                    for key, value in model.named_parameters():
                        key = key.replace('.', '/')
                        logger.histo_summary(key, 
                                             value.data.cpu().numpy(), 
                                             overall_step)
                        try:
                            logger.histo_summary(key+'/grad', 
                                                 value.grad.data.cpu().numpy(),
                                                 overall_step)
                        except (AttributeError):
                            # During transfer learning some of the variables 
                            # don't have grads
                            pass

                if print_guess and step % print_guess_freq == 0:
                    overall_step = epoch*total_step + step
                    print(f"\n{60*'-'}\nBatch Number: {overall_step}")
                    print(f"Example training point:")
                    print(f"Actual: {fen_from_64(class_labels.cpu()[0])}")
                    print(f"Guess: {fen_from_64(argmax.cpu()[0])}")
                    print(f"Example Accuracy: {float((class_labels[0] == argmax[0]).float().mean().cpu())}")

                    # board_actual = renderer.draw(fen_from_64(class_labels.cpu()[0]))
                    # board_guess = renderer.draw(fen_from_64(argmax.cpu()[0]))
                    # renderer.show_side_by_side(board1= original_imgs[0],
                    #                            board2=board_guess, 
                    #                            board1_title='Actual (Preprocessed)',
                    #                            board2_title='Prediction (Re'
                    #                                         'ndered to image)')
        
        if not suppress_output:   
            print(f"{epoch}: Training loss: {running_loss/len(train_loader)}")
            print(f"{epoch}: Training accuracy: {accuracy}")
        
        if test_model_after_each_epoch:
            validate_model(model, overall_step=overall_step, loader=val_loader, val=True)
        

    final_val_acc = validate_model(model, overall_step=overall_step, loader=val_loader, val=True)
    final_train_acc = validate_model(model, overall_step=overall_step, loader=train_loader)
    final_test_acc = validate_model(model, overall_step=overall_step, loader=test_loader)
    return (model, log_dir, final_train_acc, final_val_acc, final_test_acc)

def test_model(model: nn.Module, 
                test_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.modules.Module,
                print_guess: bool=False,
                print_guess_freq: int=50,
                suppress_output: bool=True,
                disable_tqdm: bool=False,) -> float:
    
    model = model.to(device)
    accuracies = []
    losses = []
    total_step = len(test_loader)
        
    # for drawing predictions to images
    # renderer = DrawChessPosition(delimiter='-')
    with torch.no_grad():
        # Tqdm will create a progress bar
        with tqdm(total=len(test_loader), 
                desc=f'Test Batches', 
                unit=' minibatches',
                disable=(print_guess or disable_tqdm)) as pbar:

            # Iterate through minibatches
            for step, (images, labels, original_imgs) in enumerate(test_loader):
                images, labels = images.to(device), labels.long().to(device)

                output = model(images).to(device)
                _,class_labels = torch.max(labels,2) 
                _, argmax = torch.max(output, 2)

                accuracy = float((class_labels == 
                                argmax.squeeze()).float().mean().cpu())

                loss = criterion(output.reshape(10*64,13).float(),
                                class_labels.reshape(10*64))
                losses.append(float(loss.item()))
                accuracies.append(accuracy)

                pbar.set_postfix(test_acc=accuracy, test_loss=loss.item(), refresh=True)
                pbar.update(1)

                if print_guess and step % print_guess_freq == 0:

                    overall_step = total_step + step
                    print(f"\n{60*'-'}\nTest Batch Number: {overall_step}")
                    print(f"Example testing point:")
                    print(f"Actual: {fen_from_64(class_labels.cpu()[0])}")
                    print(f"Guess: {fen_from_64(argmax.cpu()[0])}")
                    print(f"Example Accuracy: {float((class_labels[0] == argmax[0]).float().mean().cpu())}")

                    # board_actual = renderer.draw(fen_from_64(class_labels.cpu()[0]))
                    # board_guess = renderer.draw(fen_from_64(argmax.cpu()[0]))
                    # renderer.show_side_by_side(board1= original_imgs[0],
                    #                         board2=board_guess, 
                    #                         board1_title='Actual',
                    #                         board2_title='Prediction (Re'
                    #                                         'ndered to image)')

    if not suppress_output:                   
        print(f'\nAvg. Accuracy of the network on test images: {np.average(accuracies)}')
        print(f'Avg. Loss of the network on test images: {np.average(losses)}')

    return np.average(accuracies)

## 
## Start of main
## 

np.warnings.filterwarnings('ignore') # they were getting annoying...

num_epochs = 25
log_freq=2
log_dirs = []
cnns = [CNN_BatchNormLessFiltersLastLayer, CNN_BatchNormLessFilters, CNN_NoDropout, 
        CNN_Dropout, CNN_Dropout_BatchNorm, CNN_BatchNorm]
basic_models = [FullyConnected, LogisticRegression]
all_models = cnns + basic_models
new_models = [CNN_LessFilters, CNN_BatchNormLessFilters]

for learning_rate in [.0005]:
    for model_type in new_models:
        net = model_type(batch_size=BATCH_SIZE)
        print(f'Training: {net.name}\nLearning Rate: {learning_rate}')

        if net.name == 'LogisticRegression (L2 regularization)':
            weight_decay=.05 # add L2 regularizer for logreg
        else: weight_decay=0
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        log_dir = f'./logs/{net.name}_lr{learning_rate}'
        criterion = nn.CrossEntropyLoss().to(device)

        # print a summary of the net statistics
        #         summary(net.to(device), (BATCH_SIZE*32, 3, 25, 25))

        #     images, labels, original_imgs = next(iter(train_loader))
        #     y = net.to(device)(Variable(images.to(device)))
        #     make_dot(y)

        # Run the model
        start_time = time.time()


        model, log_dir, final_train_acc, final_val_acc, final_test_acc = train_model(net,
                                                    log_dir,
                                                    train_loader,
                                                    criterion,
                                                    optimizer,
                                                    num_epochs, 
                                                    log_freq,
                                                    print_guess_freq=100,
                                                    print_guess=False,
                                                    test_model_after_each_epoch=True,
                                                    val_loader=val_loader,
                                                    test_loader=test_loader,
                                                    disable_tqdm=True) # or print_guess=False for tqdm
        log_dirs.append(log_dir)
        elapsed_time = time.time() - start_time
        
        print(f'Final Train Accuracy: {final_train_acc:.8f}')
        print(f'Final Test Accuracy: {final_test_acc:.8f}')
        print(f'Final Validation Accuracy: {final_val_acc:.8f}')
        print(f'Elapsed Time: {elapsed_time/60:.2f} minutes')
        save_model(model, f'model_{model.name}_{learning_rate}.pt')
 
