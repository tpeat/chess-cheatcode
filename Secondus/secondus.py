import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from matplotlib import image
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam
from keras.layers import Dropout
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Convolution2D
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam
from keras.models import model_from_json
# Created by: Trent Conley
# This program will train a convolutional neural net to recognize chess pieces. If this project were to be continued,
# I would try to train the CNN on images of a chessboard that were taken at an angle instead of top-down. This would
# be more realistic, but I tried to do this when I first started off and got an abysmal test accuracy at 0.07, or 7%.
# If more time were allowed, I would also consider making a new dataset, given the one that I copied off Kaggle was full
# of errors and requried manual pruning (which took more than an hour and I ended up removing about 400 images manually).
# However, it was worth it given that the final accuracy was higher than .97. Since there were many errors in the
# dataset that I caught, I would assume that there are many more. Thus, by creating my own dataset, I could push the
# accuracy to 1. I used many different python files and folders with images to create this end result, but for convience
# I pushed the folders onto github and included the other files as functions in this file.
# The dataset that I used can be found here: https://github.com/samryan18/chess-dataset.git
# Accuracy after 50 epochs: 0.9711019396781921


loaded_images = list()

# An FEN is how people who play chess describe where every chess piece is on the board. The dataset that I used had
# images labled with an FEN. So, I converted out of the FEN and make a sort of virtual chessboard, that is a 8x8 2-D
# array. I created a dictionary that held the values for each piece. The capitol letters are white pieces and lowercase
# black.
def get_FEN(test_FEN):
    # I assigned '_' to 0 because the people who created this dataset messed up and changed some of the data.
    # Setting that character equal to 0 allows my function to run smoothly and accuratly convert the FEN to a 2-D array
    # This was most likley human error.
    dict= {'_': 0, 'p':1, 'b':2, 'n':3, 'r':4, 'k':5, 'q':6, 'P':7, 'B':8, 'N':9, 'R':10, 'K':11, 'Q':12}
    # at image 8200
    # a blank will be 0, so there are 13 classes for any given square
    rows = 8
    cols = 8
    arr = np.zeros((rows,cols),dtype=int)
    work_str = test_FEN
    if work_str.find('_') != -1:
        work_str = work_str[:work_str.find('_')]
    else:
        work_str = work_str[:work_str.find('.')]
    # now we just have the FEN
    for row in range(0, 8):
        part = work_str[:work_str.find('-')]
        if part == '':
            part = work_str
        col = 0
        for j in range(0, len(part)):
            if part[j].isdigit():
                col = col + int(part[j])
            else:
                arr[row][col] = dict[part[j]]
                col = col + 1
        work_str = work_str[work_str.find('-') + 1:]
    return arr


# I used this method to convert the overly-large images provided by the kaggle dataset
def convert_to_size(dir, size):
    for filename in listdir(dir):
        pic = Image.open(dir + '/' + filename)
        pic.thumbnail((size, size))
        pic.save(dir + '/' + filename)

# convert_to_size('labeled_preprocessed', 800)
# This fills a folder with pictures of singular pieces on squares. This way, I do not have to rely on multi-label
# classification. Once it finishes, there will be about 32,000 images to train and test on. Thoes images will have a
# 40x40 resolution and be in color.
def fill_folder(dir, save_fol, size = 320):
    square_size = size/8
    index = 0
    for filename in listdir(dir):
        if filename != '.DS_Store':
            pic = Image.open(dir + '/' + filename)
            arr = get_FEN(filename[:filename.find('.png')])
            for row in range (0, 8):
                for col in range (0, 8):
                    pic_crop = pic.cromp((square_size*row, (square_size*col), square_size+(square_size*row),
                                         square_size+(square_size*col)))
                    pic_crop.save(save_fol+ '/' + str(arr[col][row]) + '-' + str(index) + '.png')
                    index = index +1
            index = index + 1
        print (index)
# fill_folder('chess-dataset/labeled_preprocessed', 'single_squares_2')


# Python3 program to rotate a matrix by 90 degrees
# I copied this from https://www.geeksforgeeks.org/inplace-rotate-square-matrix-by-90-degrees/


# An Inplace function to rotate
# N x N matrix by 90 degrees in
# anti-clockwise direction
def rotateMatrix(mat, N = 8):
    # Consider all squares one by one
    for x in range(0, int(N / 2)):

        # Consider elements in group
        # of 4 in current square
        for y in range(x, N - x - 1):
            # store current cell in temp variable
            temp = mat[x][y]

            # move values from right to top
            mat[x][y] = mat[y][N - 1 - x]

            # move values from bottom to right
            mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y]

            # move values from left to bottom
            mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x]

            # assign temp to left
            mat[N - 1 - y][x] = temp

        # Function to print the matrix


def displayMatrix(mat, N = 8):
    dict = {0: '-', 1: 'p', 2: 'b', 3: 'n', 4: 'r', 5: 'k', 6: 'q', 7: 'P', 8: 'B', 9: 'N', 10: 'R', 11: 'K',
            12: 'Q'}
    for i in range(0, N):
        mystr = ''
        for j in range(0, N):
            # TypeError "tensor is unhashable" --> use .ref() instead, need to change dict keys then
            # tensor_ref = mat[i][7-j].ref()
            mystr = mystr + str(dict[mat[i][7-j]])
        print (mystr)
        print ("")


def getBoard(pic, model, size2 = 320):
    square_size = size2 / 8
    output = [[0] * 8] * 8
    # chessboard = [[0]*8]*8
    # Here I load in the images from the folder that I previously filled.
    pic = Image.open(pic)
    count = 0
    mylist = []
    for row in range(0, 8):
        for col in range(0, 8):
            pic_crop = pic.crop((square_size * row, (square_size * col), square_size + (square_size * row),
                                 square_size + (square_size * col)))
            name = 'dump/' + 'temp.png'
            pic_crop.save(name)
            pic_data = image.imread(name)
            for file in os.listdir('dump'):
                if file.endswith('.png'):
                    os.remove('dump/' + file)
                    # pic_data = pic_data[np.newaxis, ...]

            # predict_classes is outdated: predict_step returns tensors: predict??
            # model.predict returns the probability of beign each of the classes
            probabilities = model.predict(np.asarray([pic_data]))
            pred = np.argmax(probabilities, axis=1)
            output[row][col] = pred
            count = count + 1
        # print(output)
        templist = []
        for f in range(0, 8):
            templist.append(output[row][f][0])
        mylist.append(templist)
    for i in range(0, 3):
        rotateMatrix(mylist)
    return mylist


def to_FEN(pic, model, size1 = 320, move = 'w', castle = '-'):
    dict = {0: '-', 1: 'p', 2: 'b', 3: 'n', 4: 'r', 5: 'k', 6: 'q', 7: 'P', 8: 'B', 9: 'N', 10: 'R', 11: 'K',
            12: 'Q'}
    board = getBoard(pic, model, size2 = size1)
    displayMatrix(board)



    FEN = ''
    for i in range (0, 8):
        count = 0
        for j in range(0, 8):
            key = dict[board[i][7- j]]
            if key == '-':
                count = count + 1
            else:
                if count == 0:
                    FEN = FEN + key
                else:
                    FEN = FEN + str(count) + key
                    count = 0
            if j == 7 and count > 0:
                FEN = FEN + str(count)
        FEN = FEN + '/'
    FEN = FEN[:-1]
    FEN = FEN + ' ' + move + ' ' + castle + ' - 0 1'
    return FEN

# # sunfish



def get_label(foo):
    label = foo[:2]
    if str(label[1]) == '-':
        label = label[0]
    return label

# serialize model to JSON
def save_model(model, file_name= 'model3'):
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name + ".h5")
    print("Saved model to disk")
    
    # pickle this instead

def load_model(json = 'model.json', h5 = 'model.h5'):
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(h5)
    print("Loaded model from disk")
    return model

def run(build = False, high_res = False, json = 'model.json', h5 = 'model.h5', size = 320, boardfolder = 'board_to_analyze', dir = 'equal_pic', skip = True):
    if not skip and not build:
        loaded_images = list()
        output = np.array([])
        # Here I load in the images from the folder that I previously filled.
        for filename in listdir(dir):
            if filename != '.DS_Store':
                img_data = image.imread(dir + '/' + filename)
                output = np.append(output, [int(get_label(filename))])
                # store loaded image
                loaded_images.append(img_data)
        all_images = np.array(loaded_images)

        # Reshaped data so that I can work with it as a 2-D array
        output = output.reshape(len(output), 1)

        # Here I splice the data into test and train.
        length = int(len(output)*0.8)
        y_train = output[:length]
        y_test = output[length:]
        x_train = all_images[:length]
        x_test = all_images[length:]


        # Preparing the data:
        print("Scaling input data...")
        max_val = np.max(x_train).astype(np.float32)
        print("Max value: " +  str(max_val))
        x_train = x_train.astype(np.float32) / max_val
        x_test = x_test.astype(np.float32) / max_val
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

        # Convert class vectors to binary class matrices.
        # print ((y_train))
        num_classes = len(np.unique(y_train))
        # if not build:
        #     num_classes = 13
        print("Number of classes in this dataset: " + str(num_classes))
        if num_classes > 2:
            print("One hot encoding targets...")
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

        print("Original input shape: " + str(x_train.shape[1:]))

        if build:
            
            # make this a function

            ### Second, build a model:
            model = Sequential()

            model.add(Conv2D(48, (3, 3), padding='same',
                             input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            if high_res:
                model.add(Conv2D(96, (8, 8)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.15))

            model.add(Conv2D(48, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))
        else:
            model = load_model(json, h5)
        model.summary()

        mloss = 'categorical_crossentropy'
        opt = RMSprop()

        model.compile(loss=mloss,
                      optimizer=opt,
                      metrics=['accuracy'])

        if build:
            ### Fourth, train and test the model!
            epochs = 50
            history = model.fit(x_train, y_train,
                                epochs=epochs,
                                verbose=2,
                                validation_data=(x_test, y_test),
                                shuffle=True)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('\nTest accuracy:', score[1])
    else:
        model = load_model(json, h5)
        mloss = 'categorical_crossentropy'
        opt = RMSprop()

        model.compile(loss=mloss,
                      optimizer=opt,
                      metrics=['accuracy'])
    if not build:
        for filename in listdir(boardfolder):
            if filename != '.DS_Store':
                print(to_FEN(boardfolder + '/' + filename, model, size1 = size))
    else:
        save_model(model)
# run('single_squares', True)




def get_least_occuring(dir):
    count = np.array([0] * 13)
    for filename in listdir(dir):
        if filename != '.DS_Store':
            label = get_label(filename)
            count[int(label)] = count[int(label)] + 1
    return count.min()

def same_num_pieces(dir, save_fol):
    # i think i need to increase the resolution of the pictures and add more layers to the CNN because we have less
    # inputs
    count = np.array([0] * 13)
    smallest = get_least_occuring(dir)
    for filename in listdir(dir):
        if filename != '.DS_Store':
            label = get_label(filename)
            if count[int(label)] < smallest:
                count[int(label)] = count[int(label)] + 1
                pic = Image.open(dir + '/' + filename)
                pic2 = pic
                pic2.save(save_fol + '/' + filename)

# same_num_pieces('single_squares_2', 'equal_pic2')
# I am filling a folder with a higher images of the boards so that i will have higher resolution images
# I figured i need an accuracy of at least .99 for the individual pieces on this new folder because that means
# that only 50% of chess boards (assuming that they are equally distrubited by all types of pieces (including blanks))
# will be accuratly predicted.
# the equation for this is f(x) = accuracy^64, since there are 64 slots for pieces. .99^64 roughly equals 50%.
# Ideally, I would like to push the accuracy to .9999, since that yields a 99% chance of me predicing the board
# right wich i will live happily with. I will note that all of these numbers are a little high given that on a real
# chess board most of the spaces will be blanks, so my accuary will end up being much higher.
# When I ran with the lower resolution images in the equal pic 2 folder, i got a test accuracy of 0.95112. my train
# accuracy was 0.9974 which highly suggest that I was over training the model, given that the test accuracy went down
# as i increased the epochs.
# I will ask the teacher to find out how to resolve this issue.

# fill_folder('board_to_analyze2', 3000)


# run('equal_pic2', True)


# something is wrong with the shape. for some reason the model that I saved has a shape of 10.
# same_num_pieces('single_squares_2', 'equal_pic')
run(json = 'model2.json', h5 = 'model2.h5')
# run('single_squares_2', False, False, 'model2.json', 'model2.h5')


#plz dont crash

