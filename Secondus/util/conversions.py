import numpy as np
import re
from skimage.util.shape import view_as_blocks
from skimage import transform as sktransform
from skimage import io
import os

def FEN_to_twod_char_array(string_rep: str, delimiter: str='-') -> np.array:

    board_2d_arr = np.zeros((8,8)).astype(str)
    for i,row in enumerate(string_rep.split(delimiter)):
        j=0
        for c in row:
            if str.isalpha(c):
                board_2d_arr[i][j] = c
                j+=1
            elif str.isdigit(c):
                for k in range(j,j+int(c)):
                    j=k
                    board_2d_arr[i][j] = delimiter
                    j+=1
            else:
                raise ValueError('Bad Forsyth-Edwards Notation')
        assert j==8, f'Not enough characters in row {i}, string: "{row}"'
    assert i==7, f'Not enough rows on board: "{string_rep}"'
    return board_2d_arr


def twod_char_array_to_FEN(board_2d_arr: np.array, null_char: str='-') -> str:
    string_rep = ''
    for i,row in enumerate(board_2d_arr):
        blank_counter = 0
        for j,c in enumerate(row):
            if str.isalpha(c):
                if blank_counter > 0:
                    string_rep += str(blank_counter)
                blank_counter = 0
                string_rep += c
            elif c == null_char:
                blank_counter += 1
            else:
                raise ValueError(f'Illegal Character: {c}')
        if blank_counter > 0:
                string_rep += str(blank_counter)
        if i < 7:
            string_rep += null_char
        assert j==7, f'Not enough pieces in row {j} {i}, string: "{row}"'
    assert i==7, f'Not enough rows on board.'
    return string_rep

def onehot_from_fen(fen, piece_symbols = 'prbnkqPRBNKQ'):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for c in fen:
        if(c in '12345678'):
            output = np.append(
              output, np.tile(eye[12], (int(c), 1)), axis=0)
        elif str.isalpha(c):
            idx = piece_symbols.index(c)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)
        else:
            raise ValueError('Bad Forsyth-Edwards Notation')
    if np.shape(output) != (64, 13):
        raise ValueError(f'Invalid Forsyth-Edwards Notation—board shape: '
                         f'{np.shape(output)}')
    return output

def fen_from_onehot(one_hot, piece_symbols = 'prbnkqPRBNKQ'):
    output = ''

    if np.shape(one_hot) != (64, 13):
        raise ValueError(f'Invalid one hot encoding shape: '
                         f'{np.shape(one_hot)}')
    for i in range(64):
        if(np.argmax(one_hot[i]) == 12):
            output += 'blank'
        else:
            output += piece_symbols[np.argmax(one_hot[i])]
        if(i % 8 - 7 == 0 and i != 63):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace('blank' * i, str(i))

    return output


def fen_from_64(one_hot, piece_symbols = 'prbnkqPRBNKQ'):
    output = ''
    for i in range(64):

        if(one_hot[i] == 12):
            output += ' '
        else:
            output += piece_symbols[one_hot[i]]
        if(i % 8 - 7 == 0 and i != 63):
            output += '-'
    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output


def fen_from_filename(fname):
    base = os.path.basename(fname)
    return os.path.splitext(base)[0]

def process_image(img, downsample_size = 200):
    square_size = int(downsample_size/8)
    img = sktransform.resize(io.imread(img), 
                                  (downsample_size, downsample_size), 
                                  mode='constant')
    tiles = view_as_blocks(img, block_shape=(square_size, 
                                                  square_size, 
                                                  3)).squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3), img  