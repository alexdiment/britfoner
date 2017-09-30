'''

Functions to convert input/output data into tensors and viceversa

'''
from codecs import open
from collections import defaultdict
from os.path import join
from typing import Iterable, List, Dict, Set, Tuple

import numpy as np
from keras.engine.training import Model
from numpy import zeros, ndarray, argmax
from .seq2seq.models import AttentionSeq2Seq
from sklearn.model_selection import train_test_split

from britfoner import Seq, Alphabet, Inv_Alphabet, Index, _GAP, _symbols, _PREFIX, _SUFFIX, _MODEL_OUT


def dataset_from(src: str, val_size: float = .05, random_state: int = 42) \
        -> Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Index]:
    '''
    Creates a dataset read in from the the given file name. The dataset
    consists of rain/test input-output tensors plus an index

    :param src: the file name with the data
    :param val_size: the proportion in [0, 1] of data points used for validation
    :param random_state: the seed for picking the validation set
    :return: a dataset consisting of tensors and index, as a tuple
    '''
    words, sounds = items_from(src)

    index = index_from(words, sounds)

    X = all_encoded(padded(words), index.letter, reverse=True)

    Y = all_encoded(padded(sounds), index.phone)

    return train_test_split(X, Y, test_size=val_size, random_state=random_state), index


def items_from(src: str) -> Tuple[Iterable[Seq], Iterable[Seq]]:
    '''
    Reads a sequence of inputs and outputs from a file
    The file's format should be a line per data point, the input sequence made up
    of symbols with no separator, followed by a comma, followed by the white-space
    separated output sequence

    :param src: the data file
    :return: a tuple with the input and output sequences
    '''
    with open(src, 'r', 'utf-8') as in_file:
        return zip(*[to_tuple(entry)  # remove length condition
                     for entry in in_file
                     if entry.strip()])


def index_from(words: Iterable[Seq], sounds: Iterable[Seq]) -> Index:
    '''
    Constructs an index given the input and output sequences

    An index contains a number of datastructures necessary to encode and decode
    the data

    :param words: the input sequences
    :param sounds: the output sequences
    :return: an Index
    '''
    word_to_sounds = defaultdict(set)
    letters, phones = set(_symbols), set(_symbols)

    for word, sound in zip(words, sounds):
        letters.update(word)
        phones.update(sound)
        word_to_sounds[word].add(sound)

    inv_letter_index = tuple(sorted(letters))
    letter_index = {letter: idx for idx, letter in enumerate(inv_letter_index)}
    inv_phone_index = tuple(sorted(phones))
    phone_index = {phone: idx for idx, phone in enumerate(inv_phone_index)}

    return Index(len(letter_index), len(max(words, key=len)) + 2,
                 len(phone_index), len(max(sounds, key=len)) + 2,
                 letter_index, inv_letter_index,
                 phone_index, inv_phone_index,
                 word_to_sounds)


def padded(seqs: Iterable[Seq]) -> List[Seq]:
    '''
    Pads all sequences to the length of the longest

    :param seqs: the raw sequences
    :return: the padded sequences
    '''
    max_length = len(max(seqs, key=len))

    return [bounded(seq, max_length) for seq in seqs]


def all_encoded(seqs: List[Seq], alphabet: Alphabet, reverse=False) -> ndarray:
    '''
    Encodes a list of strings into a tensor

    :param seqs: alist of sequences as string tuples
    :param alphabet: a mapping from character to index
    :param reverse: true if the sequences should be encoded in reverse
    :return: a 3-D tensor as a 3-D numpy array index by sequence, position and vector component
    '''

    X = zeros((len(seqs), len(seqs[0]), len(alphabet)), dtype=np.bool)

    for i, seq in enumerate(seqs):
        for t, phone in enumerate(seq[::-1] if reverse else seq):
            X[i, t, alphabet[phone]] = 1

    return X


def decoded(seq_vec: ndarray, inv_alphabet: Inv_Alphabet, reverse=False) -> Seq:
    '''
    Decodes a matrix representing a sequence into a tuple

    :param seq_vec: sequence as 2D numpy array
    :param inv_alphabet: sorted alphabet
    :param reverse: True if the string should be reversed
    :return: the decoded sequence as a tuple of strings
    '''
    dec = tuple(inv_alphabet[argmax(vec)] for vec in seq_vec
                if inv_alphabet[argmax(vec)] not in _symbols)

    return dec[::-1] if reverse else dec


def padding_for(width: int, max_width: int) -> Seq:
    '''
    Builds padding for sequences longer than :const:`MAX_LENGTH`

    :param width: length of sequence
    :param max_width:
    :return: tuple containing 0 to ``max_width`` repetitions of the padding symbol
    '''

    return (_GAP,) * (max_width - width)


def bounded(seq: Seq, max_length: int):
    '''
    Bounds sequence to help model understand where the it starts and where it ends. Addionally
    it pads the sequence if its longer than :py:const:`MAX_LENGTH`

    :param seq: sequence to be bounded
    :param max_length: maximum sequence length
    :return: a sequence with a prefix a suffix and a padding at the end
    '''
    return _PREFIX + seq + _SUFFIX + padding_for(len(seq), max_length)


def dictionary_from(src: str) -> Dict[Seq, Set[Seq]]:
    '''
    #
    Returns a mapping from word to pronunciation(s)

    :param src: file to read the dictionary data from
    :return: a map of words to their pronunciations as Dict[Seq, Set[Seq]]
    '''
    word_to_sounds = defaultdict(set)

    with open(src, 'r', 'utf-8') as in_file:

        for entry in in_file:

            if not entry.strip(): continue

            word, sound = to_tuple(entry)
            word_to_sounds[word].add(sound)

    return word_to_sounds


def indexes_from(dictionary: Dict[Seq, Set[Seq]]) -> Tuple[Alphabet, Inv_Alphabet]:
    '''
    #
    builds input and inverted output indexes for converting from/to text to vectors

    :param dictionary: a word to proununciations mapping
    :return: an input and an inverted output index as a tuple
    '''
    letters, phones = set(_symbols), set(_symbols)

    for word, sounds in dictionary.items():

        letters.update(word)

        for sound in sounds:
            phones.update(sound)

    return {letter: idx for idx, letter in enumerate(sorted(letters))}, tuple(sorted(phones))


def model_from(src: str) -> Model:
    '''
    #
    loads sequence to sequence model from file
    :param src: model file name
    :return: a model
    '''
    # uses model file name to set appropiate model parameters before loading model weights,
    # this is workaround for a defect in seq2seq that prevents reading the whole model
    input_length, input_dim, hidden_n, output_length, output_dim, depth = map(int, src.split('.h5')[0].split('x'))

    model = AttentionSeq2Seq(output_dim=output_dim,
                             output_length=output_length,
                             hidden_dim=hidden_n,
                             input_dim=input_dim,
                             input_length=input_length,
                             unroll=False,
                             depth=depth)

    model.load_weights(join(_MODEL_OUT, src))

    return model


def to_tuple(entry: str) -> Tuple[Seq, Seq]:
    '''
    #
    Converts a string containing an input, output sequence pair into a tuple of seqeunces
    :param entry: the string containing the entry
    :return: a tuple containing an input and an output sequence
    '''
    return tuple(entry.split(',')[0].split('(')[0]), tuple(entry.split(',')[1].split())
