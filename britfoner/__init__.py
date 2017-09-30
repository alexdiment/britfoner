import logging
import os
from collections import namedtuple
from os.path import dirname, join, realpath

from typing import Tuple, Dict


# stops warnings about unused performance libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

_UNSTRESSED_BRITFONE = join(dirname(realpath(__file__)), 'britfone.main.no-stress.2.0.1.csv')
_MODEL_OUT = dirname(realpath(__file__))

_START, _END, _GAP = '*', '¬', '·'
_symbols = {_GAP, _START, _END}
_PREFIX, _SUFFIX = (_START,), (_END,)

Seq = Tuple[str, ...]
Alphabet = Dict[str, int]
Inv_Alphabet = Tuple[str, ...]
Index = namedtuple('Index', ['x_dim', 'x_n', 'y_dim', 'y_n',
                             'letter', 'inv_letter',
                             'phone', 'inv_phone',
                             'word_to_sounds'])
