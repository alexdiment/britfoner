import sure
sure.enable() # stops pycharm from removing sure import
from numpy import array, ndarray
from britfoner import _UNSTRESSED_BRITFONE, Index, _END, _GAP, _START, Inv_Alphabet, Alphabet
from britfoner.IO import items_from, index_from, decoded, all_encoded


def test_reads_in_csv_as_sorted_tuples():
    #
    words, sounds = items_from(_UNSTRESSED_BRITFONE)

    words[0].should.eql(("'", 'C', 'O', 'S'))
    sounds[0].should.eql(('k','ə','z'))


def test_builds_index_from_items():
    words = [('A', 'B', 'C'), ('C', 'D')]
    sounds = [('x', 'y'), ('x', 'z')]

    true_index = Index \
        (4 + 3,
         3 + 2,
         3 + 3,
         2 + 2,
         {_START: 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, _END: 5, _GAP: 6},
         (_START, 'A', 'B', 'C', 'D', _END, _GAP),
         {_START: 0, 'x': 1, 'y': 2, 'z': 3, _END: 4, _GAP: 5},
         (_START, 'x', 'y', 'z', _END, _GAP),
         {words[0]: {sounds[0]}, words[1]: {sounds[1]}})

    index_from(words, sounds).should.eql(true_index)


def test_decodes_matrix_into_sequence():
    mat: ndarray = array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    inv_alphabet: Inv_Alphabet = ['A', 'B', 'C']

    decoded(mat, inv_alphabet).should.eql(('B', 'A', 'C'))
    decoded(mat, inv_alphabet, reverse=True).should.eql(('C', 'A', 'B'))


def test_encodes_sequences_into_3D_tensor():
    seqs = [tuple('CAB'), tuple('BAC')]
    alphabet: Alphabet = {'A': 0, 'B': 1, 'C': 2}

    tensor: ndarray = array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]])
    rev_tensor: ndarray = array([[[0, 1, 0], [1, 0, 0], [0, 0, 1]],[[0, 0, 1], [1, 0, 0], [0, 1, 0]]])

    (all_encoded(seqs, alphabet) == tensor).all().should.eql(True)
    (all_encoded(seqs, alphabet, reverse=True) == rev_tensor).all().should.eql(True)
