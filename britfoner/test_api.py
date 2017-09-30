import sure

sure.enable()  # stops pycharm from removing sure import
from britfoner.api import pronounce


def test_gives_pronunciations_of_word_in_dictionary():
    #
    pronounce('row').should.eql({('ɹ', 'əʊ'), ('ɹ', 'aʊ')})


def test_gives_pronunciation_of_word_not_in_dictionary():
    #
    pronounce('thrones').should.eql({('θ', 'ɹ', 'əʊ', 'n', 'z')})


def test_gives_no_pronunciations_for_words_longer_than_18_chars():
    #
    pronounce('counterrevolutionaries').should.eql(set())
