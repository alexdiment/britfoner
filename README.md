
<h1>britfoner<br/></h1><br/>

<p>automated pronunciation for British English with Keras+Tensorflow</p><br/>

<p>
  <a href="https://travis-ci.org/JoseLlarena/britfoner">
    <img alt="Travis Status" src="https://travis-ci.org/JoseLlarena/britfoner.svg?branch=master">
  </a>
<p><br/>

_Britfoner_ is an api for translating English words to their phonetic form (in British English). It uses
phonetic dictionary [_Britfone_](https://github.com/JoseLlarena/Britfone) as a first lookup and a Keras+Tensorflow
grapheme-to-phoneme converter (trained also  on _Britfone_) as a backup.

_Britfoner_ incorporates code from [seq2seq](https://github.com/farizrahman4u/seq2seq) and [recurrentshop](https://github.com/farizrahman4u/recurrentshop)

_Britfoner_ is limited to words 18 characters and less.


[Further details](#more-background)

## Install


__Windows__

Download and install [Miniconda](https://conda.io/miniconda.html)

Then create a virtual environment

```shell
mkdir gp2
cd gp2
conda create -n g2p python=3.6
```

activate it:

```shell
activate g2p
```
install _britfoner_ plus dependencies:
 ```shell
(g2p) conda install h5py scikit-learn 
(g2p) pip install git+https://github.com/JoseLlarena/britfoner.git
```

__Linux__


Download and install [python 3.6](https://www.python.org/downloads/release/python-362/)

create a virtual environment

```shell
jose@jose-dev:~/projects$ mkdir g2p && cd gp2 && python3 -m venv env
jose@jose-dev:~/projects/g2p$ source activate env
(env) jose@jose-dev:~/projects/g2p$ 
```

activate it:

```shell
jose@jose-dev:~/projects/g2p$ source activate env
(env) jose@jose-dev:~/projects/g2p$ 
```
install _britfoner_ with dependencies:
 ```shell
(env) jose@jose-dev:~/projects/g2p$ pip install git+https://github.com/JoseLlarena/britfoner.git
```
## Usage


```shell
(env) jose@jose-dev:~/projects/g2p$ python -c "import britfoner.api as api; print(api.pronounce('success'))"
Using TensorFlow backend.
{('s', 'ə', 'k', 's', 'ɛ', 's')}
 ```
 
[Full API documentation](https://josellarena.github.io/britfoner/index.html)

## <a name="more-background"></a>Background

_Britfoner_ maps English words to their pronunciations as per the [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet).
It looks up a word (matching `[A-Za-z ']+`) in [_Britfone_](https://github.com/JoseLlarena/Britfone), a British English
pronunciation dictionary, returning the possible pronunciations. If the word is not found, it uses a Keras Deep Learning
(with a Tensorflow backend) model as a backup.

The model is a [Sequence to Sequence model with Attention](http://arxiv.org/abs/1409.0473), with a single-layered 256-hidden-unit bidirectional-encoder and decoder.
It was trained on 16,042 unaligned word-pronunciation pairs, using 163 pairs for validation (99%/1% split), for 220 epochs (1 1/2 hours on a 
i5+GTX1050 GPU), with the Adam
optimiser, learning rate 10<sup>-2</sup>, decay 10<sup>-5</sup> and 10% dropout. The output is unnormalised, the loss is mean squared error and
the output is decoded with a greedy strategy. The final word error rate was 15.95%.  
 
 
## Changelog

see [Changelog](https://github.com/JoseLlarena/britfoner/blob/master/CHANGELOG.md)

## License

GPL2 @ [Jose Llarena](https://www.npmjs.com/~josellarena)