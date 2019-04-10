from setuptools import setup

setup(
    name='britfoner',
    version='1.0.2',
    description='British English pronunciation with Keras+Tensorflow',
    author='Jose Llarena',
    author_email='jose.llarena@gmail.com',
    url='https://github.com/josellarena/britfoner',
    license='MIT',
    install_requires=['tensorflow', 'h5py', 'keras', 'scikit-learn'],
    packages=['britfoner','britfoner.recurrentshop', 'britfoner.seq2seq', 'britfoner.recurrentshop.backend'],
    package_data={'britfoner': ['britfone.main.no-stress.2.0.1.csv', '20x32x256x19x48x1.h5']}
)
