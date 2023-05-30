# Music_noise_reduction

## Contents
* [Authors]
* [Introduction]
* [Dataset Description]
* [Solution Description]
* [Requirements]
* [Instalation] 

## Authors
| Organization   | Name | Email | 
|----------|-------------|-------------|
| PUJ-Bogota | Sebastián Pineda| juanspineda@javeriana.edu.co|
| PUJ-Bogota  |  Daniel Duque | daniel_duque@javeriana.edu.co |

## Introduction
Deep Learning Experimentation for training an autoencoder model capable of removing noise features from given audios

## Dataset Description

We used the following datasets:
* MusicNet (https://www.kaggle.com/datasets/imsparsh/musicnet-dataset)

* Microsoft Scalable noisy speech Dataset (https://github.com/microsoft/MS-SNSD)



## Solution Description 

On the src folder you will find the notebooks used for building autoencoders:

* src\autoencoder1.ipynb : autoencoders trained on pure wav sequence
* src\autoencoder2.ipynb : autoencoders trained on mel spectrograms
* src\noise_adding.ipynb : notebook for overlaying noise audio on clean audio samples
* src\transformers.py : sample code for training transformers based on mel spectrograms


## Requirements

Basic reference of which libraries and versions were used

Package                       Version
----------------------------- --------------------
absl-py                       1.4.0
aiohttp                       3.8.4
aiosignal                     1.3.1
alabaster                     0.7.12
alembic                       1.9.3
anaconda-client               1.11.0
anaconda-navigator            2.3.1
anaconda-project              0.11.1
aniso8601                     9.0.1
anyio                         3.5.0
appdirs                       1.4.4
argon2-cffi                   21.3.0
argon2-cffi-bindings          21.2.0
arrow                         1.2.2
astroid                       2.11.7
astropy                       5.1
astunparse                    1.6.3
async-timeout                 4.0.2
asyncer                       0.0.2
atomicwrites                  1.4.0
attrs                         21.4.0
Automat                       20.2.0
autopep8                      1.6.0
Babel                         2.9.1
backcall                      0.2.0
backports.functools-lru-cache 1.6.4
backports.tempfile            1.0
backports.weakref             1.0.post1
bcrypt                        3.2.0
beautifulsoup4                4.11.1
binaryornot                   0.4.4
bitarray                      2.5.1
bkcharts                      0.2
black                         22.6.0
bleach                        4.1.0
bokeh                         2.4.3
boto3                         1.24.28
botocore                      1.27.28
Bottleneck                    1.3.5
brotlipy                      0.7.0
cachetools                    5.3.0
certifi                       2022.9.14
cffi                          1.15.1
chardet                       4.0.0
charset-normalizer            2.0.4
click                         8.0.4
cloudpickle                   2.2.1
clyent                        1.2.2
colorama                      0.4.5
colorcet                      3.0.0
coloredlogs                   15.0.1
comtypes                      1.1.10
conda                         23.1.0
conda-build                   3.22.0
conda-content-trust           0.1.3
conda-pack                    0.6.0
conda-package-handling        1.9.0
conda-repo-cli                1.0.20
conda-token                   0.4.0
conda-verify                  3.4.2
constantly                    15.1.0
contourpy                     1.0.7
cookiecutter                  1.7.3
coverage                      7.2.5
cryptography                  37.0.1
cssselect                     1.1.0
cycler                        0.11.0
Cython                        0.29.32
cytoolz                       0.11.0
daal4py                       2021.6.0
dask                          2022.7.0
databricks-cli                0.17.4
datashader                    0.14.1
datashape                     0.5.4
debugpy                       1.5.1
decorator                     5.1.1
defusedxml                    0.7.1
diff-match-patch              20200713
dill                          0.3.4
distributed                   2022.7.0
docker                        6.0.1
docutils                      0.18.1
entrypoints                   0.4
et-xmlfile                    1.1.0
Faker                         18.9.0
fastapi                       0.95.2
fastjsonschema                2.16.2
filelock                      3.6.0
filetype                      1.2.0
flake8                        4.0.1
Flask                         2.2.2
Flask-JWT                     0.2.0
Flask-JWT-Extended            4.2.3
Flask-RESTful                 0.3.9
Flask-SQLAlchemy              2.5.1
flatbuffers                   1.12
fonttools                     4.38.0
frozenlist                    1.3.3
fsspec                        2022.7.1
future                        0.18.2
gast                          0.4.0
gensim                        4.1.2
gitdb                         4.0.10
GitPython                     3.1.30
glob2                         0.7
google-auth                   2.18.1
google-auth-oauthlib          1.0.0
google-pasta                  0.2.0
greenlet                      2.0.2
grpcio                        1.54.2
h11                           0.14.0
h5py                          3.7.0
HeapDict                      1.0.1
holoviews                     1.15.0
humanfriendly                 10.0
hvplot                        0.8.0
hyperlink                     21.0.0
idna                          3.3
imagecodecs                   2021.8.26
ImageHash                     4.3.1
imageio                       2.26.1
imagesize                     1.4.1
importlib-metadata            5.2.0
incremental                   21.3.0
inflection                    0.5.1
iniconfig                     1.1.1
intake                        0.6.5
intervaltree                  3.1.0
ipykernel                     6.15.2
ipython                       7.31.1
ipython-genutils              0.2.0
ipywidgets                    7.6.5
isort                         5.9.3
itemadapter                   0.3.0
itemloaders                   1.0.4
itsdangerous                  2.1.2
jax                           0.4.10
jdcal                         1.4.1
jedi                          0.18.1
jellyfish                     0.9.0
Jinja2                        3.1.2
jinja2-time                   0.2.0
jmespath                      0.10.0
joblib                        1.2.0
json5                         0.9.6
jsonschema                    4.16.0
jupyter                       1.0.0
jupyter_client                7.3.4
jupyter-console               6.4.3
jupyter_core                  4.11.1
jupyter-server                1.18.1
jupyterlab                    3.4.4
jupyterlab-pygments           0.1.2
jupyterlab-server             2.10.3
jupyterlab-widgets            1.0.0
keras                         2.12.0
keyring                       23.4.0
kiwisolver                    1.4.4
lazy_loader                   0.2
lazy-object-proxy             1.6.0
libarchive-c                  2.9
libclang                      16.0.0
llvmlite                      0.38.0
locket                        1.0.0
lxml                          4.9.1
lz4                           3.1.3
Mako                          1.2.4
Markdown                      3.3.4
MarkupSafe                    2.1.2
matplotlib                    3.6.3
matplotlib-inline             0.1.6
mccabe                        0.6.1
menuinst                      1.4.19
mido                          1.2.10
mistune                       0.8.4
mkl-fft                       1.3.1
mkl-random                    1.2.2
mkl-service                   2.4.0
ml-dtypes                     0.1.0
mlflow                        2.1.1
mock                          4.0.3
mpmath                        1.2.1
msgpack                       1.0.3
multidict                     6.0.4
multipledispatch              0.6.0
munkres                       1.1.4
mypy-extensions               0.4.3
navigator-updater             0.3.0
nbclassic                     0.3.5
nbclient                      0.5.13
nbconvert                     6.4.4
nbformat                      5.5.0
nest-asyncio                  1.5.5
networkx                      3.0
nltk                          3.7
nose                          1.3.7
notebook                      6.4.12
numba                         0.55.1
numexpr                       2.8.3
numpy                         1.23.5
numpydoc                      1.4.0
oauthlib                      3.2.2
olefile                       0.46
onnxruntime                   1.14.1
opencv-contrib-python         4.7.0.72
opencv-python-headless        4.7.0.72
openpyxl                      3.0.10
opt-einsum                    3.3.0
packaging                     21.3
pandas                        1.5.3
pandocfilters                 1.5.0
panel                         0.13.1
param                         1.12.0
paramiko                      2.8.1
parsel                        1.6.0
parso                         0.8.3
partd                         1.2.0
pathlib                       1.0.1
pathspec                      0.9.0
patsy                         0.5.2
pep8                          1.7.1
pexpect                       4.8.0
pickleshare                   0.7.5
Pillow                        9.5.0
pip                           23.1.2
pkginfo                       1.8.2
platformdirs                  2.5.2
plotly                        5.10.0
pluggy                        1.0.0
pooch                         1.7.0
poyo                          0.5.0
prometheus-client             0.14.1
prompt-toolkit                3.0.20
Protego                       0.1.16
protobuf                      3.19.6
psutil                        5.9.0
ptyprocess                    0.7.0
py                            1.11.0
pyarrow                       10.0.1
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycodestyle                   2.8.0
pycosat                       0.6.3
pycparser                     2.21
pyct                          0.4.8
pycurl                        7.45.1
pydantic                      1.10.7
PyDispatcher                  2.0.5
pydocstyle                    6.1.1
pyerfa                        2.0.0
pyflakes                      2.4.0
Pygments                      2.11.2
PyHamcrest                    2.0.2
PyJWT                         2.4.0
pylint                        2.14.5
pyls-spyder                   0.4.0
PyMatting                     1.1.8
PyNaCl                        1.5.0
pyodbc                        4.0.34
pyOpenSSL                     22.0.0
pyparsing                     3.0.9
pyreadline3                   3.4.1
pyrsistent                    0.18.0
PySocks                       1.7.1
pytest                        7.1.2
python-dateutil               2.8.2
python-lsp-black              1.0.0
python-lsp-jsonrpc            1.0.0
python-lsp-server             1.3.3
python-multipart              0.0.6
python-slugify                5.0.2
python-snappy                 0.6.0
pytz                          2022.1
pyviz-comms                   2.0.2
PyWavelets                    1.4.1
pywin32                       302
pywin32-ctypes                0.2.0
pywinpty                      2.0.2
PyYAML                        6.0
pyzmq                         23.2.0
QDarkStyle                    3.0.2
qstylizer                     0.1.10
QtAwesome                     1.0.3
qtconsole                     5.2.2
QtPy                          2.2.0
queuelib                      1.5.0
regex                         2022.7.9
rembg                         2.0.37
requests                      2.28.1
requests-file                 1.5.1
requests-oauthlib             1.3.1
rope                          0.22.0
rsa                           4.9
Rtree                         0.9.7
ruamel.yaml                   0.17.21
ruamel.yaml.clib              0.2.6
ruamel-yaml-conda             0.15.100
s3transfer                    0.6.0
scikit-image                  0.19.2
scikit-learn                  1.2.1
scikit-learn-intelex          2021.20221004.171935
scipy                         1.10.1
Scrapy                        2.6.2
seaborn                       0.11.2
Send2Trash                    1.8.0
service-identity              18.1.0
setuptools                    59.8.0
shap                          0.41.0
sip                           4.19.13
six                           1.16.0
smart-open                    5.2.1
sniffio                       1.2.0
snowballstemmer               2.2.0
sortedcollections             2.1.0
sortedcontainers              2.4.0
soupsieve                     2.3.1
Sphinx                        5.0.2
sphinxcontrib-applehelp       1.0.2
sphinxcontrib-devhelp         1.0.2
sphinxcontrib-htmlhelp        2.0.0
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.3
sphinxcontrib-serializinghtml 1.1.5
spyder                        5.2.2
spyder-kernels                2.2.1
SQLAlchemy                    1.4.46
starlette                     0.27.0
statsmodels                   0.13.2
sympy                         1.10.1
tables                        3.6.1
tabulate                      0.8.10
TBB                           0.2
tblib                         1.7.0
tenacity                      8.0.1
tensorboard                   2.12.3
tensorboard-data-server       0.6.1
tensorboard-plugin-wit        1.8.1
tensorflow                    2.9.0
tensorflow-estimator          2.12.0
tensorflow-intel              2.12.0
tensorflow-io-gcs-filesystem  0.31.0
termcolor                     2.3.0
terminado                     0.13.1
testpath                      0.6.0
text-unidecode                1.3
textdistance                  4.2.1
threadpoolctl                 2.2.0
three-merge                   0.1.1
tifffile                      2023.3.21
tinycss                       0.4
tldextract                    3.2.0
toml                          0.10.2
tomli                         2.0.1
tomlkit                       0.11.1
toolz                         0.11.2
tornado                       6.1
tqdm                          4.64.1
traitlets                     5.1.1
ttkbootstrap                  1.10.1
Twisted                       22.2.0
twisted-iocpsupport           1.0.2
typing_extensions             4.3.0
ujson                         5.4.0
Unidecode                     1.2.0
urllib3                       1.26.11
uvicorn                       0.22.0
w3lib                         1.21.0
watchdog                      3.0.0
wcwidth                       0.2.5
webencodings                  0.5.1
websocket-client              0.58.0
Werkzeug                      2.0.3
wheel                         0.37.1
widgetsnbextension            3.5.2
win-inet-pton                 1.1.0
win-unicode-console           0.5
wincertstore                  0.2
wrapt                         1.14.1
xarray                        0.20.1
xlrd                          2.0.1
XlsxWriter                    3.0.3
xlwings                       0.27.15
yapf                          0.31.0
yarl                          1.9.2
zict                          2.1.0
zipp                          3.15.0
zope.interface                5.4.0

## Instalation 

* To create enviroment on conda:

conda create --name <env> --file requirements.txt

* To create enviroment using pip

If you want a file which you can use to create a pip virtual environment (i.e. a requirements.txt in the right format) you can install pip within the conda environment, then use pip to create requirements.txt.
<br>
<br>
conda activate <env>
conda install pip
pip freeze > requirements.txt
<br>
<br>
Then use the resulting requirements.txt to create a pip virtual environment:

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt


Some usefull links:
  https://www.tensorflow.org/io/tutorials/audio
  https://towardsdatascience.com/audio-ai-isolating-instruments-from-stereo-music-using-convolutional-neural-networks-584ababf69de
  https://www.kaggle.com/datasets/imsparsh/musicnet-dataset
  https://www.tensorflow.org/tutorials/audio/simple_audio
