# Human body recognition

This project aims to track an operator while he is working to determine the distance he travels and the time he needs for every operation on the machine.

## How to setup the environment and install everything to get started

First, I recommand you to have a Linux computer or virtual machine. If you don't, you won't be able to use tracking from yolo because of some problems with "lap" library on windows... I think that there is a way to install lap on windows but I wasn't able to find it, that's why I swiched to Linux with "Oracle VM VirtualBox".

### 1) Download repository 

Open a terminal in the folder where you want to download the project, and type the following command:

```bash
git clone https://TOKEN@github.com/mateorinaldi/tri-courrier.git
```

Then type `cd XXXX` to enter the folder.

### 2) Installation of Poetry

Once **Python3.10** is installed (and added to the PATH), type the following command to install `poetry` :
```bash
pip3.10 install poetry
```
> `poetry` is a Python dependency manager for virtualizing environments.

### 3) Install dependencies

#### Python librairies

Start by setting poetry so that the virtual environment is created in the project file:

```bash
poetry config virtualenvs.in-project true
```

Then change the path to python so that poetry uses python 3.10 (replace <python_path.exe> with your own path to the python 3.10 executable):

```bash
poetry env use <python_path.exe>
```

Launch the virtual environment with the command :

```bash
poetry shell
```

Then, to **install dependencies**, run the following command:

```bash
poetry install
```

After this, it is possible that some dependencies are not able to install properly so run this command (you have to be in the virtual environment when typing this command):

```bash
pip3.10 install -r requirements.txt
```

You can now run and modify main.py program as you want !

