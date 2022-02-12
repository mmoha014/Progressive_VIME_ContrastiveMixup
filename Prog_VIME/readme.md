# Progressive VIME

Checkout the requirements.txt file, if you have conda pre-installed cd into the directory where you have downloaded the source code and run the following:

```
conda create -n prog_vime python==3.7.10
conda activate prog_vime

pip install -r requirements.txt
```
To run experiments they are launched from the train-trafVio.py file. The source code for Traffic Violation dataset is provided. To run **Progressive Vime** on the dataset, 
use the following command:

```
python train-trafVio.py
```
