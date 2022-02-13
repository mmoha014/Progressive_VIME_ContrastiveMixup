# Progressive Contrastive Mixup
Checkout the requirements.txt file, if you have conda pre-installed cd into the directory where you have downloaded the source code and run the following:

```
conda create -n Progressive_ContrMixup python==3.8.1
conda activate Progressive_ContrMixup

pip install -r requirements.txt
```

The current configuration uses combination of _ProjectionHead+Decoder + Update_ (Table 6) in paper. To change the combination of the components change the config file in configs/TrafficVio/contrastivemixup.json (loss_args). Also, the dataset should be copied in _data_ folder

To run experiment it is launched from the train.py file, run the following command:
```
python train.py
```

We do not use early stopping for runs. The table in the paper shows  the best accuracy among all runs.
