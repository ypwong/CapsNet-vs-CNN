# CapsNet vs CNN

## How to Use

1) Generate the dataset with 
    ```
    python data_generator.py
    ```
    Each set of dataset has their own configuration files which can be found in the `dataset_factory` folder. You can make any changes there.
2) If the above is successful, there will be a folder (e.g. `generated_dataset`) appear in the main directory. The folder should contain another 2 folders namely `feature_level` and `object_level`. Inside each of these folders, the dataset would be separated into two folders for testing and training. Any chosen model will be first trained and tested on the feature level datasets and then the convolutional layers will be freezed before trained and tested on the object level dataset (transfer learning).
3) If the above steps are done without any error, you can now choose a model to train. You can do so by
```
python eval.py --model [CHOSEN MODEL] --data_folder [FOLDER WHERE THE DATASET IS LOCATED]
```
4) You can also supply other parameters as arguments if you wish. Check the available parameters with 
```
python eval.py -h
```
5) The script trains the model on the feature-level dataset first. It'll choose the model with the highest accuracy to be loaded for object-level training with its conv layers freezed.  At every epoch, the loss and accuracy will be logged out for you to view.


License
----

MIT

