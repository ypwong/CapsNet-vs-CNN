# CapsNet vs CNN 

This experiment is to test the ability of both CapsNet and CNN on the retention of spatial relationship between learned features. There are 2 kinds of dataset. One dataset consists of images of rectangles vs triangles. The other consists of images of arrows vs non-arrows (composed using rectangles and triangles). The idea here is to train the models on the first dataset first (triangles vs rectangles). Once the training is done, the convolutional layers should be freezed and the same model should now be re-trained on the second dataset. 

### Steps
1) Generate dataset with `python gen_data/gen_data.py`.
2) Start the training/testing process with `python train.py --mode CNN --freeze_conv False`
		-	This should be done two times. The first time is with a model of your selection and False as your parameter for `--freeze_conv`. The second time is with the same selected model with True as your parameter for `--freeze_conv`. 

The result of the training/testing process will be saved as graphs and confusion matrix in `figures` folder. Use `--help`  on `train.py` or `gen_data.py` to get more information about the optional parameters. 

## License
MIT
