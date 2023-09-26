# Version 0.1
First model scheme added with using tensorflow.keras with GPU support. First training phase has been concluded with batch size 16 and epoch of 10.
Also first image of the validation dataset is selected and it's label predicted by model then plotted. It was false.
First Scores are: 
- Accuracy: 0.4375
- Precision: 0.45625000000000004
- Recall: 0.4375
- F1 Score: 0.4431818181818181
- It is actually not good for starting, but i'm aware that i have started with very simple structure.

# Version 0.2
Model structure revised from 3 convolutional layers to 4. Also dense layer size multiplied by 4. Sample image plotting mechanism switched to random 4 images.
- Accuracy: 0.5625
- Precision: 0.7479166666666668
- Recall: 0.5625
- F1 Score: 0.5568181818181818
Bugs: 
- There are no individual representative for 2 classes in the validation set. That's why confusion matrix seems unfinished.
- More complex approach can be done for model.

# Version 0.3
New data loading module added. It uses stratifying train_test_split method for getting at least a sample from each class. Because, our dataset is highly imbalanced and our training phase was affected by this problem.
Visualizing methods also revised. Now we are randomly taking 4 images from next batch with considering they have also unique labels.
Training now be done with the batch size 16 and 25 epochs with the train-valid ratio of 75:25.
- Accuracy: 0.625
- Precision: 0.9166666666666666
- Recall: 0.625
- F1 Score: 0.6571199633699633
The only main problem here is, even if we do stratifying the

# Version 0.4
Additional 1 conv23(128) layer, Dense Layer(256) replaced with 512.
- Accuracy: 0.6875
- Precision: 0.71875
- Recall: 0.6875
- F1 Score: 0.6613095238095239
It performed slightly better but we need some additional changes.

# Version 0.5
1 Conv2D((128,3,3)) layer changed with (256,3,3) and added extra batch norm. , max pool. and dropout. Also, Added additional Dense(128) connected layer. 
Train-Valid ratio changed to 7:3
- Accuracy: 0.6875
- Precision: 0.7678571428571428
- Recall: 0.6875
- F1 Score: 0.6875
