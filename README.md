This projects aims at classifying clothes among 9 classes using a convolutional neural network (CNN).

The implementation is all done using Google's deep learning library TensorFlow.

Prior to running the scripts, please make sure that you have installed the `requirements.txt` within a `virtualenv`.

Also, set up accordingly the paths of the Tianchi dataset in `Loader.py`, `eval.py` and `app.py`.

`preprocess` folder contains the source files used to preprocess the data (from RGB to grayscale and normalisation step).

In order to train the scripts, type `python cnn/eval.py`.

In order to evaluate the latest trained model, type `python cnn/train.py`.

Trained models are saved in the model/checkpoints folder. The hyper_params file contains the hyper-parameters used to train the model saved.

The summaries used by Tensorboard are saved in model/logs/summaries folder.

In order to visualise the results in Tensorboard, type within the TensorFlow environnement, `tensorboard --logdir=model/logs/summaries`.

The file cnn/app.py is a small app that restore the model stored in model/checkpoints and test the data contained in test_app/images/.