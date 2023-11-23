# dl-trade-model-excerpts

Contains most of the code involved in hyperparameter tuning of the deep learning models. The entire process is run by executing model_selector.py. The main steps are:
1)	Selecting a model architecture from hypermodels.py (not included)
2)	Loading the dataset (or creating or new one) with intraday_data.py. 
3)	Setting up the GPUs using the MultiWorkerMirroredStrategy distribution strategy. Currently configured for computer with two RTX A4500 GPUs.
4)	Shuffling the dataset (NumPy arrays at this point) and separating features from labels.
5)	Defining KerasTuner’s Bayesian Optimization tuner.
6)	Run the search. For each trial:
a.	 Build model (and a duplicate Monte Carlo version that has permanently-enabled dropout layers) with hyperparameters specified by tuner.
b.	Define custom-built callbacks (early stopping, learning rate scheduler, learning rate and loss plotters) found in custom_callbacks.py.
c.	Wrap dataset by tf.data.Dataset objects (done at the start of every trial since the batch size is a hyperparameter).
d.	Run custom training loop found in custom_loop.py.
e.	Save the more profitable model (and Monte Carlo duplicate) weights if the model is profitable with respect to the validation set at the end of any epoch.

Training analysis:
-	Metrics are logged at the end of each epoch and plotted using TensorBoard.
-	Several batches in each trial are profiled for detailed performance information and the trace.
  
Refer to the Wiki for additional information.

