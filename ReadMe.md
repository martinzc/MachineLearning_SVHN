Deep convolutional neural network

Team: 
Finished_in_TH

Team members: 
Chao Zhou (cz16)
Donnie Kim (dk17)

Language implemented in:	
Python 3.5

Framework used:
Anaconda	
Google Tensorflow

Environment:
Windows 10

GPU Requirement:
Should be able to run CUDA 8.0

List of dependencies:
Components for GPU
	CUDA 8.0
	cuDNN 5.1
Python Packages:	
	numpy
	matplotlib
	pickle
	pandas
	scipy
	random
	sklearn
	tensorflow-gpu

Run details:

0. To run a batch file in windows, go to the directory in command and type the name of the script. For example, if you want to run train_model.bat, simply type "train_model" after you have directed into the directory.

1. Set up dependencies
	a. If Python 3.5.x is not installed on your machine, install it from https://www.python.org/downloads/release/python-352/
	
	b. If the NVIDIA GPU graphics card driver is not installed, please find your model and install the driver: http://www.nvidia.com/Download/index.aspx

	c. In order to run Tensorflow-gpu, the user have to have 2 essential components for utilizing NVIDIA Graphics card: CUDA 8.0 and cuDNN 5.1. If you have not installed the two of them, here's the instruction:

		c.1) If the machine does not have Visual Studio, please download from this link: \url{https://www.visualstudio.com/downloads/}. Community version should suffice our purpose. Once downloaded, click the .exe file and follow the instructions.

		c.2) For CUDA 8.0, follow the link https://developer.nvidia.com/cuda-downloads, click windows, and select the appropriate version, and select local. Once downloaded, run the executable file and follow the instructions. There are steps one can follow to verify whether the installation is correctly done, which can be found in this website: http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#axzz4edgBQEyk.

		c.3) For cuDNN 5.1, follow the link here: https://developer.nvidia.com/rdp/cudnn-download. You will have to make an account to download. Agree to the terms, and select Download cuDNN v5.1 (Jan20, 2017), for CUDA 8.0. Once downloaded and unzipped, you will see three folders: bin, include, and lib.  
			- Copy "cudnn64_5.dll" in "bin" folder into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
			- Copy "cudnn.h" in "include" folder into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include
			- Copy "cudnn.lib" in "lib" folder into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64

	d. Install Anaconda on your machine. Pip install has a lot of troubles in windows. For example scipy doesn't install correctly with pip. Thus, we Choose to use conda install for all our dependencies. To install Anaconda, go to https://www.continuum.io/downloads. Once installed, with the Anaconda app, create a new environment by clicking the "Create" button in the lower left, right next to three other buttons. Make sure you select Python 3.5. Let's set the name of the environment to "grader_env"
	
	e. Activate that environment by typing "activate grader_env" in the command line. Then navigate to the folder where our code is stored.
	
	f. Once the above steps are done, run the script set_up.bat to install all the dependencies for this project. This will install numpy, matplotlib, pickle, pandas, scipy, random, sklearn, and tensorflow-gpu.

	
2. Pre-process data
	
You should already have six different files: cropped_extra_lowerleft.csv, cropped_extra_lowerright.csv, cropped_extra_upperleft.csv, 
cropped_extra_upperright.csv, cropped_extra_middle.csv, and cropped_test.csv. However, if you would like to recreate these files, you could do so by running the script preprocess_data.bat to preprocess the data. Before you do so, make sure you have extra.csv and test.csv and place it in the same folder as the script that you are running. This script executes CropImage.py and CropImage_test.py and if successful, will output the six cropped files mentioned above.

	
3. Training the model
	
Run the script "train_model.bat" to train the model. This script loads up train_svhn.py and will train the five cropped extra files on our cnn model. 
Because we are using emsemble method, it saves five different models inside the folder saved_model. The name of the model can be changed at line 25 in the file train_ensemble_model.py. It is currently set to "ChaoZhou_DonnieKim_SVHN_CNN_Ensemble_model".

4. Using pre-trained model to output result
	
Our pretrained model that yields the result that we submit to Kaggle is located in the folder saved_models. Run the script "predict_result.bat" to feed cropped_test.csv into the model and generate result. Our currently pretrained model is named "svhn_April19_2Layer2Conv_ensemble" and there are actually five of them, but the code will do the indexing for you. You could also change the model name at line 25 at predict_ensemble.py to load up different models. In the end, the code would generate a csv file called "results_dk17_cz16.csv". Open the file and add ImageId to the first entry so that it conforms with the Kaggle format before submitting it. 

5. Extra code

The code visualize data, if run, would create an image that shows the first image in a csv file. This is how we visualize the data at the first place and what inspired us to crop the image.