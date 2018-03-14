# Introduction
Building convolutional neural networks for the analysis of x-ray images. This analysis is based entirely on a dataset of x-ray images provided by the NIH. For more information on the dataset, visit https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community.

# Getting Started
The dataset used for this project has been provided by the U.S. National Institute of Health. All data used for this project can be downloaded from the at https://nihcc.app.box.com/v/ChestXray-NIHCC.
In order for this project to work properly, these images need to be downloaded and extracted into the images directory of this project. Additionally, the Data_Entry_2017.csv file should be downloaded and added to the resources directory of this project.

Once this has been done, simply run the convx_utils.py file in the sources directory. This will preprocess the data from the data entry csv and restructure the images into the following directory structure...

<pre>
images
|-----train
|        |-----healthy
|        +-----unhealthy
|-----test
|        |-----healthy
|        +-----unhealthy
|-----validation
|        |-----healthy
|        +-----unhealthy
</pre>

# Running the Code
Once this is done, the analysis code can be run. You can find this code, along with helpful descriptions, in the convx.ipynb file located in the source directory.

