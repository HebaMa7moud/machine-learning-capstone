**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Inventory Monitoring at Distribution Centers


## Project: Inventory Monitoring at Distribution Centers

Inventory Monitoring at Distribution Centers project is about the robots used to move objects at those centers, where those objects are carried in bins which can contain multiple objects. The gool of this project is to build a model that can count the number of objects in each bin. 
To build this project you AWS SageMaker is used to fetch data from a database, preprocess it, and then train a machine learning model. 


## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.


Setting up the environment for building the project and preparing your data for training your models as follow:

1.Set up AWS by opening it through the classroom and open sagemaker studio and create a folder for the project.

2-Download the Starter Files by cloning Github RIPO (https://github.com/udacity/nd009t-capstone-starter) and uploading starter files to workspace.

3-Downloading training data and arranges it in subfolders. Each of these subfolders contain images where the number of objects is equal to the name of the folder.  The downloaded data is a small subset of   Amazon Bin Images Dataset  which is used as recommended ,  preprocessing data by splitting it randomly and divide it into training, validation and testing subsets and uploading those subsets to S3 bucket.

4-Installing the necessary packages for this project's execution: smdebug, jinja2, sagemaker,boto3, torchvision,PIL, numpy,matplotlib.pyplot, tqdm, uuid

## Files used in this project:

The following are the files used in the project:

1- sagemakder.ipynb: Jupyter notebook used to install packages for project's execution, fetch data, define hyperparameters ranges to finetune a pretrained model with      hyperparameter tuning, extract best hyperparameters, train model with best hyperparameters, create profiler and debugger reports, deploy models and query the          endpoint.

2- train.py: Python training script used to train the model and to perform hyperparameter tuning.

3- debug_model.py: Python training script that is trained using best hyperparameters and used to perform model profiling and debugging.

4- infernce.py: Python script that implements the following functions to get a prediction: model_fn function that calls the loaded model, input_fn function to process    input and and predict_fn function to customize how the model server gets predictions from the loaded model.

5- benchmarak_model.ipynb: Jupyter notebook used to install packages for project's execution, fetch data, train benchmark model by running training_a_cnn_solution.py      script. 

6- training_a_cnn_solution.py: Python training script used to train the model and to perform hyperparameter tuning.


## Dataset

### Overview
**TODO**: Explain about the data you are using and where you got it from.

Amazon Bin Image Dataset (https://registry.opendata.aws/amazon-bin-imagery/) is the used dataset for this project.
Amazon Bin Image Dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. 

Amazon Bin Image Datasets are available in the aft-vbi-pds S3 bucket in the us-east-1 AWS Region 


### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it

Over 500,000 bin JPEG images and corresponding JSON metadata files describing items in the bin are available in the aft-vbi-pds S3 bucket in the us-east-1 AWS Region.

This data is downloaded using the following code:
![code to download data](https://user-images.githubusercontent.com/81697137/233710766-309b079b-7ae6-4c62-839d-a7a0f83da2e7.png)


This code  is used to download the data and arranged  it into subfolders. Each of these subfolders contain images where the number of objects is equal to the name of the folder.


## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

A pretrained model (Resnet50 Model) is used to execute classification task. Resnet50 Model is one of large neural network architecture models,where this type of models show excellent performance when dealing with image classification. 

Types of hyperparameters used in training model are: batch-size, epochs, learning rate.
why chosen?

Evaluate model performance:

Testing Loss = 1.5138607222688965

Test Accuracy = 28.9356 %

The dataset used to train the model is small so that model accuracy is low.


## Machine Learning Pipeline

1- Upload Training Data:
        1- download a small subset of Amazon Bin Image Dataset.
        
        2- split the downloaded subset randomly over each individual object class and divide it into; 80% for training set, 5% for validation set and 15% for testing               set.   
        
        3- uploading train, valid and test datasets to S3 bucket.
  
2- Model Training Script:

      train.py script is created to be used to train a pretrained model with fixed hyperparameters.


3- Train in SageMaker: install necessary dependencies and set up training estimator and use SageMaker to run that training script and   train the model.

4- Perform a prediction on the model:

     1- Deploy the model to an endpoint.
     
     2- test the deployed endpoint using random object images.



## Standout Suggestions

## Hyperparameter Tuning:
train.py scipt is trained again using the following ranges for the hyperparameters to finetune the model:

hyperparameter_ranges= {
                        "lr": ContinuousParameter(0.001, 0.1),

                        "epochs":IntegerParameter(1,5),
                        
                        "batch-size":CategoricalParameter([32, 64, 128, 256, 512])}
                        
Best hyperparameters:
![best hp1](https://user-images.githubusercontent.com/81697137/233723705-b9adb8dd-eefd-4e65-bedc-a410a3a510b4.png)

![best hp](https://user-images.githubusercontent.com/81697137/233722097-04dd898e-92ac-46fd-b965-d09eb089c6a6.png)


## Model Profiling and Debugging:

Model debugging and profiling is implemented through the following steps:

1-Import SMDebug framework class.

2-Set the SMDebug hook for the test phase.

3-Set the SMDebug hook for the training phase.

4-Set the SMDebug hook for the validation phase.

5-Register the SMDebug hook to save output tensors.

6-Pass the SMDebug hook to the train and test functions.

model debugging output:

![debug output](https://user-images.githubusercontent.com/81697137/233722341-befb421e-ac49-43aa-8b71-fb209f221ba4.png)

Oservation on debugging output:
It was observed that at some point testing loss is higher than training loss, this is due to the difference in the amount of images of each class. Images belong to class 1 may be the reason since it's the lowest amount of data, so small training set means high testing loss.

This behaviour could be fixed by using large datasets to train the model, but we are committed to the recommended subset in the project.

- System merics during model debugging and profiling training job:
- 
  CPU utilization most of the time after the begining is flat at around 50% so the system isn't overwhelmed.
  ![cpu utilization debug job](https://user-images.githubusercontent.com/81697137/233729652-caeb7239-bfa8-4702-8083-db0c86fc51ff.png)




## Model Deploying and Querying:

A python script is created to implement a specific functions to get a predictions, Net function, the model_fn function(calls the loaded model saved model.pth after retraining train.py script using finetuned parameters), input_fn function (process the image/url uploaded to the endpoint into an object for prediction) and predict_fn function (Takes the object and performs inference against the loaded model).
The deployed endpoint:
![endpoint screenshot](https://user-images.githubusercontent.com/81697137/233731803-a123d216-071e-46dc-900f-82f065fd4fb1.png)


Quering the deployed endpoint by running an prediction on the endpoint using random object images using the following code:
![predict code](https://user-images.githubusercontent.com/81697137/233732881-0c7e04ea-358e-46f0-a8f9-bdd6d81572c7.png)

The prediction results:
![20](https://user-images.githubusercontent.com/81697137/233733893-baa7486e-2868-4d90-a676-626b8bd0c659.png)
![21](https://user-images.githubusercontent.com/81697137/233733940-5f5ee447-2512-4539-8e05-b86fb5ba1595.png)
![22](https://user-images.githubusercontent.com/81697137/233733990-25281f84-4bc9-4f7b-acce-43688d1c3f4d.png)
![23](https://user-images.githubusercontent.com/81697137/233734025-6ff98f4f-ed7a-472a-8c7d-99f36101078b.png)
![5](https://user-images.githubusercontent.com/81697137/233734146-7e146c2b-1174-4fc1-abed-773b0ece55d0.png)
![4](https://user-images.githubusercontent.com/81697137/233738703-1bbd9916-90e7-453a-8ede-d2d98667a694.png)




## Reduce Costs:
COST ANALYSIS:


SPOT INSTANCE:

- Spot instance is used to lower the cost but using it is very risky where it low the cost but lower reliability, it might be turned off at any time.

- To train the model using a spot instance the following parameters should be identified:

   .create a unique checkpoint id name for chekpoint s3 path incase spot instance is interrupted so checkpoint can be used to resume from the last inrerruption.
   
   .set spot instance arguments; 1- use_spot_instances= True.
   
                                 2- max_run time.
                                 
                                 3- max_wait time which is the amount of time willing to wait for Spot infrastructure to become available, it must be
                                    equal or greater than max_run for spot instance is available.
                                    


- The training job ended with status 'Stopped' rather than 'Completed' this is normal in spot instance case as mentioned earlier.
. At the end of trained job Managed Spot Training savings was 26.8% which is counted as follow:(1- billable seconds/Training seconds)*100
 




