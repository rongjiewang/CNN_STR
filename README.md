# CNN_STR
Utilizing Convolutional Neural Network (CNN) to classify Short Tandem Repeat (STR) sequences.
## Install
This is a step by step instruction for installing the CNN-STR for python 3.7*.
### Requirements for python modules & versions
* tensorflow  == 2.1.0 
* keras == 2.3.1 
* biopython == 1.76
* numpy == 1.18.4  
### Command to install
    pip install -r requirements.txt

##Training
Training the model with train dataset, saved the model parameters to  "saved_model" folder.  Command as follow:

    python train_model.py
    
 ## Testing
Testing the model with saved training model with test dataset, output the accuracy for test dataset.  Command as follow:
   
    python test_model.py
    
## Contact
If you have any question, please contact the author rjwang.hit@gmail.com
