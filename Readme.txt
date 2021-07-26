COURSE PROJECT : CS626
#####################################################################################

Team members:
==============

Name				Roll No.

Akshay Batheja			203059011
Shivam Ratnakant Mhaskar 	20305R002

######################################################################################

Project Title:
==============

Automatic Emoji Recommendation System

Project Description:
====================

To recommend emojis based on the sentiment of the tweet text. 

########################################################################################

Steps to execute the project:
=============================

To execute CNN :

1. Go to directory code/CNN:
	
	cd code/CNN

2. Pip Install the requirements.txt:
	
	pip install -r requirements.txt
	
3. You can directly run the cnn.py file or create its object and then call the functions specified in the code documentation

	python cnn.py

4. It will show all the evaluation metrics for the respective model
5. At the end it will ask for the input of the sentence for testing of the system. Just enter a sentence and it will output the recommended emojis for the input sentence.


To execute CNN-LSTM :
======================

1. Go to directory code/CNN:
	
	cd code/CNN-LSTM

2. Pip Install the requirements.txt:
	
	pip install -r requirements.txt
	
3. You can directly run the cnn.py file or create its object and then call the functions specified in the code documentation

	python cnnlstm.py

4. It will show all the evaluation metrics for the respective model
5. At the end it will ask for the input of the sentence for testing of the system. Just enter a sentence and it will output the recommended emojis for the input sentence.


To execute Logistic Regression :
===============================

1. Go to directory code/CNN:
	
	cd code/Logistic \Regression

2. Pip Install the requirements.txt:
	
	pip install -r requirements.txt
	
3. You can directly run the cnn.py file or create its object and then call the functions specified in the code documentation

	python LR.py

4. It will show all the evaluation metrics for the respective model
5. At the end it will ask for the input of the sentence for testing of the system. Just enter a sentence and it will output the recommended emojis for the input sentence.




=============================================

Steps to get UI Working:

1. Download model and required folders from following link:

https://drive.google.com/file/d/1drAM6cCCcSEcrzaXdMUKAGYaQHv5qy2O/view?usp=sharing
2. Install django: pip install django
3. Unzip the downloaded folder into UI/AERS/emoji directory:
4. Got to dir: UI/AERS/emoji and install requirements 
	pip install -r requirements.txt 
5. Go to directory : UI/AERS
6. Run following command:
	python manage.py runserver 
7. The Ui is hosted at http://127.0.0.1:8000
