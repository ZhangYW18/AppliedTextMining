# Analyse the Sentiment of Sentences
## 1. Preprocess Dataset

Put your training CSV data called "training.csv" under folder "data". It is required that the data must contain columns named "text", storing the texts, and "object", a number valued 0(negative)/1(positive) representing the sentiment of the corresponding texts.

Modify "DATASET_COLUMNS" in line 11 of project/clean.py according to your dataset.

```python
DATASET_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
```

Run project/clean.py and you will get "tweets_processed.csv" under folder "data".
```shell script
cd project
python clean.py
```

## 2. Train Models

Create a folder called "models" under "project" folder. Run project/model.py. You will get 4 models under folder "project/models".


```shell script
mkdir models
python model.py
```

## 3. Predict the Testcases
Modify the testcases in file project/predict.py as you like. Run project/predict.py. You will see the predicted sentiment of your texts.
```shell script
python predict.py
```

## Local Python and Packages Version
```shell script
victor@ColedeMBP:~/PycharmProjects/AppliedTextMining(master⚡) » python --version
Python 3.6.0
victor@ColedeMBP:~/PycharmProjects/AppliedTextMining(master⚡) » pip list
Package                  Version     
------------------------ ------------
absl-py                  0.9.0       
asgiref                  3.2.7       
astor                    0.8.1       
boto3                    1.12.32     
botocore                 1.15.32     
cachetools               4.0.0       
certifi                  2019.11.28  
chardet                  3.0.4       
Django                   3.0.5       
docutils                 0.15.2      
gast                     0.3.3       
gensim                   3.8.1       
google-api-core          1.16.0      
google-auth              1.12.0      
google-cloud-core        1.3.0       
google-cloud-storage     1.26.0      
google-resumable-media   0.5.0       
googleapis-common-protos 1.51.0      
grpcio                   1.27.2      
h5py                     2.10.0      
idna                     2.9         
jmespath                 0.9.5       
joblib                   0.14.1      
Keras                    2.2.4       
Keras-Applications       1.0.8       
Keras-Preprocessing      1.1.0       
Markdown                 3.2.1       
nltk                     3.4.5       
numpy                    1.18.2      
pandas                   0.25.3      
pip                      20.0.2      
protobuf                 3.11.3      
pyasn1                   0.4.8       
pyasn1-modules           0.2.8       
python-dateutil          2.8.1       
pytz                     2019.3      
PyYAML                   5.3.1       
requests                 2.23.0      
rsa                      4.0         
s3transfer               0.3.3       
scikit-learn             0.22.2.post1
scipy                    1.4.1       
setuptools               46.1.3      
six                      1.14.0      
sklearn                  0.0         
smart-open               1.10.0      
sqlparse                 0.3.1       
tensorboard              1.12.2      
tensorflow               1.12.0      
termcolor                1.1.0       
urllib3                  1.25.8      
Werkzeug                 1.0.0       
wheel                    0.34.2
```
