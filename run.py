import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

# load data
data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items
print("Data Loaded!")

# train model
print("Training model...")
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)
print("Model Trained!")

while True:
    message = input('\nInsert a message or "Q" to quit... \n >>> ')
    if message == 'Q': break
    Classifier
    Vectorizer
    try:
        if len(message) > 0:
          vectorize_message = Vectorizer.transform([message])
          predict = Classifier.predict(vectorize_message)[0]
          predict_proba = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
        print(error)
    
    predict = "Not Spam" if (predict == "ham") else ("Spam")
    
    print('Message: {}'.format(message))
    print('Predict: {}'.format(predict))
    print('Predict Probability: \n - Spam: {} \n - Not Spam: {}'.format(predict_proba[0][1], predict_proba[0][0]))
    