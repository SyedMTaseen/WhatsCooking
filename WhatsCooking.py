# -*- coding: utf-8 -*-
"""


@author: Taseen Syed
"""


from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
import json
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import operator

app = QtWidgets.QApplication([])

dlg = uic.loadUi("WhatsCookingUI.ui")

X1 = []
Y1 = []
id_test = []
X2 = []
choice = 1
cuisines = []
occurances_dict = {}
probabilities_dict = {}
vocab = []
prediction = {}
cuisine_probabilities = {}

def main():
    print("hello");

   
    
def NBTrain():
    global classifier
    classifier = GaussianNB().fit(tf1, Y1)
    
def KNNTrain():   
    global classifier
    classifier = DecisionTreeClassifier(max_depth = 40).fit(tf1, Y1) 
    
def FTTrain():
    global classifier
    classifier=RandomForestClassifier(max_depth=40, n_estimators=20).fit(tf1,Y1)
     
def OurNBTrain():
    global occurances_dict
    global vocab
    global probabilities_dict
    print("Main")
    for i in range(len(cuisines)):
        for entry in train_json:
            if(entry['cuisine']==cuisines[i]):
                for ingredient in entry['ingredients']:
                    if ingredient not in vocab:
                        vocab.append(ingredient)
                    if ingredient not in occurances_dict[cuisines[i]]:
                        occurances_dict[cuisines[i]][ingredient]=1;
                    if ingredient in occurances_dict[cuisines[i]]:
                        occurances_dict[cuisines[i]][ingredient]=occurances_dict[cuisines[i]][ingredient]+1
                        
        for ingredient in occurances_dict[cuisines[i]]:
            probability=occurances_dict[cuisines[i]][ingredient]+1
            probability=probability/(sum(occurances_dict[cuisines[i]].values())+len(vocab))
            probabilities_dict[cuisines[i]][ingredient]=probability
        
    
        
    
def PredictList():
   print("edit")
   ingredientString=dlg.IngredientTxtBox.toPlainText()
   print(ingredientString)
   ingredientString=ingredientString.split(',')
   print(ingredientString)
   #Prediction.
   to_predict = ingredientString
   for i in range(len(cuisines)):
       class_probability=1
       for ingredient in to_predict:
           if ingredient in probabilities_dict[cuisines[i]]:
               class_probability=class_probability*probabilities_dict[cuisines[i]][ingredient]
           if ingredient not in probabilities_dict[cuisines[i]]:
               temp_probability=1/(sum(occurances_dict[cuisines[i]].values())+len(vocab))
               class_probability=class_probability*temp_probability
       prediction[cuisines[i]]=class_probability
        
   print(prediction)
   print(max(prediction.items(), key=operator.itemgetter(1))[0])
   dlg.IngredientTxtBox.setText("The ingredients make a "+max(prediction.items(), key=operator.itemgetter(1))[0]+" Cuisine")
    
    
    
def PredictTxtData():   
    print("Main")
    prediction=classifier.predict(tf2)
    out = io.open('predictions.csv','w')
    for i in range(len(X2)):
        out.write('%s,%s\n'%(id_test[i],prediction[i]))
    dlg.IngredientTxtBox.setText("Predictions saved in predictions.csv, inside the Project folder.")
    
    
if __name__=="__main__":
    main()
dlg.NBTrainButton.clicked.connect(NBTrain)
dlg.KNNTrainButton.clicked.connect(KNNTrain)
dlg.FTTrainButton.clicked.connect(FTTrain)
dlg.OurNBTrainButton.clicked.connect(OurNBTrain)
dlg.PListButton.clicked.connect(PredictList)
dlg.PTDButton.clicked.connect(PredictTxtData)
train_file = io.open('train.json','r')
train_json = json.loads(train_file.read())
test_file = io.open('test.json','r')
test_json = json.loads(test_file.read())
for entry in train_json:
    X1.append(" ".join(entry['ingredients']))
    Y1.append(entry['cuisine'])
    if entry['cuisine'] not in cuisines:
        cuisines.append(entry['cuisine'])
        occurances_dict[entry['cuisine']] = {}
        probabilities_dict[entry['cuisine']] = {}
        prediction[entry['cuisine']] = {}
for o in test_json:
    id_test.append(o['id'])
    X2.append(" ".join(o['ingredients']))
    
vect = TfidfVectorizer().fit(X1)
tf1=vect.transform(X1).todense()
tf2=vect.transform(X2).todense()
#classifier=RandomForestClassifier(max_depth=40, n_estimators=20).fit(tf1,Y1)






dlg.show()
app.exec()

