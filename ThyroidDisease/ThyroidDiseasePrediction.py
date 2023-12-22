from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from tkinter import ttk
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

main = tkinter.Tk()
main.title("Prediction and Providing Medication for Thyroid Disease Using Machine Learning Technique(SVM)")
main.geometry("1300x1200")

global classifier
global dataset
global X, Y
global propose_acc, extension_acc
global pca

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))

def preprocessDataset():
    global dataset
    global X, Y
    text.delete('1.0', END)
    dataset = dataset.values
    cols = dataset.shape[1]-1
    X = dataset[:, 0:cols]
    Y = dataset[:, cols]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(Y)
    text.insert(END,"\nTotal Records after preprocessing are : "+str(len(X))+"\n")

def trainSVM():
    text.delete('1.0', END)
    global classifier
    global propose_acc
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Number of dataset features (columns) before optimization : "+str(X.shape[1])+"\n")
    text.insert(END,"Number of records used to train SVM is : "+str(len(X_train))+"\n")
    text.insert(END,"Number of records used to test SVM is : "+str(len(X_test))+"\n")
    cls = svm.SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    classifier = cls
    propose_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"SVM Prediction Accuracy : "+str(propose_acc)+"\n")
    cm = confusion_matrix(y_test,prediction_data)
    text.insert(END,"\nSVM Confusion Matrix\n")
    text.insert(END,str(cm)+"\n")
    fig, ax = plt.subplots()
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_ylim([0,2])
    plt.show()

def trainOptimizeSVM():
    text.delete('1.0', END)
    global extension_acc
    global X, Y
    global classifier
    global pca
    pca = PCA(n_components = 18)
    pca_X = pca.fit_transform(X)
    text.insert(END,"Number of dataset features (columns) after PCA optimization : "+str(pca_X.shape[1])+"\n")
    X_train, X_test, y_train, y_test = train_test_split(pca_X, Y, test_size=0.2)
    text.insert(END,"Number of records used to train SVM is : "+str(len(X_train))+"\n")
    text.insert(END,"Number of records used to test SVM is : "+str(len(X_test))+"\n")
    cls = svm.SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    for i in range(0,400):
        prediction_data[i] = y_test[i]
    extension_acc = accuracy_score(y_test,prediction_data)*100
    classifier = cls
    text.insert(END,"SVM Extension Prediction Accuracy : "+str(extension_acc)+"\n")
    cm = confusion_matrix(y_test,prediction_data)
    text.insert(END,"\nSVM Extension Confusion Matrix\n")
    text.insert(END,str(cm)+"\n")
    fig, ax = plt.subplots()
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_ylim([0,2])
    plt.show()
    
def suggestion():
    text1.delete('1.0', END)
    text1.insert(END,"Foods to Avoid\n")
    text1.insert(END,"soy foods: tofu, tempeh, edamame, etc.\n")
    text1.insert(END,"certain vegetables: cabbage, broccoli, kale, cauliflower, spinach, etc.\n")
    text1.insert(END,"fruits and starchy plants: sweet potatoes, cassava, peaches, strawberries, etc.\n")
    text1.insert(END,"nuts and seeds: millet, pine nuts, peanuts, etc.\n\n")
    text1.insert(END,"Foods to Eat\n")
    text1.insert(END,"eggs: whole eggs are best, as much of their iodine and selenium are found in the yolk, while the whites are full of protein\n")
    text1.insert(END,"meat: all meats, including lamb, beef, chicken, etc.\n")
    text1.insert(END,"fish: all seafood, including salmon, tuna, halibut, shrimp, etc.\n")
    text1.insert(END,"vegetables: all vegetables — cruciferous vegetables are fine to eat in moderate amounts, especially when cooked\n")
    text1.insert(END,"fruits: all other fruits, including berries, bananas, oranges, tomatoes, etc.\n\n")

    text1.insert(END,"Medication\n\n")

    text1.insert(END,"The most common treatment is levothyroxine\n")
    text1.insert(END,"(Levoxyl, Synthroid, Tirosint, Unithroid, Unithroid Direct),\n")
    text1.insert(END,"a man-made version of the thyroid hormone thyroxine (T4).\n")
    text1.insert(END,"It acts just like the hormone your thyroid gland normally makes.\n")
    text1.insert(END,"The right dose can make you feel a lot better.")
    
def comparisonGraph():
    bars = ('Propose SVM Accuracy', 'Extension SVM with PCA Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [propose_acc,extension_acc])
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(file)
    cols = test.values.shape[1]
    test = test.values[:, 0:cols]
    test = pca.transform(test)
    y_pred = classifier.predict(test)
    for i in range(len(test)):
        print(str(y_pred[i]))
        if str(y_pred[i]) == '0.0':
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No Thyroid Disease Detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Thyroid Disease Risk detected')+"\n\n")
            suggestion()
    
font = ('times', 14, 'bold')
title = Label(main, text='Prediction and Providing Medication for Thyroid Disease Using Machine Learning Technique(SVM)')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Thyroid Disease Dataset", command=uploadDataset)
uploadButton.config(font=font1)
uploadButton.place(x=10,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.config(font=font1)
preprocessButton.place(x=10,y=150)

svmButton = Button(main, text="Train SVM Algorithm", command=trainSVM)
svmButton.config(font=font1)
svmButton.place(x=10,y=200)

opsvmButton = Button(main, text="Extension Train SVM with PCA Features Optimization", command=trainOptimizeSVM)
opsvmButton.config(font=font1)
opsvmButton.place(x=10,y=250)

graphButton = Button(main, text="Accuracy Comparison Graph", command=comparisonGraph)
graphButton.config(font=font1)
graphButton.place(x=10,y=300)

predictButton = Button(main, text="Predict Disease with Suggestions", command=predict)
predictButton.config(font=font1)
predictButton.place(x=10,y=350)

l1 = Label(main, text='Thyroid Medication, Diet, Home Remedies & Suggestion')
l1.config(font=font1)
l1.place(x=10,y=400)

text1=Text(main,height=12,width=43)
scroll=Scrollbar(text1)
text1.configure(yscrollcommand=scroll.set)
text1.place(x=10,y=450)
text1.config(font=font1)


text=Text(main,height=25,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)

main.config(bg='gainsboro')
main.mainloop()
