import cv2
import csv
import os
import numpy as np
import pandas as pd
import numpy.linalg as npla
import pickle
from joblib import dump,load
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier        #Using multilayer perceptorn for classification task

def pca_labels():
    data = pd.read_csv("SML_Train.csv")
    train_img = data['id']
    train_img_vector = []
    print("Getting images vector!!")
    for img_name in train_img:
        img = cv2.imread("SML_Train/"+img_name,0).shape
        train_img_vector.append(img.flatten())
    
    print(len(train_img_vector))
    print("Doing PCA transformation!!")
    pca = PCA(n_components=0.95)
    pca_vectors = pca.fit_transform(train_img_vector)

    print("PCA done. Writing file!!")
    with open("pca_vectors.csv",'w') as file:
        writer = csv.writer(file)
        writer.writerows(pca_vectors)

def read_pca():
    print("Reading pca_vector file")
    pca_vectors = pd.read_csv("pca_vectors.csv",header=None)
    category = pd.read_csv("SML_Train.csv")["category"]
    
    
    for i in range(25):
        print("Category = ",i)
        occ = category[category == i]
        ind = []
        for j in occ.iteritems():
            ind.append(j[0])
        cat_pca_vect = pca_vectors.iloc[ind]
        mean = np.mean(cat_pca_vect,axis=0)
        std = npla.norm(np.std(cat_pca_vect,axis=0))
        
        k = 0
        for j in reversed(ind):
            vect = pca_vectors.iloc[j]
            dist = npla.norm(np.subtract(vect,mean))
            if (dist >= 1.25*std ):
                pca_vectors.drop(pca_vectors.index[j],inplace = True)
                category.drop(category.index[j],inplace = True)
                k+=1

        print("droped = ",k)
        print(pca_vectors.shape)
        print(category.shape)
        pca_vectors.reset_index(drop = True,inplace = True)
        category.reset_index(drop = True,inplace = True)
    
    print("Final length = ")
    print(pca_vectors.shape)
    print(category.shape)
    print("Writing final trainale data")
    pca_vectors.to_csv('train_vectors.csv', index = False)
    category.to_csv('train_cat.csv', index = False)

def train():
    print("Reading files!!!")

    train_vec = pd.read_csv("train_vectors.csv")
    train_cat = pd.read_csv("train_cat.csv").values.ravel()
    print(train_cat[1])
    print(len(train_cat))
    print(train_vec.iloc[0])
    print("Initilizing model")
    model = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100),verbose=True)
    print("Fitting the model!!!")
    model.fit(train_vec,train_cat)
    print("Saving Model!!")
    dump(model, 'model.joblib')

def test():
    img_name = os.listdir("SML_Test")
    img_name.sort(key=lambda x: int(x[5:len(x)-4]))
    imgs_vec = []
    print("Reading test images!!")
    for img in img_name:
        img_mat = cv2.imread("SML_Test/"+img,0)
        imgs_vec.append(img_mat.flatten())
    
    print("Dong PCA transformation!!!")
    pca = PCA(n_components=1371)
    test_pca = pca.fit_transform(imgs_vec)

    model = load("model.joblib")
    print("Predicting")
    y = model.predict(test_pca)
    print(len(y))
    print("Writing file")
    with open("2017233_Harkishan_submission.csv",'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id","category"])
        for i in range(len(img_name)):
            writer.writerow([img_name[i],y[i]])

    

                  
#pca_labels()
#read_pca()
#train()
test()


