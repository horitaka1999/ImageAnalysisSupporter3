from sklearn.decomposition import PCA
from itertools import combinations
import numpy as np 
class pcaVector:
    def __init__(self,contorData,parameter = 4):
        self.frontVector = []
        self.backVector = []
        self.datasize = len(contorData)
        self.contorData = contorData
        self.argData = []
        print(self.datasize * 2 /100)
        for i in range(len(contorData)):
            frontdata = []
            backdata = []
            for k in range(i,i+parameter):
                    frontdata.append(contorData[k%self.datasize])
            for k in range(i-parameter,i):
                    backdata.append(contorData[k%self.datasize])
            pca = PCA(n_components = 1)
            pca.fit(frontdata)
            pca2 = PCA(n_components = 1)
            pca2.fit(backdata)
            self.frontVector.append(pca.components_[0])
            self.backVector.append(pca2.components_[0])
        self.frontVector = np.array(self.frontVector)  
        self.backVector = np.array(self.backVector)

    def check(self,index):
        if 0 <= index < len(self.contorData):
            return True
        
       
    def calcMaxArg(self,index,kparameter):
        tmp = [i for i in range(index-kparameter//2,index+kparameter//2)]
        store = []
        for i,k in combinations(tmp,2):
            vec1 = self.frontVector[i%self.datasize]
            vec2 = self.backVector[k%self.datasize]
            arg = np.inner(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            arg = 1-abs(arg)
            store.append(arg)
        return max(store)
    
    def analysis(self,kparameter):
        self.argData = []
        for index in range(self.datasize):
            self.argData.append((self.calcMaxArg(index,kparameter),index))
        
    def reOverIndex(self,thresh):
        rev = []
        for index in range(self.datasize):
            if self.argData[index][0] > thresh:
                rev.append(self.argData[index][1])
        return rev

    



            
            
            
        