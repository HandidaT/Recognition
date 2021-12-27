import gzip,_pickle,random
import numpy as np
import pandas as pd
import time
from pprint import pprint

# Data loading and preprocessing

def load_data():
    with gzip.open('mnist.pkl.gz','rb') as file:
        training_data,validation_data,test_data=_pickle.load(file,encoding='latin1')
    
    print(len(training_data[0][49999]),len(training_data[1]),len(validation_data[1]),len(test_data[1]),type(training_data[1]))
    df0 = pd.DataFrame(training_data[0])
    df0[784]=training_data[1]
    #print(df0)
    df1 = pd.DataFrame(validation_data[0])
    df1[784]=validation_data[1]
    #print(df1)
    return df0.loc[:300,:], df1.loc[300:410,:],df0.loc[:410,:]

training_data,testing_data,dataset=load_data()
print(testing_data[784])

###defining entropy
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    #print("Entropy ",elements,counts,entropy,len(elements))
    return entropy

##Info Gain

def InfoGain(data,split_attribute_name,target_name=784):
    total_entropy = entropy(data[target_name])
    vals,counts = np.unique(data[split_attribute_name],return_counts=True)
    #cal the weighted entropy
    #print("IIIIIIIIIIIIIIIIIIIIIInfoGain len(vals)=",len(vals))
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).
                                dropna()[target_name])for i in range(len(vals))])
    
    #formula for information gain
    Information_Gain = total_entropy-Weighted_Entropy
    print("InfoGain after", total_entropy, Weighted_Entropy, Information_Gain)
    return Information_Gain

glob=0
def ID3(data,originaldata,features,target_attribute_name=784,
        parent_node_class=None):
    global glob
    glob+=1
    print("glob ",glob)
    #If all target_values have the same value,return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        glob-=1
        #print("--1 glob",glob,np.unique(data[target_attribute_name]))
        return np.unique(data[target_attribute_name])[0]
    
    #if the dataset is empty
    elif len(data) == 0:
        glob-=1
        #print("--2 glob",glob)
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],
                                                                           return_counts=True)[1])]
    
    #If the feature space is empty
    elif len(features) == 0:
        glob-=1
        #print("--3 glob",glob)
        return parent_node_class 

    #If none of the above condition holds true grow the tree

    else:
        #print("--4-----------infogain will be called",len(features),"times(no of features)")
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],
                                                                           return_counts=True)[1])]

    #Select the feature which best splits the dataset
    item_values = [InfoGain(data,feature,target_attribute_name)for feature in features] #Return the infgain values
    best_feature_index = np.argmax(item_values)
    #print("item_values ",item_values,best_feature_index,"how can it be argmax")
    best_feature = features[best_feature_index]

    #Create the tree structure
    tree = {best_feature:{}}

    #Remve the feature with the best info gain
    features = [i for i in features if i!= best_feature]

    #Grow the tree branch under the root node
    #print(best_feature,np.unique(data[best_feature]),"parent_node_class",[(np.unique(data[target_attribute_name],return_counts=True)[1])])
    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature]==value).dropna()
        #print("---------------------",value,target_attribute_name,parent_node_class,"\n", sub_data)
        #call the ID3 algotirthm
        subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
        #Add the subtree
        #print("----------------------------------------------subtree",subtree)
        #print("for value ",value)
        tree[best_feature][value] = subtree
        #print(subtree)
        pprint(tree)
    glob-=1
    #print("--5 glob", glob)
    return(tree)

#Predict
def predict(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
               result = tree[key][query[key]]
            except:
               return default

            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
##check the accuracy

def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data
#training_data = train_test_split(dataset)[0]
#testing_data = train_test_split(dataset)[1]

def test(data,tree):
   queries = data.iloc[:,:-1].to_dict(orient="records")
   predicted = pd.DataFrame(columns=["predicted"])

   #calculation of accuracy

   for i in range(len(data)):
       predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
   print(predicted)
   print(data[784].reset_index(drop=True))
   print("The Prediction accuracy is:",(np.sum(predicted["predicted"]==data[784].reset_index(drop=True))/len(data))*100,'%')
  

#Train the tree,print the tree abnd predict the accuracy
tree = ID3(training_data,training_data,training_data.columns[:-1])
pprint(tree)
test(testing_data,tree)

