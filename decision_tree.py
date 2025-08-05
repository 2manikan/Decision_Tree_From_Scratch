# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:58:29 2025

@author: manid
"""

import numpy as np
import pandas as pd

#define tree class
class Tree:
    def __init__(self):
        self.root_value = None #(index to check). index represents the feature
        self.possible_choices_of_features = set()
        self.subtrees = {}   #class_possibility:tree(root has the next index to check)
        self.level = 1
        self.respective_dataset = None

    def modify_root_value(self, value):  #modifies the index to check
        self.root_value = value

    def insert(self, class_possibility):
        self.subtrees[class_possibility] = Tree()


#finding best feature
def find_best_feature(dataset, not_allowed):
    
    min_entropy = 9999999
    respective_feature = None
    possibilities_of_max_entropy_feature = None
    
    #iterate through every feature (aka column)
    for feature in range(dataset.shape[1]):  #range(dataset.shape[1]) represents each column
        
        if not(feature in not_allowed):
            
            entropy = 0    
        
            #CALCULATE ENTROPY FOR THAT PARTICULAR FEATURE
            #iterate over sub-datasets split by possibilities of the feature in question
            possibilities_of_current_feature = set(dataset[:,feature].flatten())
            
            for pos in possibilities_of_current_feature:
                sub_dataset = dataset[dataset[:,feature] == pos]
                
                #obtain raw frequencies of each prediction possibility.
                unique_values, counts = np.unique(sub_dataset[:,-1], return_counts=True)
                
                #calculate frequency
                frequencies = (counts/sub_dataset.shape[0])
                
                #we aim for the least entropy
                #the '*' is elementwise. we do * -1 because entropy is always negative bc we use probability --> can tell from graphing xlogx
                entropy += (( frequencies * np.log(frequencies)  ).sum()) * -1
                
            
            #update minimum entropy
            if min_entropy > entropy:
                min_entropy = entropy
                respective_feature = feature
                possibilities_of_max_entropy_feature = possibilities_of_current_feature
    
    return (respective_feature, possibilities_of_max_entropy_feature)
    

#----------------------------------------------------------------------------------------

#DATASET PREP
#reading in the csv file using pandas (better for reading in csv files)
df=pd.read_csv("C://Users//manid//Downloads//bas.txt", delimiter=',')
df = df.iloc[:, 1:]  #removing the "Animals" column (first column)

#converting categories to integers so that comparisons in decision tree are O(1) time and not O(n)
unique_values = []       #for storing unique values of each feature
for i in range(df.shape[-1]):   #traversing each column
    df.iloc[:,i], unique = pd.factorize(df.iloc[:,i])  #modifying each column in place
    unique_values.append(unique)

#convert data frame to numpy (each feature is now denoted as different indices)
dataset = df.to_numpy()
 


#initialization
decision_tree = Tree()
decision_tree.respective_dataset = dataset
decision_tree.possible_choices_of_features.add(dataset.shape[-1]-1) #we cannot use last column (prediction) as a feature!
stack = [decision_tree]
max_level = 4
sub_dataset = None


#conditions/base cases
while len(stack) > 0:
    
     #pop from stack
     choice = stack.pop()
     
     
     
     #process node from stack
     best_feature, possibilities_of_feature = find_best_feature(choice.respective_dataset, choice.possible_choices_of_features)   #WE NEED TO FILTER DATASET BASED ON PREVIOUS CONDITIONS!!!!!
     choice.modify_root_value(best_feature)
     choice.possible_choices_of_features.add(best_feature)
     
     # print([i.level  for i in stack])
     # print(choice.level)
     # print(best_feature)
     # print("--->",possibilities_of_feature)
     # print(choice.respective_dataset)
     
     #append all the children to stack
     for possible in possibilities_of_feature:
         if choice.level < max_level:   #do not add ANY more children if we already reached the height
             choice.subtrees[possible] = Tree()
             #update possible choices of features
             choice.subtrees[possible].possible_choices_of_features.update(choice.possible_choices_of_features)
             #increase the level of the child nodes
             choice.subtrees[possible].level += choice.level
             #prepare respective dataset
             choice.subtrees[possible].respective_dataset = choice.respective_dataset[choice.respective_dataset[:,best_feature] == possible]
             #append to stack
             stack.append(choice.subtrees[possible])
    
     
     
     print(choice.respective_dataset)
     print(f"splitting on {best_feature}: ")   
     print(choice.subtrees)
     print(choice.level)
     print("-----------------")
     
     


