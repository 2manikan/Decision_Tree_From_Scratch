# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:58:29 2025

@author: manid
"""

import numpy as np

#define tree class
class Tree:
    def __init__(self):
        self.root_value = None #(index to check). index represents the feature
        self.possible_choices_of_features = set()
        self.subtrees = {}   #class_possibility:tree(root has the next index to check)
        self.height = 1

    def modify_root_value(self, value):  #modifies the index to check
        self.root_value = value

    def insert(self, class_possibility):
        self.subtrees[class_possibility] = Tree()


#finding best feature
def find_best_feature(dataset):
    
    max_entropy = -9999999
    respective_feature = None
    possibilities_of_max_entropy_feature = None
    
    #iterate through every feature (aka column)
    for feature in range(dataset.shape[1]):  #range(dataset.shape[1]) represents each column
        #obtain possibilities and raw frequencies
        unique_values, counts = np.unique(dataset[:,0], return_counts=True)
        
        #calculate entroy
        frequencies = (counts/dataset.shape[0])
        entropy = ( frequencies * np.log(frequencies)  ).sum()   # the '*' is elementwise
        
        #update maximum entropy
        if max_entropy < entropy:
            max_entropy = entropy
            respective_feature = feature
            possibilities_of_max_entropy_feature = unique_values
    
    return (feature, possibilities_of_max_entropy_feature)
    

#dataset prep
dataset = np.random.rand(4,5)

    

#initialization
decision_tree = Tree()
stack = [decision_tree]
max_height = 3

#conditions/base cases
while len(stack) > 0:
     if decision_tree.height >= max_height:
         break
    
     #pop from stack
     choice = stack.pop()
     
     #process node from stack
     best_feature, possibilities_of_feature = find_best_feature(dataset)
     choice.modify_root_value(best_feature)
     choice.possible_choices_of_features.add(best_feature)
     
     #append all the children to stack
     for possible in possibilities_of_feature:
         choice.subtrees[possible] = Tree() 
         choice.subtrees[possible].possible_choices_of_features.add(best_feature)  #this recursively updates for each node, no need for something like a .extend() function
         
     #can safely do this because it's guaranteed that we enter the above for loop
     choice.height+=1


