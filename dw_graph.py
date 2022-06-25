from turtle import shape
import numpy as np
import pandas as pd
import random

class Graph:
    # Constructor to construct a graph
    def __init__(self, edges, n):
 
        # A list of lists to represent an adjacency list
        self.adjList = [None] * n
 
        # allocate memory for the adjacency list
        for i in range(n):
            self.adjList[i] = []
 
        # add edges to the directed graph
        for (src, dest, weight) in edges:
            # allocate node in adjacency list from src to dest
            self.adjList[src].append((dest, weight))
 
# Function to print adjacency list representation of a graph
def printGraph(graph):
    for src in range(len(graph.adjList)):
        # print current vertex and all its neighboring vertices
        for (dest, weight) in graph.adjList[src]:
            print(f'({src} —> {dest}, {weight}) ', end='')
        print()


# if __name__ == '__main__':
 
# Input: Edges in a weighted digraph (as per the above diagram)
# Edge (x, y, w) represents an edge from `x` to `y` having weight `w`
edges = [(1, 2, 5), (1, 3, 7), (1, 19, 4), (2, 9, 10), (2, 11 ,4), (3, 2, 2), (23, 1, 5), (22, 23, 4), (22, 3, 6), (20, 1 ,4), (19, 2, 7), (19, 20 , 9), (20, 21, 6), (22, 14, 5), (14, 13, 2), (13, 14, 1),(13, 6, 4), (6, 22, 5), (6, 3, 4), (4, 6, 2), (4, 3, 7), (4, 11, 5), (11, 3, 8), (9, 19, 5), (9, 12, 3), (21, 12, 4), (12, 10, 3), (10, 11, 7), (11, 25, 5), (4, 25, 6), (13, 24, 2), (9, 7, 1),(7, 8, 5), (8, 25, 9), (25, 7, 2), (7, 10, 4), (7, 17, 6), (17, 18, 9), (18, 21, 4), (18, 10, 8),(17, 5, 4),(5, 25, 2), (5, 15, 3), (15, 5, 9), (15, 16, 7), (15, 14, 3), (14, 24, 3), (24, 25, 5),(16, 24, 3), (24, 5, 9)]    
# No. of vertices 
n = 50
        
# construct a graph from a given list of edges
graph = Graph(edges, n)
        
# print adjacency list representation of the graph
printGraph(graph)

# q_table = np.matrix(np.random.uniform(low=-1,high=0,size=(26,26)))
q_table = np.matrix(np.zeros(shape=(26,26)))
q_table-=100 
#initialising our q-table with -100 in all places
reward=np.zeros(shape=(26,26))
#initialising all rewards to zero

for edge in edges:
    q_table[edge[0],edge[1]]=-edge[2]
    #to include the effect of the weight of the path in the graph

    if(edge[1]==25):
        reward[edge[0],edge[1]]=100-edge[2]
        # assigining a higher reward to the path that leads to the terminal path depending on their weight
     
#Function to move from one node to another 
def NextNode(node,epsilon):
    random_value=random.uniform(0,1)
    sample=[]
    if(random_value<epsilon):  
        #exploration continue if random value is less than epsilon by choosing a random node
        for edge in edges:
            if(edge[0]==node):
                sample.append(edge[1])            
        
    else:
        sample=np.where(q_table[node,]==np.max(q_table[node, ]))[1] 
        #choose from the nodes with maximum q_value to be the next node

    next_node=int(np.random.choice(sample,1))
    return next_node

#function to update q-value for the action that has been taken 
def Update_qvalue(node1,node2,alpha,discount):
    max_index=np.where(q_table[node2,]==np.max(q_table[node2,]))[1]

    if(max_index.shape[0]>1):
        max_index=int(np.random.choice(max_index,size=1))
    else:
        max_index=int(max_index)
    
    max_value=q_table[node2,max_index]
    #stores max future reward
    
    q_table[node1,node2]=((1-alpha)*q_table[node1,node2]+alpha*(reward[node1,node2]+discount*max_value))
    # print(pd.DataFrame(q_table).head())
    #updates the q-value for our action based on the q-learning algorithm

#Improving q values of our function at random nodes by repeatedly walking through them
def learn(epsilon,alpha,discount):
    for i in range(50000):
        start=np.random.randint(1,25)
        next_node=NextNode(start,epsilon)
        Update_qvalue(start,next_node,alpha,discount)
        
def shortest_path(initial_node,terminal_node):
    #function to find the shortest path using the updated q-values once the model has finished learning
    path=[initial_node]
    next_node=np.argmax(q_table[initial_node,])
    path.append(next_node)

    while next_node!=terminal_node:
        next_node=np.argmax(q_table[next_node,])
        path.append(next_node)
        
    return path
    
learn(0.5,0.8,0.8)

for i in range(1,25):
    print(shortest_path(i,25))














