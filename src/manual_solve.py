#!/usr/bin/python
""" 
Arjun prakash 
21239525
MSc. AI

AND

Mayank Dwivedi
21230080
MSc. AI

GitHub: https://github.com/kakashi336/ARC
"""

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_c3f564a4(x):
    data=x
    def predict_bit(indexof_bit):
        if data[indexof_bit[0]-1][indexof_bit[1]+1] >0:
            return data[indexof_bit[0]-1][indexof_bit[1]+1]
        
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j]==0:
                data[i][j]=predict_bit([i,j])
    return x

def solve_9f236235(x):
    data=x
    #Variable to store the color of partition
    partition_color=0
    partition_count=0
    
    #traverse through the matrix
    for i in data:
        #Partition will have same color accross the row and the length of set will be one. 
        if len(set(i))==1:
            #set partition Color.
            partition_color=set(i)
            # Count the number of partition lines.
            partition_count+=1    
            
    # Calculating the length of continous color grid by taking the length of first row, subtracting the count of partitions
    #and dividing it by the size of output matrix shape.
    color_length=len(data[0]-partition_count)//(partition_count+1)
    
    #jump variable to shift into the next color grid
    jump=len(data[0]-partition_count)//color_length
    
    # Create output of respective size
    output=np.zeros((partition_count+1,partition_count+1),dtype=int)

    # Assign the output grid color 
    for i in range(0,data.shape[0],color_length+1):
        for j in range(0,data.shape[1],color_length+1):
            
            #take one pixel from each color grid and assign it to the output after inversing 0 axis 
            output[i//(jump+1)][(jump-1)-(j//(jump+1))]=data[i][j]         
    
    return output

def solve_6ecd11f4(x):
    data=x
    
    col_size,row_size=data.shape[0],data.shape[1]  #Shape of whole array
    # Get the colors,index and counts of unique colors
    colors,indx,counts=np.unique(data,return_counts=True,return_index=True)
    
    # Ignore the color with 0 index (Black color) and get the max count of color. This will give color of the pattern 
    a=np.where(counts[1:]==counts[1:].max())  
    color_code=a[0][0]+1  # color code of pattern pixel 
    
    
    output_index=[]  #Index of matrix (output) first column 
    
    # Traverse through rows
    for i in range(data.shape[0]):
        #Traverse through columns
        for j in range(data.shape[1]):
            
            # Find the output color matrix location
            if data[i][j]!=color_code and data[i][j]!=0:
                # Storing the index of the first column
                output_index.append([i,j])
                break     
     
    
    # get The sliced output color matrix from the input 
    out_matrix=data[output_index[0][0]:output_index[0][0]+len(output_index),
                    output_index[0][1]:output_index[0][1]+len(output_index)]
    
    # starting indexes of the pattern 
    x,y=indx[color_code]//row_size,indx[color_code]%row_size   
    length_of_boxes=0
    
    # Calcuate the grid dimension of pattern
    for i in range(data.shape[0]):
        if data[x][y+i]==color_code:
            length_of_boxes+=1
        else:
            break
    
    # Match the pattern and change the color of the output matrix to 0 depending on 
    # the pattern
    for i in range(out_matrix.shape[0]):
        for j in range(out_matrix.shape[1]):
            if data[x+i*length_of_boxes][y+j*length_of_boxes]==0:
                out_matrix[i][j]=0
    return out_matrix


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

