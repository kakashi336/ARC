#!/usr/bin/python
""" 
Group Submission
----------------

Arjun prakash 
21239525
MSc. AI
GitHub: https://github.com/kakashi336/ARC

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


#---------------------Start of functions-----------------------#
"""
c3f564a4.json
This function takes a grid (Which contains a pattern and blank bits) and returns the same sized grid after 
filling the blank bits with pattern. 

1. Iterate over the whole input grid and check for blanks (black value)
2. If black color is encountered, predict the pattern by checking the right diagonal color.
3. Assign the black pixel with the predicted value.
4. return the final grid.

Summary of features used
For all the tasks, I have used only Numpy, and no other libraries or any pip installs.
Completed the task using numpy array indexing. 

"""

def solve_c3f564a4(x):
    # Function to predict bit by checking its neighbour color bits
    def predict_bit(indexof_bit):
        #Check diagonal bits and return the same color code. 
        if x[indexof_bit[0]-1][indexof_bit[1]+1] >0:
            return x[indexof_bit[0]-1][indexof_bit[1]+1]
        
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Check for blanks and predict bit color.
            if x[i][j]==0:
                x[i][j]=predict_bit([i,j])
    #Return final grid
    return x

# Tested the function with test-data and passed all the testcases

"""
9f236235.json
This function takes in (m,m) sized grid and outputs (n,n) sized grid. The input grid consists of mini colored squared grids seperated
with partition lines in both axis. The task is to identify the color of this mini grids and output a grid of size(Number of partitions+1).
The output is to be inversed (Left mirrored image) and returned.

1. Find the number of partitions and the color of partition line.
2. Get the size of output martix (dimension= number of partition lines+1,number of partition lines+1) 
3. Get the size of mini grids size (number of color grids between each partitions)
4. Iterate over the grid and visit one pixel from every mini grid and assign the color to the output 
   after inversing (mirroring the image).
5. Return the output grid

Summary of features/library used:
For all the tasks, I have used only Numpy, and no other libraries or any pip installs.

"""
def solve_9f236235(x):
    data=x
    #Variable to store the color of partition
    partition_color=0
    # Count the number of partition lines
    partition_count=0
    
    #traverse through the matrix
    for i in data:
        #Partition will have same color accross the row and the length of set will be one. 
        if len(set(i))==1:
            
            # Get the color of partition.
            partition_color=set(i)
            # Count the number of partition lines.
            partition_count+=1    
            
    # Calculating the length of continous color grid by taking the length of first row, subtracting the count of partitions
    #and dividing it by the size of output matrix shape.
    color_length=len(data[0]-partition_count)//(partition_count+1)
    
    #jump variable will be used to shift the current index to the next grid color 
    jump=len(data[0]-partition_count)//color_length
    
    # Create output of respective size and zero the data
    output=np.zeros((partition_count+1,partition_count+1),dtype=int)

    # Assign the output grid color. Traverse through the grid with step=color_length+1 (+1 for partition)
    for i in range(0,data.shape[0],color_length+1):
        
        for j in range(0,data.shape[1],color_length+1):
            #Formula for applying color for grid: x= i/jump+1,y=(jump-1)-(j//jump+1). This automatically takes the color
            #from pattern and assignms to the output after mirroring.
            
            #take one pixel from each color grid and assign it to the output after inversing 0 axis 
            output[i//(jump+1)][(jump-1)-(j//(jump+1))]=data[i][j]         
    
    return output
# Tested the function with test-data and passed all the testcases


"""

6ecd11f4.json
The input grid (size: (m,m)) consists of a small color grid of size (n,n) and pattern of single color 'c'. The task is to output 
a grid of size (n,n) after merging the colors with respective pattern (replace color with 0 if the pattern consists of black (color code=0)
color pixel. 

1. Get the row size,col size, all color values, their counts and index from the input grid
2. get the color of the pattern and find the starting index of both the pattern and colored grid matrix.
3. Calculate the size of pattern grids (by going to the index of pattern and iterating till the black color
    is found. Store the finaliteration value.)
4. visit each pattern pixel one time and see if the color is black. If it is black, replace the respective output
    color to black.
5. return the final grid

Summary of features used
For all the tasks, I have used only Numpy and no other libraries or any pip installs.
Used numpy.unique() to get unique data,count and index from the input grid along with max() function. 

"""
def solve_6ecd11f4(x):
    data=x

    col_size,row_size=data.shape[0],data.shape[1]  #get Shape of whole input array
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
     
    
    # get The sliced color matrix from the input 
    out_matrix=data[output_index[0][0]:output_index[0][0]+len(output_index),
                    output_index[0][1]:output_index[0][1]+len(output_index)]
    
    # Store starting index of the pattern 
    x,y=indx[color_code]//row_size,indx[color_code]%row_size   
    length_of_boxes=0
    
    # Calcuate the grid dimension of pattern
    for i in range(data.shape[0]):
        #Traverse from the starting index of where pattern starts and get the shape
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

# Tested the function with test-data and passed all the testcases

"""
Commonalities:
1. Used np.shape function frequently to iterate through the array and get the shape for indexing 
    purpose
2. Used calculated indexing to properly iterate and index variables


#----------------------- End of Functions-------------------------------#


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

