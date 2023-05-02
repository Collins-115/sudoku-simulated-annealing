import numpy as np
from random import randint,random
from math import log
from copy import deepcopy 
import time 
import os 
N_repetitions = 1
MAX_SWAPS = 3000000

'''
One of the most important thing to know about this solver is the way how a sudoku is represented. 
Sudoku will be represented as a 9x9 Matrix. The rows of the matrix are the rows of the grid, and the elements of each row is the corresponding column 
inside of it. 
In this solver we will refer to the subcell of the sudoku matrix. The subcell are the divisions that corresponds to the game.

      C1 C2 C3 C4 C5 C6 C7 C8 C9
R1            |        |     
R2    subcell1|subcell2|subcell3
R3            |        |     
      --------------------------
R4            |        |     
R5    subcell4|subcell5|subcell6
R6            |        |     
      --------------------------
R7            |        |           
R8    subcell7|subcell8|subcell9
R9            |        |     

Each subcell will be composed by two type of cells, that can be divided into numbers and free cells. The free cells are the ones that do not present
a number in the initial configuration of the game. Each cell has in ID inside its subcell

IDs in the subcell:
1 2 3
4 5 6
7 8 9
'''


#easy = 43 free cells
#medium = 51 free cells
#extreme = 58 free cells


#The coordinates inside the sudoku matrix are extracted accordingly to the IDs inside the subcell. An ID corresponds to the position of a cell inside the subcell, 
  #and each subcell acts inside a range of rows and columns. 
  #E.g. the free cell 7 of the subcell 6 will correspond to the cell in R2C1 inside the subcell that intersects rows 6-7-8 and columns 0-1-2. 
  #Then inside the sudoku matrix it will be in the position [8,1]. It is the second cell of the last row in the sudoku grid
def get_coordinates(sub,id):
  return (sub//3)*3 + id//3, (sub%3)*3 + id%3


#This function allows to calculate the score of the rows of a given matrix A. The score is determined by the number of values that occurs
#more than 1 time or 0 times in the row. The lower the score, the better the solution. 0 will correspond to the exact solution
def scores_per_rows(A,score_per_rows = np.zeros(shape=(1,9)),specific_rows=[]):
  if specific_rows:
    counter = [0] * 9
    for el in A[specific_rows[0]]:
      counter[int(el)-1] +=1
    score_per_rows[0,specific_rows[0]] = len(list(filter(lambda x: x!=1, counter)))
    counter = [0] * 9
    for el in A[specific_rows[1]]:
      counter[int(el)-1] +=1
    score_per_rows[0,specific_rows[1]] = len(list(filter(lambda x: x!=1, counter)))
    return score_per_rows
  
  for i,row in enumerate(A):
    counter = [0] * 9
    for el in row:
      counter[int(el)-1] +=1
    score_per_rows[0,i] = len(list(filter(lambda x: x!=1, counter)))
  return score_per_rows


#This function allows to take a sudoku grid A, the number of a subcell sub, and to swap the values of two free cells id1 and id2 inside of it
def swap_neighbours(A,sub,id1,id2):
  #The work is done in the matrix then it is necessary to get the coordinates of each free cells in the subcell
  x1, y1 = get_coordinates(sub,id1)
  x2, y2 = get_coordinates(sub,id2) 
  A[x1,y1], A[x2,y2] = A[x2,y2], A[x1,y1]
  return A

#This function allows to update the score of the solution given the free cells that are being swapped. 
def scoring_function_neighbours(A,sub,id1,id2,cur_score_rows,cur_score_columns):
  x1, y1 = get_coordinates(sub,id1)
  x2, y2 = get_coordinates(sub,id2)

  #The update of score in row occurs only if the free cells are from different rows
  if(x1!=x2):
    cur_score_rows = scores_per_rows(A,cur_score_rows,[x1,x2])
  
  #The update of score in columns occurs only if the free cells are from different columns
  if(y1!=y2):
    cur_score_columns = scores_per_rows(A.T,cur_score_columns,[y1,y2])

  return cur_score_rows,cur_score_columns

#The schedule returns a number that will decrease slowly over time. Also its deceleration decreases over time.
def schedule(t):
  return  8/log(t)

#This function allows to calculate the score for rows and columns.
def scoring_function(A):
  return scores_per_rows(A), scores_per_rows(A.T)


#This function takes the initial grid of the sudoku, and returns the position of the free cells, and the missing values inside a subcell
def create_free(celle):
  free_cells = [[] for _ in range(9)]
  free_values = [[1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9],
                [1,2,3,4,5,6,7,8,9]]
  for i in range(9):
    for j in range(3):
      for k in range(3):
        if(celle[(i//3)*3 + j,3*(i%3) + k] == 0):
            free_cells[i].append(j*3 + k)
        else:
            free_values[i].remove(celle[(i//3)*3 + j,3*(i%3) + k])

  return free_values, free_cells

#This function allows to fullfil a grid in which there are free cells. Each subcell will contain all of the values between 1-9. 
def populate_cells(celle,free_cells,free_values):
  for i,sub_cell in enumerate(free_cells):
    assignment = np.random.choice(free_values[i],size=len(free_values[i]),replace=False)
    for j,cell in enumerate(sub_cell):
      celle[(i//3)*3 + cell//3,(i%3)*3+cell%3] = assignment[j]
  return celle

#This function allows to implement a simulated annealling step. 
#celle is the grid representing the problem
#current conflict represents the current number of elements that appear more than once or 0 times in the rows and columns. It is also called score
#cur_score_rows is a list of scores for each row
#cur_score_columns is a list of scores for each column
#t is the time, or the current number of iteration
#free cells is a list containing list of free cells in each subcell
#The algorithm works by choosing a random subcell, swapping the values of free cells inside of it. If the score decreases (meaning there are less conflicts), 
#the swapping is performed and it is returned the current new grid and new score values. If it doesn't decreases, the new setting is accepted with a probability 
#inversionally proportional to the badness of the swap
def simulated_annealing(celle,current_conflict,cur_score_rows,cur_score_columns,t,free_cells):
    subcell = randint(0,8)
    neighbours = np.random.choice(free_cells[subcell],2,replace=False)
    celle_new = swap_neighbours(deepcopy(celle),subcell,neighbours[0],neighbours[1])
    new_score_rows,new_score_columns = scoring_function_neighbours(celle_new,subcell,neighbours[0],neighbours[1],cur_score_rows.copy(),cur_score_columns.copy())
    new_conflict = np.sum(new_score_rows) + np.sum(new_score_columns)
    if(new_conflict < current_conflict):
      return celle_new,new_conflict,new_score_rows,new_score_columns
    else:
      T = schedule(t)
      delta = current_conflict-new_conflict
      number = np.exp((delta/T))
      
      if number > random():
        return celle_new,new_conflict,new_score_rows,new_score_columns
      else:
        return celle, current_conflict, cur_score_rows, cur_score_columns

#This function allows to take a matrix representing the sudoku problem and creates the environment for solving it. They are first identified the missing values 
#from the grid and the cells that can be filled. These last ones are then filled randomly, with numbers that ensure that each subcell contains no conflicting numbers
#and then the score is calculated. The algorithm then will apply the simulated annealling over time.
def solve_with_simulated_annealing(sudoku):
  free_values,free_cells = create_free(sudoku)
  cells = populate_cells(sudoku,free_cells,free_values)
  cur_score_rows,cur_score_columns = scoring_function(cells)
  current_conflict = np.sum(cur_score_rows) + np.sum(cur_score_columns)
  t = 2
  while t < MAX_SWAPS:
    cells, current_conflict, cur_score_rows, cur_score_columns = simulated_annealing(cells,current_conflict, cur_score_rows, cur_score_columns,t,free_cells) 
    t+=1
    if current_conflict == 0:  
      break
  return current_conflict,cells,t

#This function allows to load from the files in the directory the sudoku grids
def load_levels(folders):
  levels = {}
  for folder in folders:
    for filename in os.listdir(f'./{folder}/'):
      sudoku = []
      with open(f'./{folder}/{filename}','r') as f:
        for line in f:
          sudoku.append([int(x) for x in line.split(',')])
        levels[filename.split('.')[0]] = deepcopy(sudoku)
  
  return levels 
if __name__ == '__main__':
  
  levels = load_levels(['easy','medium','extreme'])

  for sudoku in levels:
    execution_times = []
    iteration_list = []
    scores = []
    for i in range(N_repetitions):
      start_time = time.time()
      score, solution, iterations = solve_with_simulated_annealing(np.array(levels[sudoku]))
      end_time = time.time()
      print(score,solution,iterations)
      execution_times.append(end_time-start_time)
      iteration_list.append(iterations-1)
      scores.append(score)
      #print(score, end_time-start_time, iterations-1)
    
    #Save into file
    '''
    with open(f'./{sudoku}.txt','w') as f:
        print('ok')
        f.write(f'{execution_times}\n{iteration_list}\n{scores}')
    print(round(mean(execution_times),2),round(stdev(execution_times),2),round(mean(iteration_list),2),round(stdev(iteration_list),2),len(list(filter(lambda x: x>0,scores))))
    '''

