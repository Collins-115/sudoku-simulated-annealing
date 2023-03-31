import numpy as np
from random import sample,randint,random
from math import log
from copy import deepcopy 
import time 
from matplotlib import pyplot as plt
from statistics import mean, stdev
import os 

#easy = 43 free cells
#medium = 51 free cells
#extreme = 58 free cells

    
def swap_neighbours(A,sub,id1,id2):
  x1 = (sub//3)*3 + id1//3
  x2 = (sub//3)*3 + id2//3
  y1 = (sub%3)*3+id1%3
  y2 = (sub%3)*3+id2%3
  temp = A[x1][y1]
  A[x1][y1] = A[x2][y2]
  A[x2][y2] = temp 
  return A 

def scoring_function_neighbours(A,sub,id1,id2,cur_score_rows,cur_score_columns):
  x1 = (sub//3)*3 + id1//3
  x2 = (sub//3)*3 + id2//3
  y1 = (sub%3)*3+id1%3
  y2 = (sub%3)*3+id2%3  
  if(x1!=x2):
    counter = [0] * 9
    for el in A[x1]:
      counter[int(el)-1] +=1
    cur_score_rows[0,x1] = len(list(filter(lambda x: x!=1, counter)))
    counter = [0] * 9
    for el in A[x2]:
      counter[int(el)-1] +=1
    cur_score_rows[0,x2] = len(list(filter(lambda x: x!=1, counter)))
  
  if(y1!=y2):
    counter = [0] * 9
    for el in A.T[y1]:
      counter[int(el)-1] +=1
    cur_score_columns[0,y1] = len(list(filter(lambda x: x!=1, counter)))
    counter = [0] * 9
    for el in A.T[y2]:
      counter[int(el)-1] +=1
    cur_score_columns[0,y2] = len(list(filter(lambda x: x!=1, counter)))

  return cur_score_rows,cur_score_columns

def schedule(t):
  return  8/log(t)

def scoring_function(A):
  
  score_per_rows = np.zeros(shape=(1,9))
  score_per_columns = np.zeros(shape=(1,9))
  for i,row in enumerate(A):
    counter = [0] * 9
    for element in row:
      counter[int(element)-1] +=1
    
    score_per_rows[0,i] = len(list(filter(lambda x: x!=1, counter)))
  for i,row in enumerate(np.transpose(A)):
    counter = [0] * 9
    for element in row:
      counter[int(element)-1] +=1
    
    score_per_columns[0,i] = len(list(filter(lambda x: x!=1, counter)))
  
  return score_per_rows, score_per_columns

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
def populate_cells(celle,free_cells,free_values):
  for i,sub_cell in enumerate(free_cells):
    assignment = np.random.choice(free_values[i],size=len(free_values[i]),replace=False)
    for j,cell in enumerate(sub_cell):
      celle[(i//3)*3 + cell//3,(i%3)*3+cell%3] = assignment[j]
  return celle

def simulated_annealling(celle,current_conflict,cur_score_rows,cur_score_columns):
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
  N_repetitions = 100
  levels = load_levels(['easy','medium','extreme'])

  for sudoku in levels:
    execution_times = []
    iterations = []
    scores = []
    for i in range(N_repetitions):
      celle = np.array(levels[sudoku])
      free_cells = [[] for _ in range(9)]
      free_values,free_cells = create_free(celle)
      celle = populate_cells(celle,free_cells,free_values)
      
          
      start_time = time.time()
      cur_score_rows,cur_score_columns = scoring_function(celle)
      current_conflict = np.sum(cur_score_rows) + np.sum(cur_score_columns)
      t = 2
      
      while t<3000000:
        celle, current_conflict, cur_score_rows, cur_score_columns = simulated_annealling(celle,current_conflict, cur_score_rows, cur_score_columns) 
        t+=1
        
        # if (t%10000 == 0):
        #   print(t,current_conflict)
        

        if current_conflict == 0:  
          break
      # for row in celle:
      #   print(row)
      end_time = time.time()
      

      execution_times.append(end_time-start_time)
      iterations.append(t-1)
      scores.append(current_conflict)
      #print(current_conflict, end_time-start_time, t-1)
    with open(f'./{sudoku}.txt','w') as f:
        print('ok')
        f.write(f'{execution_times}\n{iterations}\n{scores}')
    #print(round(mean(execution_times),2),round(stdev(execution_times),2),round(mean(iterations),2),round(stdev(iterations),2),len(list(filter(lambda x: x>0,scores))))
    

