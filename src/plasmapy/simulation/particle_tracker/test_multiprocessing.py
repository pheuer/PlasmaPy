import multiprocessing
import numpy as np
from os import getpid
import time

size = 1000
x = np.random.random(size)


batchsize = 100
nbatches = int(size/batchsize)
batches = [ x[i*batchsize:(i+1)*batchsize] for i in range(nbatches)]





class Grid:
    def __init__(self):
        self.arr = np.random.random(batchsize)

    def increment(self, x):
        time.sleep(0.01)
        return self.arr + x
    
    
def push_fcn(grid, x):
    print(grid)
    #print("I'm process", getpid())
    return grid.increment(x)
    
    
# Single grid object, will be copied for each
# process automatically by python
grid = Grid()
    


if __name__ == '__main__':
    
    output = np.zeros(size)
    
    with multiprocessing.Pool(6) as pool:
        
        #grid = multiprocessing.sharedctypes.copy(arr)
        TASKS = [(grid,batch ) for batch in batches]
        

        
        result = pool.starmap(push_fcn, TASKS)
        
        for i, res in enumerate(result):
            output[i*batchsize:(i+1)*batchsize] = res
        
    print(output.shape)
        