import multiprocessing
import numpy as np
from os import getpid

x = np.random.random(10000)
arr = np.random.random(10)
output = np.zeros(10000)
batchsize = 10
nbatches = 1000
batches = [ x[i*batchsize:(i+1)*batchsize] for i in range(nbatches)]

def fcn(args):
    print("I'm process", getpid())
    x,y = args
    return x + y


if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        
        #grid = multiprocessing.sharedctypes.copy(arr)
        
        grid = arr
        TASKS = [(batch, grid) for batch in batches]
        

        
        result = pool.map(fcn, TASKS)
        
        print(result)
        
        #print(result)
        
        """
        tasks = [ (np.random.random(10), x) for x in batches]
        
        results  = [ pool.apply_async(fcn,t) for t in tasks]
        for r in results:
            print(r.get()) 
        """