import multiprocessing
import numpy as np
from os import getpid
import time
import astropy.units as u
from functools import partial
from plasmapy.plasma.grids import CartesianGrid

nparticles = int(1e4)
x = np.random.random((nparticles,3))


batchsize = int(2)
nbatches = int(nparticles/batchsize)
batches = [ x[i*batchsize:(i+1)*batchsize, :] for i in range(nbatches)]


grid = CartesianGrid(-10*u.mm, 10*u.mm, num=200)
grid.add_quantities(B_x = np.random.random((200,200,200))*u.T)


def interp(x, grid=None):
    res = grid.nearest_neighbor_interpolator(x, 'B_x')
    return res, id(grid)




if __name__ == '__main__':
    
    #t0 = time.time()
    #output1, gid = interp(grid, x)
    #print(f"Serial: {time.time() - t0} s")
        
    
    
    t0 = time.time()
    output2 = np.zeros(nparticles)
    gids = []
    
    #x = multiprocessing.Array('d', x)
    with multiprocessing.Pool(processes=6) as pool:

        result = pool.map(partial(interp, grid=grid), batches)
        
        for i, res in enumerate(result):
            output2[i*batchsize:(i+1)*batchsize] = res[0]
            gids.append(res[1])
    
    
    gids = np.unique(np.array(gids))
    print(gids.size)
    print(f"Multipool: {time.time() - t0} s")
