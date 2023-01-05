import numpy as np
from scipy.signal import convolve2d



def main():
    print("main")

    #array =  np.random.randint(16, size=(3, 3,2))
    #print("INPUT ARRAY : \n", array)
    #print("\nIndices of min element : ", np.transpose(array,(1, 0,2)))
    #print("\nShape of min element : ", np.transpose(array,(1, 0,2)).shape)

    #print(np.flip(array,(0,1,2)))
    #print(np.flip(array,(1,0)))

    #sliced = []
    #list = [1,2,4,5,6]
    #for index,row in enumerate (list):
    #    sliced = sliced +  [np.arange(row,index)]
    #print (len(sliced))

    shape = (3,4,2)

    z = np.random.randint(100,size = shape)
    r = np.zeros_like(z)
    rng = np.arange(0,shape[1])
    rng1 = np.arange(4)
    #z[rng1, rng1+1,:] = np.array([2**2,2+0.5])
    print("Matrix Z before Change: \n",z)

    def mainDiag():
            ans = []
            array = z
            diags = [array.diagonal(i,0,1).T for i in range(0,-array.shape[0],-1)] \
            + [array.diagonal(i,0,1).T for i in range(0,array.shape[1] )]

            return diags

    diagonals = mainDiag()

    if shape[1] > shape[0]:
        for i in range (shape[0]):
            #print("X: ",rng[i:min(shape[1],shape[0])])
            #print("Y: ",rng[0 : min(shape[0]-i,shape[1])])
            r[rng[i:shape[0]], rng[0 : shape[0]-i],:] = diagonals[i]
    else:
        rng = np.arange(0,shape[0])
        for i in range (shape[0]):
            #print("X,I: ",i," ",rng[i:min(shape[0]-i , shape[1]+i)])
            #print("Y: ",rng[0 : min(shape[0]-i,shape[1]+1)])
            r[rng[i:shape[1]+i], rng[0 : min(shape[0]-i,shape[1])],:] = diagonals[i]
            #print("matrix Z: \n",z)

    for i in range (shape[1]):
        r[rng[0:min(shape[0],shape[1]-i)], rng[0:min(shape[0],shape[1]-i)]+i,:] = diagonals[-shape[1]+i]

    
    
    print("R after all changes: \n",r)

    #diags = [array[::-1,:].diagonal(i,0,1).T for i in range(-array.shape[0]+1,array.shape[1] )]
#diags.extend(matrix.diagonal(i) for i in range(3,-4,-1))
    #print ([n.tolist() for n in diags])

if __name__=="__main__":
    main()