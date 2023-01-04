import numpy as np
from scipy.signal import convolve2d



def main():
    print("main")

    array =  np.random.randint(16, size=(3, 3,2))
    print("INPUT ARRAY : \n", array)
    #print("\nIndices of min element : ", np.transpose(array,(1, 0,2)))
    #print("\nShape of min element : ", np.transpose(array,(1, 0,2)).shape)

    print(np.flip(array,(0,1,2)))
    print(np.flip(array,(1,0)))

    #diags = [array[::-1,:].diagonal(i,0,1).T for i in range(-array.shape[0]+1,array.shape[1] )]
#diags.extend(matrix.diagonal(i) for i in range(3,-4,-1))
    #print ([n.tolist() for n in diags])

if __name__=="__main__":
    main()