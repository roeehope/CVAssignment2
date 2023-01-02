import numpy as np
from scipy.signal import convolve2d



def main():
    print("main")

    array =  np.random.randint(16, size=(4, 6,5))
    print("INPUT ARRAY : \n", array)
    print("\nIndices of min element : ", np.argmin(array, axis = 2))
    print("\nShape of min element : ", np.argmin(array, axis = 2).shape)

if __name__=="__main__":
    main()