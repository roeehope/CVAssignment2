"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        # initializing the SSDD tensor with zeros
        #num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        #ssdd_tensor = np.zeros((num_of_rows,
        #                        num_of_cols,
        #                        len(disparity_values)))
        """INSERT YOUR CODE HERE"""

        
        def shifted(mat,shift):
            """
            given a 2d array (mat) and a shift amount,
            returns the matrix, shifted by the given
            number of columns(new columns initialized with zeros)
            """
            rolled = np.roll(mat,shift,axis=1)
            if shift > 0:
                rolled[:,0:shift] = 0
            elif shift < 0:
                rolled[:,shift:] = 0
            return rolled

        def ssd_calc(disparity,left,right):
            rightS = shifted(right,-disparity)
            sd = np.sum(np.square(left - rightS),axis=2)
            kernel = np.ones((win_size,win_size))
            convolution = convolve2d(sd,kernel,mode='same',boundary='fill', fillvalue=0)
            return convolution
            
        layers = []
        for d in disparity_values:
            res = ssd_calc(d,left_image,right_image)
            layers.append(res)

        ssdd_tensor = np.stack(layers,axis = 2 )


        # normalizing the SSDD
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """

        #print (ssdd_tensor.shape)
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""

        label_no_smooth = np.argmin(ssdd_tensor, axis = 2) 

        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        #print(num_labels)
        #print(num_of_cols)
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""

        l_slice[:,0] = c_slice[:,0]
        
        mMatrix = np.zeros_like(l_slice)


        for i in range(1,num_of_cols):
            for d in range(num_labels):
                a = l_slice[d,i-1]

                if d - 1 >= 0 and d+1 < num_labels:
                    b = min(l_slice[d+1,i-1],l_slice[d-1,i-1])+ p1
                elif d - 1 >= 0 :
                    b = l_slice[d-1,i-1]+ p1
                else:
                    b = l_slice[d+1,i-1]+ p1

                c = np.min(l_slice[:,i-1])+ p2

                mMatrix[d,i] = min(a,b,c)

            l_slice[:,i] = c_slice[:,i] +mMatrix[:,i]- min(l_slice[:,i-1])


        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        

        for i in range(ssdd_tensor.shape[0]):
            l[i,:,:] = np.transpose(self.dp_grade_slice(np.transpose(ssdd_tensor[i,:,:]),p1,p2))
    
        return self.naive_labeling(l)

    def extractSlices(self,ssdd_tensor: np.ndarray):

        def mainDiag():
            ans = []
            array = ssdd_tensor
            diags = [array.diagonal(i,0,1).T for i in range(0,-array.shape[0],-1)] \
            + [array.diagonal(i,0,1).T for i in range(0,array.shape[1] )]

            return diags

        def secondaryDiag():
            ans = []
            array = ssdd_tensor[::-1,:]
            diags = [array.diagonal(i,0,1).T for i in range(-array.shape[0]+1,array.shape[1] )]
            return diags

        def makeSliceTensor(arrToSlice):
            return [arrToSlice[i,:,:] for i in range(arrToSlice.shape[0])]

        dictSlices = {}

        # we will go with the direction of the task
        dictSlices[1] = makeSliceTensor(ssdd_tensor)
        dictSlices[5] = makeSliceTensor(np.fliplr(ssdd_tensor))

        dictSlices[3] = makeSliceTensor(np.transpose(ssdd_tensor,(1, 0,2)))
        dictSlices[7] = makeSliceTensor(np.flipud( np.transpose(ssdd_tensor,(1, 0,2))))
        #dictSlices[7] = np.flip( np.transpose(ssdd_tensor,(1, 0,2)) ,(1,0))



        dictSlices[2] = mainDiag()
        dictSlices[6] = [np.flip(i,1) for i in dictSlices[2]]

        dictSlices[4] = secondaryDiag()
        dictSlices[8] = [np.flip(i,1) for i in dictSlices[3]]

        return dictSlices
        


    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE
        for i in range(ssdd_tensor.shape[0]):
            l[i,:,:] = np.transpose(self.dp_grade_slice(np.transpose(ssdd_tensor[i,:,:]),p1,p2))
    
        return self.naive_labeling(l)

        dictSlices[1] = ssdd_tensor
        dictSlices[5] = np.fliplr(ssdd_tensor)

        dictSlices[3] = np.transpose(ssdd_tensor,(1, 0,2))
        dictSlices[7] = np.flip( np.transpose(ssdd_tensor,(1, 0,2)) ,0)
        """
        def picture_straight(n,shape = ssdd_tensor.shape):
            return self.naive_labeling(self.createCostMapStraight(directions[n],p1,p2,shape))


    

        directions = self.extractSlices(ssdd_tensor)

        horizontal_shape = np.transpose(ssdd_tensor,(1,0,2)).shape

        direction_to_slice[2] = self.naive_labeling(self.createCostMapDiagonal(directions[2],p1,p2,ssdd_tensor.shape))


        #direction_to_slice[1] = picture_straight(1)
        direction_to_slice[1] = direction_to_slice[2]


        #direction_to_slice[2] = self.naive_labeling(self.createCostMapDiagonal(direction_to_slice[2],p1,p2,ssdd_tensor.shape))

        direction_to_slice[3] = direction_to_slice[1] #self.dp_labeling(directions[3],p1=p1,p2=p2) #self.dp_labeling(directions[3],p1=p1,p2=p2)
        #direction_to_slice[3] = np.transpose(picture_straight(3,horizontal_shape),(1,0))

        direction_to_slice[4] = direction_to_slice[1]

        direction_to_slice[5] = direction_to_slice[1] #np.fliplr(self.dp_labeling(directions[5],p1=p1,p2=p2)) #direction_to_slice[1]
        #direction_to_slice[5] = np.fliplr(picture_straight(5))

        direction_to_slice[6] = direction_to_slice[1]
        direction_to_slice[7] = direction_to_slice[1]

        #direction_to_slice[7] = np.transpose(np.flipud(picture_straight(7,horizontal_shape)),(1,0)) #direction_to_slice[1]

        direction_to_slice[8] = direction_to_slice[1]

        

        return direction_to_slice

    def diagonalsToMat(self,diagonals,shape):
        print("diagonals",len(diagonals))
        print("shape TO Mat", shape)
        #print("DUMMY: ", dummy )
        '''
        ans = []
            array = ssdd_tensor
            diags = [array.diagonal(i,0,1).T for i in range(-array.shape[0]+1,array.shape[1] )]
            return diags
        '''
        r = np.zeros(shape)

        rng = np.arange(0,shape[1])

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

        return r

        

    def createCostMapDiagonal(self,list,p1,p2,shape):
        l = np.zeros(shape)
        sliced = []
        for index,row in enumerate (list):
            sliced.append(np.transpose(self.dp_grade_slice(np.transpose(row),p1,p2)))

        print("shape of map diagonal",shape)
        print("diagnoals of map diagonal",len(sliced))
        return self.diagonalsToMat(sliced,shape)


    def createCostMapStraight(self,list,p1,p2,shape):

        print("shape: ",shape)
        #print("list shape: ",(len(list),list[0].shape[0],list[0].shape[1]))

        l = np.zeros(shape)
        for index,row in enumerate (list):
            l[index,:,:] = np.transpose(self.dp_grade_slice(np.transpose(row),p1,p2))

        return l



    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

