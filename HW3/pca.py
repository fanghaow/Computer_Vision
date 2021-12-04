import numpy as np

class PCA():
    def __init__(self, data):
        self.data = data # (self.N, self.D)
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]

    def analy_solution(self):
        # convarience matrix S
        convMat = np.cov(self.data.T) # (self.D, self.D)
        # S * a1 = lambda * a1
        eigen_values, eigen_vectors = np.linalg.eig(convMat) # (1, self.D), (self.D, self.D[less])
        # print("Eigenvector: \n",eigen_vectors,"\n")
        # print("Eigenvalues: \n", eigen_values, "\n") 
        
        ''' Pick 95% explained components '''
        indices = np.argsort(eigen_values) # small -> big
        percent = 0
        self.pcVec = []
        for i in range(1, self.D + 1):
            index = indices[self.D - i]
            percent += eigen_values[index] / np.sum(eigen_values)
            self.pcVec.append(eigen_vectors[:,index])
            if percent > 0.95:
                break
        
        self.pcVec = np.array(self.pcVec)
        compNum = self.pcVec.shape[0]
        print(str(compNum) + " pricipal components\n")
        
        self.projData = np.dot(self.data, self.pcVec.T).real # Only acquire R
        convMat = np.cov(self.projData.T)
        # print("projected data: \n", self.projData)
        # print("convarience matrix: \n", convMat)

        return self.projData, self.pcVec

if __name__ == "__main__":
    pca = PCA(np.array([[1,2,3,4,5], [6.1,6.78,8.09,8.99,10.1]]))
    pca.analy_solution()