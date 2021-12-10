import numpy as np

class PCA():
    def __init__(self, data, percent_th=0.95):
        self.data = data # (self.N, self.D)
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.percent_th = percent_th

    def analy_solution(self):
        convMat = np.cov(self.data.T) # (self.D, self.D)
        eigen_values, eigen_vectors = np.linalg.eig(convMat) # (1, self.D), (self.D, self.D[less])
        ''' Pick 95% explained components '''
        indices = np.argsort(eigen_values) # small -> big
        percent = 0
        self.pcaVec = []
        for i in range(1, self.D + 1):
            index = indices[self.D - i]
            percent += eigen_values[index] / np.sum(eigen_values)
            self.pcaVec.append(eigen_vectors[:,index])
            if percent > self.percent_th:
                break
        
        self.pcaVec = np.array(self.pcaVec)
        compNum = self.pcaVec.shape[0] # Number of principal components
        print(str(compNum) + " principal components\n")
        return self.pcaVec

if __name__ == "__main__":
    pca = PCA(np.array([[1,2,3,4,5], [6.1,6.78,8.09,8.99,10.1]]))
    pca.analy_solution()