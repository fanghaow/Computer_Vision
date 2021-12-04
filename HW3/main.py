import numpy as np
import cv2 as cv
from pca import PCA

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EigenFace():
    def __init__(self, isFaceDect=False):
        self.filepath = "input/"
        self.isFaceDect = isFaceDect
        self.loadImg()
        self.preprocess()
        self.train()

    def process(self): # Main
        correctNum = 0
        for id in range(1, self.pNum+1):
            for face in range(self.faceNum+1, 11):
                if face == 10:
                    file = self.filepath + str(id) + "/" + str(face) + ".png"
                else:
                    file = self.filepath + str(id) + "/0" + str(face) + ".png"

                predID = self.predict(file)
                if predID == id:
                    correctNum += 1
                    # print(predID)
        
        testNum = self.pNum * (10 - self.faceNum)
        print("[%d/%d], %f%% accurracy!" %(correctNum, testNum, 100 * correctNum / testNum))
    
    def loadImg(self):
        self.pNum = 40
        self.faceNum = 9
        self.N = self.pNum * self.faceNum
        self.scale = 4
        self.H = int(112 / self.scale)
        self.W = int(92 / self.scale)
        self.Data = np.zeros((self.N, self.W * self.H)) # (400,10304)
        for id in range(1, self.pNum+1):
            for faceID in range(1, self.faceNum+1):
                if faceID == 10:
                    file = self.filepath + str(id) + "/" + str(faceID) + ".png"
                else:
                    file = self.filepath + str(id) + "/0" + str(faceID) + ".png"
                img = cv.imread(file, 0) # Grayscale
                if img is None:
                    print("[ERROR]: Load training images failed! Please check your path.\n")
                    return
                
                if self.isFaceDect:
                    # face position detect
                    faces = face_cascade.detectMultiScale(img)
                    found = False
                    for face in faces:
                        (x, y, w, h) = faces[0]
                        area = w * h
                        if area > self.W * self.H * self.scale ** 2 / 2:
                            found = True
                            break
                    if not found:
                        (x, y, w, h) = (0, 0, self.W * self.scale, self.H * self.scale)

                    roiImg = img[y:y+h, x:x+w]
                else:
                    roiImg = img.copy()
                
                resizedImg = cv.resize(roiImg, (self.H,self.W))
                # cv.imshow("Resized", resizedImg)
                # cv.waitKey(100)
                imgVec = np.reshape(resizedImg, (resizedImg.size,)) # Vectorization
                self.Data[(id-1) * self.faceNum + (faceID-1), :] = imgVec

    def prepare(self, imgpath): # ROI face and Vectorization
        oriImg = cv.imread(imgpath, 0)
        if oriImg is None:
            print("[ERROR]: Load predicted image failed! Please check your path.\n")
            return
        
        if self.isFaceDect:
            faces = face_cascade.detectMultiScale(oriImg)
            if len(faces) == 0:
                (x, y, w, h) = (0, 0, self.W, self.H)
            else:
                (x, y, w, h) = faces[0]

            roiImg = oriImg[y:y+h, x:x+w]
        else:
            roiImg = oriImg.copy()

        resizedImg = cv.resize(roiImg, (self.H,self.W))
        imgVec = np.reshape(resizedImg, (resizedImg.size,))
        meanImg = np.mean(imgVec)
        stdImg = np.std(imgVec)
        preImg = (imgVec - meanImg) / stdImg
        return preImg

    def preprocess(self):
        self.preData = self.Data.copy() # (400,10304)
        self.rowNum, self.colNum = self.preData.shape
        # TODO
        minData = np.reshape(np.min(self.preData, axis=1), (self.rowNum,1)) # (400,1)
        maxData = np.reshape(np.max(self.preData, axis=1), (self.rowNum,1)) # (400,1)
        meanData = np.reshape(np.mean(self.preData, axis=1), (self.rowNum,1)) # (400,1)
        stdData = np.reshape(np.std(self.preData, axis=1), (self.rowNum,1)) # (400,1)
        ''' 1. Normalization '''
        # self.preData = (self.preData - minData) / (maxData - minData) # [0,1]
        # self.preData = (self.preData - meanData) / (maxData - minData) # [-1,1]
        ''' 2. Standardlization '''
        self.preData = (self.preData - meanData) / stdData # (-inf,inf)

    def train(self):
        myPCA = PCA(self.preData)
        self.projData, self.projVec = myPCA.analy_solution() # (N,dim)
        self.dims = self.projData.shape[1] # dimensions
        # TODO svm

    def predict(self, imgpath): 
        preImg = self.prepare(imgpath)

        projImg = np.reshape(np.dot(preImg, self.projVec.T).real, (self.dims,1)) # (dims,1)

        diff = np.sqrt((self.projData - projImg.T) ** 2) # (N,dims)
        similarity = np.sum(diff, axis=1)
        predID = np.argmin(similarity)
        ID = int(predID / self.faceNum) + 1
        face = predID - (ID - 1) * self.faceNum + 1
        similarImg = cv.imread(self.filepath + str(ID) + "/0" + str(face) + ".png")
        ''' Debugging INFO '''
        # cv.imshow("Similar", similarImg)
        # print("This person is No." + str(ID) + "and its similarity is " + str(similarity[predID]))
        # cv.imshow("Origin", oriImg)
        # cv.waitKey(0)
        return ID

def main():
    EF = EigenFace() # (isFaceDect=True)
    EF.process()
    # EF.predict("/home/fanghaow/Computer_Vision/HW3/input/41/1/1.jpg")
    # EF.predict("/home/fanghaow/Computer_Vision/HW3/input/42/1/1.jpg")

if __name__ == "__main__":
    main()