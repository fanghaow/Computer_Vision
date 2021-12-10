import numpy as np
import cv2 as cv
from pca import PCA

class EigenFace():
    def __init__(self, isFaceDect=False):
        self.filepath = "input/" # TODO
        self.isFaceDect = isFaceDect
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def reBuild(self):
        newImgs = np.dot(self.projData, self.projVec) # (N,D)
        newImgs = np.array(newImgs * self.Std + self.Mean, dtype=np.uint8)
        for i in range(newImgs.shape[0]):
            currImg = np.reshape(newImgs[i,:], (self.W,self.H))
            currImg = cv.resize(currImg, (self.W * self.scale * 8, self.H * self.scale * 8))
            cv.imwrite("output/eigenface/" + str(int(i / self.faceNum + 1)) + "-" + str(int(i % self.faceNum + 1)) + ".jpg", currImg)

    def process(self, projVec): # Test
        self.loadImg()
        self.preprocess()
        self.projVec = projVec # (dims,D)
        self.projData = np.dot(self.preData, self.projVec.T).real # (N,dims)
        self.dims = self.projData.shape[1] # dimensions
        # Define the codec and create VideoWriter object
        ''' Visualizing '''
        self.reBuild()
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.out = cv.VideoWriter('/Users/fanghao_w/Desktop/Computer_Vision/HW3/output/output.mp4', self.fourcc, 10, (1200, 800))
        correctNum = 0
        for id in range(1, self.pNum+1):
            for face in range(self.faceNum+1, 11):
                if face == 10:
                    file = self.filepath + str(id) + "/" + str(face) + ".png"
                else:
                    file = self.filepath + str(id) + "/0" + str(face) + ".png"

                predID = self.predict(file)
                font = cv.FONT_HERSHEY_TRIPLEX
                delay = 100
                pos = (400, 400)
                if id == 41:
                    if predID == id:
                        correctNum += 1
                        color = (0,255,0)
                        text = "BingGo"
                        pos = (340, 400)
                    else:
                        color = (0,0,255)
                        text = "Uhoh"
                    delay = 200
                elif predID == id:
                    correctNum += 1
                    text = "True"
                    color = (0,255,0) # BGR
                else:
                    text = "False"
                    color = (0,0,255)
                
                cv.putText(self.finalImg, text, pos, font, 5, color, 3)
                cv.putText(self.finalImg, "Source", (100, 150), font, 3, (255,255,0), 3)
                cv.putText(self.finalImg, "Similar", (700, 150), font, 3, (255,0,255), 3)
                cv.imshow("Identification", self.finalImg)
                self.out.write(self.finalImg)
                cv.waitKey(delay)
        
        testNum = self.pNum * (10 - self.faceNum)
        print("[%d/%d], %f%% accuracy!" %(correctNum, testNum, 100 * correctNum / testNum))
        text = "[" + str(correctNum) + "/" + str(testNum) + "], " \
            + str(int(100 * correctNum / testNum)) + "% accuracy!"
        cv.putText(self.finalImg, text, (50, 700), font, 2.5, (0,255,0), 3)
        cv.imshow("Identification", self.finalImg)
        cv.waitKey(1000)
        self.out.write(self.finalImg)
        self.out.release()
    
    def loadImg(self):
        self.pNum = 41
        self.faceNum = 7
        self.N = self.pNum * self.faceNum
        self.scale = 4
        self.H = int(112 / self.scale)
        self.W = int(92 / self.scale)
        self.Data = np.zeros((self.N, self.W * self.H)) # (N,D)
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
                
                if self.isFaceDect: # face position detect
                    roiImg = self.faceDetect(img)
                else:
                    roiImg = img.copy()
                
                resizedImg = cv.resize(roiImg, (self.H,self.W))
                imgVec = np.reshape(resizedImg, (resizedImg.size,)) # Vectorization
                self.Data[(id-1) * self.faceNum + (faceID-1), :] = imgVec

    def faceDetect(self, img):
        if len(img.shape) > 2:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        faces = self.face_cascade.detectMultiScale(gray)
        found = False
        for _ in faces:
            (x, y, w, h) = faces[0]
            area = w * h
            if area > self.W * self.H * self.scale ** 2 / 5:
                found = True
                break
        if not found:
            (x, y, w, h) = (0, 0, self.W * self.scale, self.H * self.scale)
        roiImg = img[y:y+h, x:x+w]
        self.face = (x, y, w, h)
        return roiImg

    def preparePred(self, imgpath): # ROI face and Vectorization
        oriImg = cv.imread(imgpath, 0)
        if oriImg is None:
            print("[ERROR]: Load predicted image failed! Please check your path.\n")
            return
        
        if self.isFaceDect:
            roiImg = self.faceDetect(oriImg)
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
        minData = np.reshape(np.min(self.preData, axis=1), (self.rowNum,1)) # (N,1)
        maxData = np.reshape(np.max(self.preData, axis=1), (self.rowNum,1)) # (N,1)
        meanData = np.reshape(np.mean(self.preData, axis=1), (self.rowNum,1)) # (N,1)
        stdData = np.reshape(np.std(self.preData, axis=1), (self.rowNum,1)) # (N,1)
        self.Mean = meanData # (N,1)
        self.Std = stdData # (N,1)
        ''' 1. Normalization '''
        # self.preData = (self.preData - minData) / (maxData - minData) # [0,1]
        # self.preData = (self.preData - meanData) / (maxData - minData) # [-1,1]
        ''' 2. Standardlization '''
        self.preData = (self.preData - meanData) / stdData # (-inf,inf)

    def train(self):
        self.loadImg()
        self.preprocess()
        myPCA = PCA(self.preData)
        self.projVec = myPCA.analy_solution().real # (dims,D)
        np.savetxt("model.txt", self.projVec) # Save model!

    def predict(self, imgpath): 
        predImg = self.preparePred(imgpath)
        projImg = np.reshape(np.dot(predImg, self.projVec.T).real, (self.dims,1)) # (D,1)
        diff = np.sqrt((self.projData - projImg.T) ** 2) # (N,D)
        similarity = np.sum(diff, axis=1)
        predID = np.argmin(similarity)
        ID = int(predID / self.faceNum) + 1
        face = predID - (ID - 1) * self.faceNum + 1
        similarImg = cv.imread(self.filepath + str(ID) + "/0" + str(face) + ".png")
        oriImg = cv.imread(imgpath)
        ''' My presentation '''
        oriImg = cv.resize(oriImg, (self.H * self.scale, self.W * self.scale))
        self.faceDetect(oriImg)
        (x, y, w, h) = self.face
        cv.rectangle(oriImg, (x, y), (x+w, y+h), (255,255,255), 2)

        similarImg = cv.resize(similarImg, (self.H * self.scale, self.W* self.scale))
        finalImg = np.hstack((oriImg, similarImg))
        self.finalImg = cv.resize(finalImg, (1200,800))
        return ID

def main():
    EF = EigenFace()
    # EF = EigenFace(isFaceDect=True)
    projVec = np.loadtxt("model.txt")
    EF.process(projVec)

if __name__ == "__main__":
    main()