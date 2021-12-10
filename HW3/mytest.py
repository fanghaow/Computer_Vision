from eigenface import EigenFace
import numpy as np

EF = EigenFace()
projVec = np.loadtxt("model.txt", dtype=np.float)
EF.process(projVec)