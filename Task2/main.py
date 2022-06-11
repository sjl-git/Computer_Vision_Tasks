import cv2
import numpy as np
import sys
import os

if not os.path.exists("2017147543"):
    os.mkdir("2017147543")
output = open(os.path.join("2017147543", "output.txt"), "w")

faces = np.zeros((1,192*168), dtype=np.int64)
for i in range(1, 40):
    fileName = "./faces_training/face"
    if i < 10:
        fileName += "0" + str(i) + ".pgm"
    else:
        fileName += str(i) + ".pgm"
    face = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE).flatten()
    faces = np.append(faces, [face], axis=0)

faces = faces[1:]
meanFace = faces.mean(axis=0)
faces = faces - meanFace
u, s, _ = np.linalg.svd(faces.T, full_matrices=False)
s = np.square(s)
s = s/s.sum()
targetCoverage = float(sys.argv[1])
coverage = 0
cnt = 0

while coverage < targetCoverage:
    coverage += s[cnt]
    cnt += 1


output.write("#"*10 + " STEP 1 " + "#"*10 + "\n")
output.write("Input Percentage: %.2f\n" % targetCoverage)
output.write("Selected Dimension: %02d\n\n" % cnt)

if cnt == 1:
    u = u[:, 0]
if cnt > 1:
    u = u[:, 0:cnt]

uT = u.T

errors = np.array([])
for i in range(39):
    coefficient = np.dot(uT, faces[i].T)
    newFace = np.dot(u, coefficient) + meanFace
    cv2.imwrite("./2017147543/face%02d.pgm" % (i+1), newFace.reshape(192, 168))
    targetFace = faces[i] + meanFace
    reconstructionErr = (np.square(newFace - targetFace)).mean(axis=0)
    errors = np.append(errors, reconstructionErr)
    faces[i] = newFace

output.write("#"*10 + " STEP 2 " + "#"*10 + "\n")
output.write("Reconstruction error\n")
output.write("Average: %.4f\n" % errors.mean())
for i in range(39):
    output.write("%02d: %.4f\n" %(i+1, errors[i]))
output.write("\n")

output.write("#"*10 + " STEP 3 " + "#"*10 + "\n")
for i in range(1, 6):
    testName = "./faces_test/test0" + str(i) + ".pgm"
    testFace = cv2.imread(testName, cv2.IMREAD_GRAYSCALE).flatten()
    coefficient = np.dot(uT, testFace.T)
    newTestFace = np.dot(u, coefficient)
    min = -1
    minIdx = -1
    for j in range(39):
        dist = np.linalg.norm(newTestFace-faces[j])
        if min < 0:
            min = dist
            minIdx = j
        else:
            if min > dist:
                min = dist
                minIdx = j
    output.write("test%02d.pgm ==> face%02d.pgm\n" %(i, minIdx+1))