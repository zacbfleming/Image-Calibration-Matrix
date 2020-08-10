import numpy as np
import cv2 as cv
import glob

### CalMat produces the 
def CalMat():
    
    c = 0
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1, 3)
    print(axis)
    objpoints = [ ]
    imgpoints = [ ]
    images = glob.glob('/home/artichoke/Computer Vision/ps3/pic/chess*.png')
    for fname in images:
        print('stare')
        img = cv.imread(fname)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)
        print(ret, c)
        c+=1
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            imgpoints.append(corners)
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
        cv.destroyAllWindows()
        img2 = cv.imread('/home/artichoke/Computer Vision/ps3/input/left12.jpg')
        h, w = img2.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        mapx , mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(img2, mapx, mapy, cv.INTER_LINEAR)
        #ifsave = input('s to save calibration matrix, and coefficients')
        #if ifsave == 's':
         #   np.savez('B', ret, mtx, dist, rvecs, tvecs)
        dst= cv.undistort(img2, mtx,dist,None,newcameramtx)
        x, y, w, h = roi
        dst = dst[0:y+h+400, 0:x+w+400]
        cv.imwrite('calibresult.png', dst)
        mean_error = 0
        c = len(objpoints)
        print(c)
        for i in range(c):
            imgpoints2, _= cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print('total error: {}'.format(mean_error/len(objpoints)) )
        #img2 = cv.imread('left12.jpg.png')
        cv.imshow('left12.jpg', img2)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow('calibresult.png', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
        XYZAxis(mtx, dist, rvecs, tvecs)
        print(mtx,dist,rvecs,tvecs)
        return mtx, dist, rvecs, tvecs

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img


def XYZAxis(mtx, dist, rvecs, tvecs):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1, 3)
    for fname in glob.glob('/home/artichoke/Computer Vision/ps3/pic/chess*.png'):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img, corners2, imgpts)
            cv.imshow('XYZ', img)
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                cv.imwrite(fname[:6]+'.png', img)
            cv.destroyAllWindows()



CalMat()
#with np.load('/home/artichoke/Computer Vision/ps3/input/B.npz') as X:
#        mtx, dist, rvecs, tvecs = [X[i] for i in ('arr_0.npy', 'arr_1.npy', 'arr_2.npy', 'arr_3.npy')]
