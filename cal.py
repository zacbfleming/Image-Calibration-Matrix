import numpy as np
import cv2
import glob

def check(objpoints,imgpoints,rvecs,tvecs,mtx,dist):
    mean_error = 0 
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: ", mean_error/len(objpoints)) 

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img 


def chss():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) 
    objpoints = [ ]
    imgpoints = [ ]
    images = glob.glob('ch*.png')#/home/artichoke/Computer Vision/ps3/pic/chess*.png')
    for image in images:
        im = cv2.imread(image)
        chk, out = cv2.findChessboardCorners(im, (7,7),None)
        print(chk,out)
        if chk:
            objpoints.append(objp)
            #corners=cv2.cornerSubPix(im,out,(7,7),(-1,-1),criteria)
            imgpoints.append(out)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (541,512),None,None)
            rrvecs = np.asarray(rvecs[0])
            print(rrvecs)
            ttvecs = np.asarray(tvecs[0])
            print(ttvecs)
            #corners = cv2.drawChessboardCorners(im, (7,7), out, chk)
            imgpts, jac = cv2.projectPoints(axis, rrvecs, ttvecs, mtx, dist)
            draw(im,out,imgpts)
            cv2.imshow('corners',im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
    
    
        
    check(objpoints,imgpoints,rvecs,tvecs,mtx,dist)
    print(rvecs)
    return ret,mtx,dist,rvecs,tvecs

 
        
       
ret,mtx,dist,rvecs,tvecs = chss()


