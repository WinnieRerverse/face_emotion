import os
import dlib
import numpy as np
import cv2
import time
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage import io
from sklearn.externals import joblib
from face_align import face


MOUTH_POINTS = list(range(48, 61))  # 嘴巴
RIGHT_BROW_POINTS = list(range(17, 22))  # 右眉毛
LEFT_BROW_POINTS = list(range(22, 27))  # 左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))  # 右眼睛
LEFT_EYE_POINTS = list(range(42, 48))  # 左眼睛
NOSE_POINTS = list(range(27, 35))  # 鼻子
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS+LEFT_EYE_POINTS+RIGHT_EYE_POINTS+NOSE_POINTS+MOUTH_POINTS)
  



label_ditc={"anger":0,"disgust":1,"fear":2,"happy":3,"neutral":4,"sadness":5,"surprise":6}
reverse_dict={v: k for k, v in label_ditc.items()}

predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
classifier=joblib.load("face_classifier.pkl")

data_file = open('data_48.pkl', 'rb')
data=pickle.load(data_file)
label_file = open('label_48.pkl', 'rb')
label=pickle.load(label_file)
data_array=np.array(data).reshape(len(data),data[0].shape[1])
label_array=np.array(label)



def read_data():
    data_list=[]
    label_list=[]
    start=time.time()
    for root, dirs, files in os.walk("C:\\train"):
        root=root+"\\"
        for i in files:
            if os.path.isfile(os.path.join(root,i)):
                label=root.split("\\")[-2]
                print(label)
                if ".png" in i:
                    img = io.imread(root+i)
#                    img=face(Base_path,cover_path)#人脸对齐
                    detector = dlib.get_frontal_face_detector()
                    dets = detector(img, 1)                    
                    landmarks=np.array([[p.x, p.y] for p in predictor(img, dets[0]).parts()])
#                    landmarks=np.vstack((landmarks[:,0]/img.shape[0],landmarks[:,1]/img.shape[1])).T#标准化
                    ALIGN_POINTS_makrs=landmarks[ALIGN_POINTS]
                    ALIGN_POINTS_makrs=ALIGN_POINTS_makrs.reshape(1,ALIGN_POINTS_makrs.shape[0]*2)

                    '''
                    采用距离作为特征点
                    LEFT_BROW=np.mean(landmarks[LEFT_BROW_POINTS],0)
                    LEFT_EYE=np.mean(landmarks[LEFT_EYE_POINTS],0)
                    RIGHT_BROW=np.mean(landmarks[LEFT_BROW_POINTS],0)
                    RIGHT_EYE=np.mean(landmarks[RIGHT_EYE_POINTS],0)
                    UP_MOUTH=np.mean(landmarks[list(range(49,54))],0)
                    DOWN_MOUTH=np.mean(landmarks[list(range(55,60))],0)
                    LEFT_MOUTH=np.mean(landmarks[[48,60]],0)
                    RIGHT_MOUTH=np.mean(landmarks[[54,64]],0)
                    distance_marks=np.array([distance(LEFT_BROW,LEFT_EYE),distance(RIGHT_BROW,RIGHT_EYE),
                                            distance(UP_MOUTH,DOWN_MOUTH),distance(LEFT_MOUTH,RIGHT_MOUTH)])
                    distance_marks=distance_marks.reshape(1,4)
                    data_list.append(distance_marks)
                    '''
                    data_list.append(ALIGN_POINTS_makrs)
                    label_list.append(label_ditc[label])
                    output_data= open('data.pkl','wb')
                    output_label=open('label.pkl','wb')
                    pickle.dump(data_list,output_data)
                    pickle.dump(label_list,output_label)
    end=time.time()
    print(end-start)
    
def out_put(rect,img,PIXEL):
    [x1,x2,y1,y2]=[rect.left(),rect.right(),rect.top(),rect.bottom()]
    if x1>=0 and x2>=0 and y1>=0 and y2>=0:
        new_img=img[y1:y2,x1:x2]
        new_img=cv2.resize(new_img,(PIXEL,PIXEL))
        return new_img
    else:
        return img


def svm_classfier():
#    x_train, x_test, y_train, y_test = train_test_split(data_array, label_array)
#    classifier=SVC(decision_function_shape='ovo',probability=True)
    classifier=RandomForestClassifier(n_estimators=100)
    classifier.fit(data_array,label_array)
    
#    classifier.fit(x_train, y_train)
#    y_pred=classifier.predict(x_test)
#    print(classification_report(y_test,y_pred))   
    joblib.dump(classifier,"face_classifier.pkl")




def predictor():

    Base_path=cv2.imread("standard.jpg",cv2.IMREAD_COLOR)
    pre = cv2.imread("test4.jpg", cv2.IMREAD_COLOR)
    dets = detector(pre,1)
    result_list=[]
    st_out_list=[]
    for i in range(len(dets)):
        out=out_put(dets[i],pre,48)
        if out.shape==pre.shape:
            continue
        warped_im2=Image.fromarray(face(Base_path,out)).convert('L')
        st_out=np.array(warped_im2)
        st_out_list.append(st_out)
        dets_out = detector(st_out,1)
        if len(dets_out)==0:
            result_list.append(7)
        else:
            landmarks=np.array([[p.x, p.y] for p in predictor(st_out, dets_out[0]).parts()])
            ALIGN_POINTS_makrs=landmarks[ALIGN_POINTS]
            ALIGN_POINTS_makrs=ALIGN_POINTS_makrs.reshape(1,ALIGN_POINTS_makrs.shape[0]*2)
            result=classifier.predict_proba(ALIGN_POINTS_makrs)
            result_list.append(result)
        return result_list
                

def plot_landmarks(img,landmarkss):
    for idx, point in enumerate(landmarkss):
        pos = (point[0], point[1])
        cv2.circle(img, pos,1, color=(255,0,0))
#    cv2.imshow("img", img) #27
#    cv2.waitKey(0)  
    return img
    
def plot_sample(x, y, axis):
   im = x.reshape(96, 96)
   axis.imshow(im, cmap='gray')
   axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)   

def distance(vector1,vector2):
    return np.sqrt(np.dot(vector1-vector2,(vector1-vector2).T))
 


    
    




