#coding:utf-8
from django.shortcuts import render
from facereco.models import IMG
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from sklearn.externals import joblib
import os
import sys
import json
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import numpy as np
import dlib
import cv2
from skimage import io
sys.path.append("/home/ai/AIApp/facereco/")
from face_align import face


def out_put(rect,img,PIXEL):
    [x1,x2,y1,y2]=[rect.left(),rect.right(),rect.top(),rect.bottom()]
    if x1>=0 and x2>=0 and y1>=0 and y2>=0:
        new_img=img[y1:y2,x1:x2]
        new_img=cv2.resize(new_img,(PIXEL,PIXEL))
        return new_img
    else:
        return img


def predict(img_path,detector,predictorr,classifier,ALIGN_POINTS):
    base_path=cv2.imread("/home/ai/AIApp/facereco/standard.jpg",cv2.IMREAD_COLOR)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    dets = detector(img,1)
    result_list=[]
    for i in range(len(dets)):
        out=out_put(dets[i],img,48)
        if out.shape==img.shape:
            continue 
        warped_im2=Image.fromarray(face(base_path,out)).convert('L')
        st_out=np.array(warped_im2)
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




def classes(result):
    class_name1 = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise','Null']
    class_name2 = ['愤怒','厌恶','惊恐','高兴','平静','悲伤','惊讶','无法识别']
    Chinese=""
    English=""
    if type(result)==int:
        Chinese='无法识别'
        English='Null'
    else:  
        for i in range(7):
            Chinese=Chinese+class_name2[i]+":"+str(format(result[0,i],'.2%'))+" "
            English=English+class_name1[i]+":"+str(format(result[0,i],'.2%'))+" "
    return Chinese,English






# test no error
img_path = 'test.bmp'


#dlib model
detector = dlib.get_frontal_face_detector() 
predictor_path = 'model/shape_predictor_68_face_landmarks.dat'
classifier_path='model/face_classifier.pkl'
predictor = dlib.shape_predictor(predictor_path)
classifier=joblib.load(classifier_path)

MOUTH_POINTS = list(range(48, 61))  # 嘴巴
RIGHT_BROW_POINTS = list(range(17, 22))  # 右眉毛
LEFT_BROW_POINTS = list(range(22, 27))  # 左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))  # 右眼睛
LEFT_EYE_POINTS = list(range(42, 48))  # 左眼睛
NOSE_POINTS = list(range(27, 35))  # 鼻子
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS+LEFT_EYE_POINTS+RIGHT_EYE_POINTS+NOSE_POINTS+MOUTH_POINTS)


def index(request):
    return render(request,'face.html', 'AI')

#表情识别
def emotion(request):
    url = request.GET['url']
    detect_url='media/detect/'+url.split('/')[-1]
    start1=time.time()
    ret=crop_image(url,detect_url,120)
    end1=time.time()
    print(end1-start1)
    if bool(ret)== False:
        content2 = 'Detected no face!'
        return HttpResponse(content2)
    start2=time.time()
    result_list= predict(url,detector,predictor,classifier,ALIGN_POINTS)
    end2=time.time()
    print(end2-start2)
    classes_ch="\r\n"
    classes_eg="\r\n"
    for result in result_list:
        ch,eg = classes(result)
        classes_ch=classes_ch+ch+"\r\n"
        classes_eg=classes_eg+eg+"\r\n"
    orgurl="media/upload/"+url.split('/')[-1]
    content = {
        'orgurl':orgurl,
        'url':detect_url,
        'result':classes_ch
    }
    return render(request, 'pic.html', content)

 
def uploadImg(request):
    if request.method == 'POST':
        new_img = IMG(
            img=request.FILES.get('img')            
        )       
        new_img.save()
        name=new_img.img.name
        url="media/" +name
        url='/home/ai/AIApp/'+url
        
        
        return HttpResponseRedirect('/emotion?url='+url)
    return render(request, 'uploadimg.html')


def change_size(rect,im,PIXEL):
    [x1,x2,y1,y2]=[rect.left(),rect.right(),rect.top(),rect.bottom()]
    new_img=im[y1:y2,x1:x2]
    new_img=cv2.resize(new_img,(PIXEL,PIXEL))
    return new_img
    
    

def fitter_dect(rects):
    new_rects=[]
    for rect in rects:
        [x1,x2,y1,y2]=[rect.left(),rect.right(),rect.top(),rect.bottom()]
        if x1>=0 and x2>=0 and y1>=0 and y2>=0:
            new_rects.append(rect)
    return new_rects   

def plot_landmarks(img,landmarkss):
    for idx, point in enumerate(landmarkss):
        pos = (point[0], point[1])
        cv2.circle(img, pos,2, color=(255,0,0))
    return img

def dect_show(merge_img):
    dets=detector(merge_img,1)
    for d in dets:
        landmarks=np.array([[p.x, p.y] for p in predictor(merge_img, d).parts()])
        plot_img=plot_landmarks(merge_img,landmarks)
    return plot_img  

  

def crop_image(from_image,to_image,PIXEL):
    im=cv2.imread(from_image)
    if len(im)<=PIXEL and len(im[0])<=PIXEL:
        cv2.imwrite(to_image,im)
        return True
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rects = detector(gray,1)
        rects=fitter_dect(rects)
        if len(rects)>=1:
            merge_img=change_size(rects[0],im,PIXEL)
            for index in range(1,len(rects)):
                resize_img=change_size(rects[index],im,PIXEL)
                merge_img=np.hstack((merge_img,resize_img))
            cv2.imwrite(to_image,dect_show(merge_img))
            return True
        else:
            print('Unable to crop "%s"' % from_image)
            return False