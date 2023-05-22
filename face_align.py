# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:16:08 2018
"""

import cv2
import dlib
import numpy


FACE_POINTS = list(range(17, 68))  # 脸
MOUTH_POINTS = list(range(48, 61))  # 嘴巴
RIGHT_BROW_POINTS = list(range(17, 22))  # 右眉毛
LEFT_BROW_POINTS = list(range(22, 27))  # 左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))  # 右眼睛
LEFT_EYE_POINTS = list(range(42, 48))  # 左眼睛
NOSE_POINTS = list(range(27, 35))  # 鼻子
JAW_POINTS = list(range(0, 17))  # 下巴



ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS+LEFT_EYE_POINTS+RIGHT_EYE_POINTS+NOSE_POINTS+MOUTH_POINTS)
EYE_POINTS=(LEFT_EYE_POINTS+RIGHT_EYE_POINTS)


# 加载训练模型
# 并实例化特征提取器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
SCALE_FACTOR=1



def transformation_from_points(points1, points2):
    """
    标准化
    """
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T #旋转矩阵

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    '''
    由 get_face_mask 获得的图像掩码还不能直接使用，因为一般来讲用户提供的两张图像的分辨率大小很可能不一样，而且即便分辨率一样，
    图像中的人脸由于拍摄角度和距离等原因也会呈现出不同的大小以及角度，所以如果不能只是简单地把第二个人的面部特征抠下来直接放在第一个人脸上，
    我们还需要根据两者计算所得的面部特征区域进行匹配变换，使得二者的面部特征尽可能重合。
    
    仿射函数，warpAffine，能对图像进行几何变换
        三个主要参数，第一个输入图像，第二个变换矩阵 np.float32 类型，第三个变换之后图像的宽高
    
    对齐主要函数
    '''
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def get_landmarks(im):#获得68个关键点的坐标
    detector = dlib.get_frontal_face_detector()
    rects = detector(im, 1)    
    if len(rects)<1:
        raise IndexError
    else:
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])   # 68*2的矩阵


def read_im_and_landmarks(im):
#    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))                        
    s = get_landmarks(im)
    return im, s



def face(Base_path,cover_path):
    im1, landmarks1 = read_im_and_landmarks(Base_path)  # 底图
    im2, landmarks2 = read_im_and_landmarks(cover_path) # 贴上来的图
    if len(landmarks2)==0:
        raise IndexError
    M = transformation_from_points(landmarks1[ALIGN_POINTS],landmarks2[ALIGN_POINTS])#选择眼睛特征作为
    warped_im2 = warp_im(im2, M, im1.shape)
    return warped_im2




