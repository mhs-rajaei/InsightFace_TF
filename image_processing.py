# -*-coding: utf-8 -*-
"""
    https://github.com/PanJinquan/Face_Detection_Recognition
    @Project: IntelligentManufacture
    @File   : image_processing.py
    @Author : panjq
    @ E-mail: pan_jinquan@163.com
    @Date   : 2019-02-14 15:34:50
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def show_batch_image(title,batch_imgs,index=0):
    image = batch_imgs[index, :]
    # image = image.numpy()  #
    image = np.array(image, dtype=np.float32)
    image=np.squeeze(image)
    if len(image.shape)==3:
        Image = image.transpose(1, 2, 0) # Channel by [c,h,w]->[h,w,c]
    else:
        image = image.transpose(1, 0)
    cv_show_image(title,image)


def show_image(title, rgb_image):
    '''
    Call matplotlib to display RGB images
    :param title: image title
    :param rgb_image: image data
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    channel=len(rgb_image.shape)
    if channel==3:
        plt.imshow(rgb_image)
    else :
        plt.imshow(rgb_image, cmap='gray')
    plt.axis('on') # Turn off the axis off
    plt.title(title) # image title
    plt.show()


def cv_show_image(title, image, type='rgb'):
    '''
    Call OpenCV to display RGB images
    :param title: image title
    :param image: Input RGB image
    :param type:'rgb' or 'bgr'
    :return:
    '''
    channels=image.shape[-1]
    if channels==3 and type=='rgb':
        Image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert BGR to RGB
    cv2.imshow(title, image)
    cv2.waitKey(0)


def get_prewhiten_image(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def image_normalization(image,mean=None,std=None):
    # Cannot be written as: image=image/255
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    if mean is not None:
        image=np.subtract(image, mean)
    if std is not None:
        np.multiply(image, 1 / std)
    return image


def get_prewhiten_images(images_list,normalization=False):
    out_images=[]
    for image in images_list:
        if normalization:
            image=image_normalization(image)
        image=get_prewhiten_image(image)
        out_images.append(image)
    return out_images


def read_image(filename, resize_height=None, resize_width=None, normalization=False, colorSpace='RGB'):
    """
    Read image data, the default is uint8, [0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization: whether normalized to [0.,1.0]
    :param colorSpace Output format: RGB or BGR
    :return: returned image data
    """

    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning: does not exist: {}", filename)
        return None
    if len(bgr_image.shape) == 2: # If it is a grayscale image, it will be converted to three channels.
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    if colorSpace=='RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    elif colorSpace=="BGR":
        image = bgr_image
    else:
        exit(0)
    # show_image(filename,image)
    # image=Image.open(filename)
    image = resize_image(image,resize_height,resize_width)
    image = np.asanyarray(image)
    if normalization:
        image=image_normalization(image)
    # show_image("src resize image",image)
    return image


def read_image_gbk(filename, resize_height=None, resize_width=None, normalization=False, colorSpace='RGB'):
    '''
    Solve the problem that imread can't read Chinese path, read image data, the default is uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization: whether normalized to [0.,1.0]
    :param colorSpace Output format: RGB or BGR
    :return: returned RGB image data
    '''
    with open(filename, 'rb') as f:
        data = f.read()
        data = np.asarray(bytearray(data), dtype="uint8")
        bgr_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # or:
    # bgr_image=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning: does not exist: {}", filename)
        return None
    if len(bgr_image.shape) == 2: # If it is a grayscale image, it will be converted to three channels.
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    if colorSpace=='RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    elif colorSpace=="BGR":
        image=bgr_image
    else:
        exit(0)
    # show_image(filename,image)
    # image=Image.open(filename)
    image = resize_image(image,resize_height,resize_width)
    image = np.asanyarray(image)
    if normalization:
        image=image_normalization(image)
    # show_image("src resize image",image)
    return image


def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, normalization=False,colorSpace='RGB'):
    '''
    Quick way to read pictures
    :param filename: image path
    :param orig_rect: the region of interest of the original image rect
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param normalization: normalization
    :param colorSpace Output format: RGB or BGR
    :return: Returns the ROI of the region of interest
    '''
    # When using IMREAD_REDUCED mode, the corresponding rect also needs to be scaled.
    scale=1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale=1/2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale=1/4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale=1/8
    rect = np.array(orig_rect)*scale
    rect = rect.astype(int).tolist()
    bgr_image = cv2.imread(filename,flags=ImreadModes)

    if bgr_image is None:
        print("Warning: does not exist: {}", filename)
        return None
    if len(bgr_image.shape) == 2: # If it is a grayscale image, it will be converted to three channels.
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    if colorSpace == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    elif colorSpace == "BGR":
        image = bgr_image
    image = np.asanyarray(image)
    if normalization:
        image=image_normalization(image)
    roi_image=get_rect_image(image , rect)
    # show_image_rect("src resize image",rgb_image,rect)
    # cv_show_image("reROI",roi_image)
    return roi_image


def resize_image(image,resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]
    if (resize_height is None) and (resize_width is None):#错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height=int(height*resize_width/width)
    elif resize_width is None:
        resize_width=int(width*resize_height/height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image


def scale_image(image,scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image,dsize=None, fx=scale[0],fy=scale[1])
    return image


def get_rect_image(image,rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    shape=image.shape#h,w
    height=shape[0]
    width=shape[1]
    image_rect=(0,0,width,height)
    rect=get_rect_intersection(rect, image_rect)
    x, y, w, h=rect
    cut_img = image[y:(y+ h),x:(x+w)]
    return cut_img


def get_rects_image(image,rects_list,resize_height=None, resize_width=None):
    rect_images = []
    for rect in rects_list:
        roi=get_rect_image(image, rect)
        roi=resize_image(roi, resize_height, resize_width)
        rect_images.append(roi)
    return rect_images


def get_bboxes_image(image,bboxes_list,resize_height=None, resize_width=None):
    rects_list=bboxes2rects(bboxes_list)
    rect_images = get_rects_image(image,rects_list,resize_height, resize_width)
    return rect_images


def bboxes2rects(bboxes_list):
    '''
    Convert bboxes=[x1,y1,x2,y2] to rect=[x1,y1,w,h]
    :param bboxes_list:
    :return:
    '''
    rects_list=[]
    for bbox in bboxes_list:
        x1, y1, x2, y2 = bbox
        rect=[ x1, y1,(x2-x1),(y2-y1)]
        rects_list.append(rect)
    return rects_list


def rects2bboxes(rects_list):
    '''
    Convert rect=[x1,y1,w,h] to bboxes=[x1,y1,x2,y2]
    :param rects_list:
    :return:
    '''
    bboxes_list=[]
    for rect in rects_list:
        x1, y1, w, h = rect
        x2=x1+w
        y2=y1+h
        b=(x1,y1,x2,y2)
        bboxes_list.append(b)
    return bboxes_list


def scale_rect(orig_rect,orig_shape,dest_shape):
    '''
    When scaling an image, the corresponding rectangle is also scaled.
    :param orig_rect: original image of rect=[x,y,w,h]
    :param orig_shape: dimension of the original image shape=[h,w]
    :param dest_shape: Dimensions of the scaled image shape=[h,w]
    :return: scaled rectangle
    '''
    new_x=int(orig_rect[0]*dest_shape[1]/orig_shape[1])
    new_y=int(orig_rect[1]*dest_shape[0]/orig_shape[0])
    new_w=int(orig_rect[2]*dest_shape[1]/orig_shape[1])
    new_h=int(orig_rect[3]*dest_shape[0]/orig_shape[0])
    dest_rect=[new_x,new_y,new_w,new_h]
    return dest_rect


def get_rect_intersection(rec1,rec2):
    '''
    Calculate the intersection coordinates of two rect
    : param rec1:
    : param rec2:
    :return:
    '''
    cx1, cy1, cx2, cy2 = rects2bboxes([rec1])[0]
    gx1, gy1, gx2, gy2 = rects2bboxes([rec2])[0]
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min (cx2, gx2)
    y2 = min (cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return (x1,y1,w,h)


def show_image_bboxes_text(title, rgb_image, boxes, boxes_name):
    '''
    :param boxes_name:
    :param bgr_image: bgr image
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    for name, box in zip(boxes_name,boxes):
        box=[int(b) for b in box]
        cv2.rectangle(bgr_image, (box[0],box[1]),(box[2],box[3]), (0, 255, 0), 2, 8, 0)
        cv2.putText(bgr_image,name, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
    # cv2.imshow(title, bgr_image)
    # cv2.waitKey(0)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv_show_image(title, rgb_image)


def show_image_rects_text(title, rgb_image, rects_list, boxes_name):
    '''
    :param boxes_name:
    :param bgr_image: bgr image
    :param boxes: [[x1,y1,w,h],[x1,y1,w,h]]
    :return:
    '''
    bbox_list = rects2bboxes(rects_list)
    show_image_bboxes_text(title, rgb_image, bbox_list, boxes_name)


def show_image_rects(win_name,image,rect_list):
    '''
    :param win_name:
    :param image:
    :param rect_list:[[ x, y, w, h],[ x, y, w, h]]
    :return:
    '''
    for rect in rect_list:
        x, y, w, h=rect
        point1=(int(x),int(y))
        point2=(int(x+w),int(y+h))
        cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)


def show_landmark_boxex(win_name,img,landmarks_list,boxes):
    '''
    Show landmark and boxex
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    image=copy.copy(img)
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4 # can be 0, 4, 8
    for landmarks in landmarks_list:
        for landmark in landmarks:
            # coordinates of the point to draw
            point = (landmark[0],landmark[1])
            cv2.circle(image, point, point_size, point_color, thickness)
    show_image_boxes(win_name, image, boxes)


def show_image_boxes(win_name,image,boxes_list):
    '''
    :param win_name:
    :param image:
    :param boxes_list:[[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    for box in boxes_list:
        x1, y1, x2, y2=box
        point1=(int(x1),int(y1))
        point2=(int(x2),int(y2))
        cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    show_image(win_name, image)


def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def save_image(image_path, rgb_image,toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2: # If it is a grayscale image, it will be converted to three channels.
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)


def combime_save_image(orig_image, dest_image, out_dir,name,prefix):
    '''
    Naming standard: out_dir/name_prefix.jpg
    :param orig_image:
    : param dest_image:
    :param image_path:
    : param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_"+prefix+".jpg")
    save_image(dest_path, dest_image)

    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name,prefix)), dest_image)


def combile_label_prob(label_list,prob_list):
    '''
    Splicing label_list and prob_list together for display
    :param label_list:
    :param prob_list:
    :return:
    '''
    info = [l + ":" + str(p) for l, p in zip(label_list,prob_list)]
    return info


if __name__=="__main__":
    # image_path="../dataset/test_images/lena1.jpg"
    # image_path="E:/git/dataset/tgs-salt-identification-challenge/train/my_masks/4.png"
    image_path = 'E:/Face/dataset/bzl/test3/test_dataset/陈思远_716/8205_0.936223.jpg'

    # target_rect=main.select_user_roi(target_path)#rectangle=[x,y,w,h]
    # orig_rect = [50, 50, 100000, 10000]

    image = read_image_gbk(image_path, resize_height=None, resize_width=None)
    # orig_image=get_rect_image(image,orig_rect)

    # show_image_rects("image",image,[orig_rect])
    show_image("orig_image",image)

