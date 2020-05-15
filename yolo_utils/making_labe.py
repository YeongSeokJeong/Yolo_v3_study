import os
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import numpy as np

def scailling_img(input_img, box_data, output_size = (412,412), max_box = 10):
    resize_image_list = []
    ih, iw= input_img.shape[:2] # original image shape
    h, w = output_size # target image shape
    box = np.array(box_data)

    scale = min(w/iw, h/ih) # scaliling size
    nw = int(iw*scale) # 이미지 크기 조절을위한 계산
    nh = int(ih*scale) 
    dx = (w-nw)//2
    dy = (h-nh)//2

    image = cv2.resize(input_img, (nw,nh), cv2.INTER_LINEAR)
    new_img = np.full((h,w,3), (128,128,128))
    new_img[dy:dy + image.shape[0], dx:dx + image.shape[1], ::] = image
    
    processed_box_data = np.zeros((max_box, 6))
    if len(box_data) > 0:
        np.random.shuffle(box_data)
        if len(box_data) > max_box:
            box_data = box_data[:10]
        box[:, [0,2]] = box[:, [0, 2]] * scale + dx
        box[:, [1,3]] = box[:, [1, 3]] * scale + dy
        processed_box_data[:len(box_data)] = box
        
    return new_img, processed_box_data
    

def read_anchors(file_path):
    anchors_list = []
    with open(file_path, 'r') as f:
        anchors = f.readline()
        for anchor in anchors.split():
            w,h = anchor.split(',')[:2]
            anchors_list.append([int(w), int(h)])
    return np.array(anchors_list)

def make_true_box(box_data, input_shape, anchors):
    num_layers = len(anchors)//3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers == 3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array([box_data])
    input_shape = (412, 412)

    true_boxes = np.array(true_boxes, dtype = 'float32')
    input_shape = np.array(input_shape, dtype = 'int32')
    boxes_xy = (true_boxes[...,0:2] + true_boxes[..., 2:4])//2
    boxes_wh = true_boxes[...,2:4] - true_boxes[..., 0:2]

    # print(boxes_xy)
    # print(boxes_wh)


    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0] # box 개수

    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]

    y_true = [np.zeros((10,grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + 1), dtype = 'float32') 
              for l in range(num_layers)]

    anchors = np.expand_dims(anchors, 0)
    anchors_maxes = anchors/2.
    anchor_mins = -anchors_maxes

    valid_mask = boxes_wh[...,0]>0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue # none box
        wh = np.expand_dims(wh, -2)
        print(wh)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        print(box_maxes)
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchors_maxes)

        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area/(box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis = -1) # index of best anchor
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t,4].astype('int32')
                    y_true[l][b,j,i,k,0:4] = true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1
                    y_true[l][b,j,i,k,5] = 1
    return y_true