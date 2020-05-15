import xml.etree.ElementTree as ET
import os
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle as pkl

def get_info(file_name):
    file_info = {'frame_num': [], 'boxes':[]}
    ignore_area = {'file_name':file_name.split('.')[0], 'region':[]}
    doc = ET.parse(directory + file_name)
    root = doc.getroot()
    ignore_region = root.findall("ignored_region")
    for ign in ignore_region:
        for box in ign.findall('box'):
            ignore_area['region'].append({
                'x':box.attrib['left'],
                'y':box.attrib['top'],
                'w':box.attrib['width'],
                'h':box.attrib['height']
            })
    frame_list = []
    frames = root.findall("frame")
    for frame in frames:
        file_info['frame_num'].append(frame.attrib['num'])
        for target in frame.findall("target_list"):
            box_list = []
            for u in target.findall("target"):
                box = u.find('box').attrib
                box_list.append({
                    'x':box['left'],
                    'y':box['top'],
                    'w':box['width'],
                    'h':box['height'],
                    'label':1
                })
            file_info['boxes'].append(box_list)
    return (ignore_area, file_info)


def delete_ignore_area(img, img_igarea):
    for box in img_igarea:
        x = int(float(box['x'])+1)
        y = int(float(box['y'])+1)
        w = int(float(box['w']))
        h = int(float(box['h']))
        img[x : x+w , y:y+h] = 0
    return img

def read_data(camera_angle):
    img_list = []
    box_value = []
    angle_number = [str(pixel_num).zfill(5)[-5:] for pixel_num in range(1, len(camera_angle[1]['frame_num'])+1)]
    for frame_num in angle_number:
        img_file_name = img_directory + camera_angle[0]['file_name'] + '/img' + frame_num + '.jpg'
        img = cv2.imread(img_file_name)
        img = delete_ignore_area(img, camera_angle[0]['region'])
        img_list.append(img)
    for step_box in camera_angle[1]['boxes']:
        num_box = []
        for box in step_box:
            x,y,w,h = float(box['x'])-1, float(box['y'])-1, float(box['w'])+1, float(box['h'])+1
            box = [x, y, x+w, y+h , 1, 1]
            num_box.append(box)
        box_value.append(num_box)
    img_list = np.array(img_list)
    box_value = np.array(box_value) 
    return img_list, box_value

