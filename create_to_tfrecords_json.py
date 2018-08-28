import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import sys
import glob
import cv2

target_objects = ['bus','truck','car']
width = 1280
height = 720

def convert_img(image_id):
    image = Image.open(image_id)
    resized_image = image.resize((512, 288), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    img_raw = image_data.tobytes()
    return img_raw

# return the scale size of object boubdingbox to [0,1] with respecet to the whole box
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]

def convert_annotation(image_id):
    with open(image_id) as json_file:
        label = json.load(json_file)
    objects = label['frames'][0]['objects']
    need_objects = [o for o in objects if o['category'] in target_objects]
    if len(need_objects) == 0:
        sys.exit()
    all_box2d = [o for o in need_objects if 'box2d' in o]
    bboxes = []
    # object['box2d']=={'x2': 357.805838, 'x1': 45.240919, 'y2': 487.906215, 'y1': 254.530367}
    for object in all_box2d:
        b = (float(object['box2d']['x1']), float(object['box2d']['x2']), float(object['box2d']['y1']), float(object['box2d']['y2']))
        cls_id = target_objects.index(object['category'])
        # print(cls_id)
        bb = convert((width, height), b) + [cls_id]
        bboxes.extend(bb)
        # print(object['category'], object['box2d'])
    if len(bboxes) < 30 * 5:
        bboxes = bboxes + [0, 0, 0, 0, 0]*(30-int(len(bboxes)/5))
    return np.array(bboxes,dtype=np.float32).flatten().tolist()

def reName(dirname,xml_file_name):

    file_handle = open(xml_file_name, mode='w+')
    flag1 = False
    flag2 = False
    # if not os.path.exists(filename_path):
    #     os.mknod(filename_path)
    # print(os.listdir(dirname))
    # print(os.listdir(dirname)[-2])
    #print(os.listdir(dirname))
    for category in os.listdir(dirname):
        # print(category)
        catdir = os.path.join(dirname,category)
        if not os.path.isdir(catdir):
            continue
        if category == "Annotations":
            flag1 = True
        if category == "Original":
            flag2 = True
        # print(flag)
        files = os.listdir(catdir)
        files.sort()
        # print(files)
        # files.remove('.DS_Store')
        count = 0
        for cur_file in files:
            print("handle" + category + " " + cur_file)
            filename = os.path.join(catdir,cur_file)
            count = count + 1
            oldDir = os.path.join(catdir,cur_file)
            if os.path.isdir(oldDir):
                continue
            filename=os.path.splitext(cur_file)[0]
            filetype=os.path.splitext(cur_file)[1]
            # newDir=os.path.join(catdir,str(count)+filetype)
            # newDir=os.path.join(catdir,str((6-len(str(count))))*'0'+str(count)+filetype)
            newDir=os.path.join(catdir,str(count).zfill(6)+filetype)
            os.rename(oldDir,newDir)
            #write to file
            if flag1 and flag2:
                file_handle.write(str(count).zfill(6)+'\n')
    file_handle.close()


def preHandle(images_path,annos_path):
    file_list = glob.glob(annos_path+"*.json")
    # image_ids = open(xml_file_name).read().strip().split()
    for image_id in file_list:
        # with open(os.path.join(annos_path,image_id+".json")) as json_file:
        with open(image_id) as json_file:
            label = json.load(json_file)
        objects = label['frames'][0]['objects']
        need_objects = [o for o in objects if o['category'] in target_objects]
        if len(need_objects) == 0 or len(need_objects) >= 30:
            os.remove(image_id)
            image_name = os.path.basename(image_id).split(".")[0]
            del_path = os.path.join(images_path,image_name+".jpg")
            os.remove(del_path)
def main(images_path, annos_path, tfrecords_name,xml_file_name):
    filename = os.path.join(tfrecords_name)
    writer = tf.python_io.TFRecordWriter(filename)

    image_ids = open(xml_file_name).read().strip().split()
    count = 0
    for image_id in image_ids:
        image_path = os.path.join(images_path,image_id+".jpg")
        anno_path = os.path.join(annos_path,image_id+".json")
        img = cv2.imread(image_path)
        shape = img.shape
        height = shape[0]
        width = shape[1]

        xywhc = convert_annotation(anno_path)
        img_raw = convert_img(image_path)
        example = tf.train.Example(features=tf.train.Features(feature={
            'xywhc':
                tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
            'img':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())
        count = count + 1
        print(count)
    writer.close()

def resize(image_path,resize_width,resize_height):
    images_list = glob.glob(image_path+"*.jpg")
    for image in images_list:
        img = Image.open(image)
        img = img.resize([resize_width,resize_height],Image.BICUBIC)
        img.save(image)


if __name__ == "__main__":
    resize_width = 512
    resize_height = 288
    images_path = "F:/ourtogether/allin/train/Annotations/"
    annos_path = "F:/ourtogether/allin/train/Original/"
    # tfrecords_name = "demo.tfrecords"
    tfrecords_name = "F:/ourtogether/allin/train/train_allin.tfrecords"
    dirname = "F:/ourtogether/allin/train/"
    xml_file_name = "F:/ourtogether/allin/train/trainval.txt"

    # preHandle(images_path, annos_path)

    # resize(images_path, resize_width, resize_height)
    reName(dirname,xml_file_name)
    main(images_path,annos_path,tfrecords_name,xml_file_name)
