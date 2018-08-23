from tiny_yolo_top import tiny_yolov3
import numpy as np
import tensorflow as tf
from tiny_config import cfg
from PIL import Image, ImageDraw, ImageFont
from draw_boxes import draw_boxes
import matplotlib.pyplot as plt
import glob
import cv2
import scipy.misc as misc


def video_picture(video_path, save_image_path):
    vc = cv2.VideoCapture(video_path)
    c = 0
    rval = vc.isOpened()
    # timeF = 1
    while rval:
        c = c + 1
        rval, frame = vc.read()
        if rval:
            cv2.imwrite(save_image_path + str(c).zfill(8) + '.jpg', frame)  # 存储为图像
            cv2.waitKey(1)
        else:
            break
    vc.release()


def predict(ckpt_dir,input_image_path,out_image_path):
    imgs_holder = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
    istraining = tf.constant(False, tf.bool)
    cfg.batch_size = 1
    cfg.scratch = True

    model = tiny_yolov3(imgs_holder, None, istraining)
    img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
    boxes, scores, classes = model.pedict(img_hw, iou_threshold=0.5, score_threshold=0.5)

    saver = tf.train.Saver()
    # ckpt_dir = './ckpt/ckpt_demo_416_class3/'
    # ckpt_dir = './ckpt/train_bdd100k_512_288_class3/'
    # ckpt_dir = './ckpt/demo1/'

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        # images = glob.glob()
        images = glob.glob(input_image_path + "*.jpg")
        images.sort()
        print(images)
        count = 0
        for image in images:
            # image_test = Image.open('image/000001.jpg')
            image_test = Image.open(image)
            resized_image = image_test.resize((416, 416), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            boxes_, scores_, classes_ = sess.run([boxes, scores, classes],
                                                 feed_dict={
                                                     img_hw: [image_test.size[1], image_test.size[0]],
                                                     imgs_holder: np.reshape(image_data / 255, [1, 416, 416, 3])})

            image_draw = draw_boxes(np.array(image_test, dtype=np.float32) / 255, boxes_, classes_, cfg.names,
                                    scores=scores_)
            img = Image.fromarray(image_draw.astype('uint8')).convert('RGB')

            misc.imsave(out_image_path+'out%d.jpg' % (count), img)
            count = count + 1

    def image_Video(input_image_path,out_avi):
        fps = 25  # 保存视频的FPS，可以适当调整

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(out_avi, fourcc, fps, (1906, 1080))  # 最后一个是保存图片的尺寸
        imgs = glob.glob(input_image_path+'*.jpg')
        for imgname in imgs:
            frame = cv2.imread(imgname)
            videoWriter.write(frame)
        videoWriter.release()
# if __name__ == "__main__":
    # video_picture(video_path, save_image_path):
