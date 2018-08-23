import tensorflow as tf
import matplotlib.pyplot as plt
from tiny_config import cfg
import numpy as np
from draw_boxes import draw_boxes
from tensorflow.python import debug as tf_debug
def parser(example):
    features = {
                'xywhc': tf.FixedLenFeature([150], tf.float32),
                'img': tf.FixedLenFeature((), tf.string)}
    feats = tf.parse_single_example(example, features)
    coord = feats['xywhc']
    coord = tf.reshape(coord, [30, 5])

    img = tf.decode_raw(feats['img'], tf.float32)
    img = tf.reshape(img, [cfg.train.image_hight_resized, cfg.train.image_width_resized, 3])
    # img = tf.reshape(img, [352, 480, 3])

    img = tf.image.resize_images(img, [cfg.train.image_hight_resized, cfg.train.image_width_resized]) #[new_height, new_width]
    # img = tf.image.resize_images(img, [cfg.train.image_resized , cfg.train.image_resized])
    # get a boolean value
    rnd = tf.less(tf.random_uniform(shape=[], minval=0, maxval=2), 1)

    def flip_img_coord(_img, _coord):
        zeros = tf.constant([[0, 0, 0, 0, 0]]*30, tf.float32)
        # flip the image by horizontaly(left to right)
        img_flipped = tf.image.flip_left_right(_img)
        idx_invalid = tf.reduce_all(tf.equal(coord, 0), axis=-1)
        coord_temp = tf.concat([tf.minimum(tf.maximum(1 - _coord[:, :1], 0), 1),
                               _coord[:, 1:]], axis=-1)
        coord_flipped = tf.where(idx_invalid, zeros, coord_temp)
        return img_flipped, coord_flipped

    img, coord = tf.cond(rnd, lambda: (tf.identity(img), tf.identity(coord)), lambda: flip_img_coord(img, coord))

    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)
    return img, coord


def data_pipeline(file_tfrecords, batch_size):
    dt = tf.data.TFRecordDataset(file_tfrecords)
    #
    dt = dt.map(parser, num_parallel_calls=4)
    # dt = dt.map(parser)
    dt = dt.prefetch(batch_size)
    dt = dt.shuffle(buffer_size=20*batch_size)
    dt = dt.repeat()
    dt = dt.batch(batch_size)
    iterator = dt.make_one_shot_iterator()
    imgs, true_boxes = iterator.get_next()

    return imgs, true_boxes


if __name__ == '__main__':
    # file_path = 'trainvalpart_all_100_512_288.tfrecords'
    # file_path = './tfrecords/test_one_.tfrecords'
    # file_path = './tfrecords/test_one_.tfrecords'
    file_path = './tfrecords/train_bdd100k_512_288_class3.tfrecords'
    # imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)
    imgs, true_boxes = data_pipeline(file_path, 1)
    sess = tf.Session()
    # debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    imgs_, true_boxes_ = sess.run([imgs, true_boxes])
    #draw_boxes(image, boxes, box_classes, class_names, scores=None):
    '''
    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.
    '''
    true = np.reshape(true_boxes_[..., 0:4], (30, 4))
    print(true)

    true[...,0:1] = true_boxes_[..., 0:1] * 512
    print(true)

    true[...,1:2] = true_boxes_[..., 1:2] * 288
    true[...,2:3] = true_boxes_[..., 2:3] * 512
    true[...,3:4] = true_boxes_[..., 3:4] * 288
    # true = np.reshape(true_boxes_[..., 0:4] , (30, 4))


    true1 = np.random.random((30,4))
    true1[:, 0] = true[:, 1] - true[:, 3] / 2  #y_main
    true1[:, 1] = true[:, 0] - true[:, 2] / 2  #x_min
    true1[:, 2] = true[:, 1] + true[:, 3] / 2  #y_max
    true1[:, 3] = true[:, 0] + true[:, 2] / 2  #x_max

    print(true1)
    # image_draw = draw_boxes(np.array(np.reshape(imgs_,(416,416,3)), dtype=np.float32) / 255, np.reshape(true_boxes_[...,0:4]*416,(30,4)), np.reshape(true_boxes_[...,4:5],(30,1)), cfg.names)
    # d
    image_draw = draw_boxes(np.reshape(imgs_,(288,512,3)), true1, list(np.reshape(true_boxes_[...,4:5],(-1))), cfg.names)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image_draw)
    plt.show()
