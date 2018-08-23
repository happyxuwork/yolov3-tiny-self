from tiny_yolo_top import tiny_yolov3
import numpy as np
import tensorflow as tf
from tiny_data_pipeline import data_pipeline
from tiny_config import cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

# file_path = './tfrecords/train_gao_.tfrecords'
file_path = 'F:/ourtogether/allin/train/train_allin.tfrecords'
# load the images and true boxes
imgs, true_boxes = data_pipeline(file_path, cfg.batch_size)
# set the progress to be training
istraining = tf.constant(True, tf.bool)
# build the model
model = tiny_yolov3(imgs, true_boxes, istraining)

loss = model.compute_loss()
# iterator the counter==so it can not trainable
global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
# the doc has a detail statement
lr = tf.train.piecewise_constant(global_step, [10000, 20000], [1e-3, 1e-4, 1e-5])
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
# for var in vars_det:
#     print(var)
with tf.control_dependencies(update_op):
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
saver = tf.train.Saver()
# ckpt_dir = './ckpt_tiny_all_100/'
ckpt_dir = './ckpt/train_bdd100k_512_288_class3/'

gs = 0
batch_per_epoch = 8604
cfg.train.max_batches = int(batch_per_epoch * 10) 
# cfg.train.image_resized = 416
cfg.train.image_width_resized = 512   # { 320, 352, ... , 608} multiples of 32
cfg.train.image_hight_resized = 288  # { 320, 352, ... , 608} multiples of 32

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        #
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 1 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 20)
# cfg.train.image_resized = 512
cfg.train.image_width_resized = 608   # { 320, 352, ... , 608} multiples of 32
cfg.train.image_hight_resized = 342  # { 320, 352, ... , 608} multiples of 32
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        # add 1 to iterator==
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 30)
# cfg.train.image_resized = 320
cfg.train.image_width_resized = 576   # { 320, 352, ... , 608} multiples of 32
cfg.train.image_hight_resized = 324  # { 320, 352, ... , 608} multiples of 32

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 40)
# cfg.train.image_resized = 352
cfg.train.image_width_resized = 352   # { 320, 352, ... , 608} multiples of 32
cfg.train.image_hight_resized = 198  # { 320, 352, ... , 608} multiples of 32
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000 == 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 50)
# cfg.train.image_resized = 480
cfg.train.image_width_resized = 480   # { 320, 352, ... , 608} multiples of 32
cfg.train.image_hight_resized = 270  # { 320, 352, ... , 608} multiples of 32
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000== 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)

cfg.train.max_batches = int(batch_per_epoch * 70)
# cfg.train.image_resized = 480
cfg.train.image_width_resized = 512   # { 320, 352, ... , 608} multiples of 32
cfg.train.image_hight_resized = 288  # { 320, 352, ... , 608} multiples of 32
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
        print('Restore batch: ', gs)
    else:
        print('no checkpoint found')
        sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if(i % 1000== 0):
            saver.save(sess, ckpt_dir+'yolov3.ckpt', global_step=global_step, write_meta_graph=False)
