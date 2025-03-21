from __future__ import division
from __future__ import print_function

import time
import os
import sys
import json
import random
import nvidia_smi
import numpy as np
from utils import *
from metrics import *
from models import GCN_Align
import tensorflow.compat.v1 as tf

def seed_everything(seed=1011):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def args_flags():
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 40, 'Initial learning rate.')
    # flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
    flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('gamma', 6.0, 'Hyper-parameter for margin based loss.')
    flags.DEFINE_integer('k', 1, 'Number of negative samples for each positive seed.')
    flags.DEFINE_float('beta', 0.9, 'Weight for structure embeddings.')
    flags.DEFINE_integer('se_dim', 128, 'Dimension for SE.')
    flags.DEFINE_float('train_ratio', 0.3, 'ratio for taining.')

    flags.DEFINE_string('data_dir', "../data/", 'data directory')
    flags.DEFINE_string('data_name', "DBP15K/zh_en/", "input dataset name")
    flags.DEFINE_string('output_dir', "../results/GCN_Align/", 'output directory')
    flags.DEFINE_integer('max_train_epoch', 200, 'max training epoch num')
    flags.DEFINE_integer('tf_gpu_no', 0, 'gpu device no')
    FLAGS(sys.argv)
    return FLAGS

def save_gpu_mem(tf_gpu_no, output_dir, txt="gpu_mem_usage_before"):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(tf_gpu_no)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    with open(os.path.join(output_dir, "running.log"), "a+") as file:
        msg = {"msg_type": txt, "value": info.used/1024/1024}
        file.write(json.dumps(msg)+"\n")

if __name__ == "__main__":
    tf.disable_v2_behavior()

    nvidia_smi.nvmlInit()
    seed_everything()
    # Set random seed
    seed = 12306
    np.random.seed(seed)
    tf.set_random_seed(seed)

    FLAGS = args_flags()
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir

    save_gpu_mem(FLAGS.tf_gpu_no, output_dir)

    # Load data
    adj, e, e1, e2, train, test = load_data(data_dir)

    # Some preprocessing
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_Align
    k = FLAGS.k

    # Define placeholders
    # ph_ae = {
    #     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    #     'features': tf.sparse_placeholder(tf.float32), #tf.placeholder(tf.float32),
    #     'dropout': tf.placeholder_with_default(0., shape=()),
    #     'num_features_nonzero': tf.placeholder_with_default(0, shape=())
    # }
    ph_se = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder_with_default(0, shape=())
    }

    # Initialize session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create model
    # model_ae = model_func(ph_ae, input_dim=ae_input[2][1], output_dim=FLAGS.ae_dim, ILL=train, sparse_inputs=True, featureless=False, logging=True)
    model_se = model_func(ph_se, input_dim=e, output_dim=FLAGS.se_dim, ILL=train, sparse_inputs=False, featureless=True, logging=True)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    t = len(train)

    # Train model
    for epoch in range(FLAGS.max_train_epoch):
        if epoch % 10 == 0:
            neg_left = np.random.choice(e1, t * k)
            neg_right = np.random.choice(e2, t * k)
        # Construct feed dictionary
        # feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
        # feed_dict_ae.update({ph_ae['dropout']: FLAGS.dropout})
        # feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
        feed_dict_se = construct_feed_dict(1.0, support, ph_se)
        feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
        feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right})
        # Training step
        # outs_ae = sess.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
        outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)
        # cost_val.append((outs_ae[1], outs_se[1]))
        cost_val.append(outs_se[1])
        tf.print(cost_val)

        # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "AE_train_loss=", "{:.5f}".format(outs_ae[1]), "SE_train_loss=", "{:.5f}".format(outs_se[1]))
        # print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=", "{:.5f}".format(outs_se[1]))

    print("Optimization Finished!")

    # Testing
    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se)
    print("SE")
    get_hits(vec_se, test)

    save_gpu_mem(FLAGS.tf_gpu_no, output_dir, "gpu_mem_usage_after")

    # save embeddings
    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    vec = sess.run(model_se.outputs, feed_dict=feed_dict_se)
    vec = np.array(vec) + 1e-9
    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True, ord=2)
    with open(os.path.join(data_dir, "ent_ids_1")) as file:
        lines = file.read().strip().split("\n")
        ent1_id_list = [int(line.split()[0]) for line in lines]
    with open(os.path.join(data_dir, "ent_ids_2")) as file:
        lines = file.read().strip().split("\n")
        ent2_id_list = [int(line.split()[0]) for line in lines]
    ent1_ids = np.array(ent1_id_list)
    ent2_ids = np.array(ent2_id_list)
    np.savez(os.path.join(output_dir, "emb.npz"), embs=vec, ent1_ids=ent1_ids, ent2_ids=ent2_ids)