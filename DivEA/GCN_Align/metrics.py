import tensorflow.compat.v1 as tf
import numpy as np
import scipy
import pdb

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name+":0")
    except:
        return tf.placeholder(tf.int32, name=name)


def align_loss(outlayer, ILL, gamma, k):
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)

    neg_left = get_placeholder_by_name("neg_left") #tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = get_placeholder_by_name("neg_right") #tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)

    neg_l_x = tf.reduce_mean(tf.reshape(neg_l_x, [t, k, -1]), 1)
    neg_r_x = tf.reduce_mean(tf.reshape(neg_r_x, [t, k, -1]), 1)

    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    B = tf.reduce_sum(tf.abs(left_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, 1])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

    B = tf.reduce_sum(tf.abs(neg_l_x - right_x), 1)
    C = - tf.reshape(B, [t, 1])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

    return tf.reduce_sum(L1 + L2) / (k * t)

def z_score(embed):
    mean, std = tf.nn.moments(embed, axes=[0])
    return (embed - mean) / tf.sqrt(std)

def infoNce1(outlayer, ILL):
    left = ILL[:, 0]
    right = ILL[:, 1]
    ll = tf.nn.embedding_lookup(outlayer, left)
    ll = z_score(ll)
    rr = tf.nn.embedding_lookup(outlayer, right)
    rr = z_score(rr)
    temp_node = 0.1
    normalize_user_emb1 = tf.nn.l2_normalize(ll, 1)
    normalize_user_emb2 = tf.nn.l2_normalize(rr, 1)
    normalize_all_user_emb2 = normalize_user_emb2
    pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
    ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
    pos_score_user = tf.exp(pos_score_user / temp_node)
    ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / temp_node), axis=1)
    cl_loss_user = -tf.reduce_mean(tf.compat.v1.log(pos_score_user / ttl_score_user))

    return cl_loss_user

def info_nce_loss(anchor, positive, negative, temperature=0.1):
    """
    InfoNCE loss function for contrastive learning.

    Parameters:
    - anchor: Tensor, embeddings for anchor samples.
    - positive: Tensor, embeddings for positive samples.
    - negative: Tensor, embeddings for negative samples.
    - temperature: Float, temperature parameter for softmax.

    Returns:
    - loss: Tensor, InfoNCE loss.
    """
    # anchor = tf.nn.l2_normalize(anchor, 1)
    # positive = tf.nn.l2_normalize(positive, 1)
    # negative = tf.nn.l2_normalize(negative, 1)

    pos_score_user = tf.reduce_sum(tf.multiply(anchor, positive), axis=1)
    ttl_score_user = tf.matmul(anchor, negative, transpose_a=False, transpose_b=True)
    pos_score_user = tf.exp(pos_score_user / temperature)
    ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / temperature), axis=1)
    # Compute InfoNCE loss
    cl_loss_user = -tf.reduce_mean(tf.compat.v1.log(pos_score_user / ttl_score_user))

    return cl_loss_user

def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    # sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))

def get_combine_hits(se_vec, ae_vec, beta, test_pair, top_k=(1, 10, 50, 100)):
    vec = np.concatenate([se_vec*beta, ae_vec*(1.0-beta)], axis=1)
    get_hits(vec, test_pair, top_k)
