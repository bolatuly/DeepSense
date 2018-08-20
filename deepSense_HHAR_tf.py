import tensorflow as tf
import numpy as np
import random
import csv
import os

layers = tf.contrib.layers

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES * 6 * 2
CONV_LEN = 3
CONV_LEN_INTE = 3  # 4
CONV_LEN_LAST = 3  # 5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 4  # len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 0

DATA_NUM = 1120


# TRAIN_SIZE = metaDict[select][0]
# EVAL_DATA_SIZE = metaDict[select][1]
# EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))


###### Import training data
def read_audio_csv(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    defaultVal = [[0.] for idx in range(WIDE * FEATURE_DIM + OUT_DIM)]

    fileData = tf.decode_csv(value, record_defaults=defaultVal)
    features = fileData[:WIDE * FEATURE_DIM]
    features = tf.reshape(features, [WIDE, FEATURE_DIM])
    labels = fileData[WIDE * FEATURE_DIM:]
    return features, labels


def input_pipeline(filenames, batch_size, shuffle_sample=True):
    filename_queue = tf.train.string_input_producer(filenames)
    example, label = read_audio_csv(filename_queue)
    min_after_dequeue = 1000  # int(0.4*len(csvFileList)) #1000
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle_sample:
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, num_threads=8, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.train.batch(
            [example, label], batch_size=batch_size, num_threads=8)
    return example_batch, label_batch


######

# def batch_norm_layer(inputs, phase_train, scope=None):
# 	return tf.cond(phase_train,
# 		lambda: layers.batch_norm(inputs, is_training=True, scale=True,
# 			updates_collections=None, scope=scope),
# 		lambda: layers.batch_norm(inputs, is_training=False, scale=True,
# 			updates_collections=None, scope=scope, reuse = True))

def batch_norm_layer(inputs, phase_train, scope=None):
    if phase_train:
        return layers.batch_norm(inputs, is_training=True, scale=True,
                                 updates_collections=None, scope=scope)
    else:
        return layers.batch_norm(inputs, is_training=False, scale=True,
                                 updates_collections=None, scope=scope, reuse=True)


def deepSense(inputs, train, reuse=False, name='deepSense'):
    with tf.variable_scope(name, reuse=reuse) as scope:
        used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))  # (BATCH_SIZE, WIDE)
        length = tf.reduce_sum(used, reduction_indices=1)  # (BATCH_SIZE)
        length = tf.cast(length, tf.int64)

        mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
        mask = tf.tile(mask, [1, 1, INTER_DIM])  # (BATCH_SIZE, WIDE, INTER_DIM)
        avgNum = tf.reduce_sum(mask, reduction_indices=1)  # (BATCH_SIZE, INTER_DIM)

        # inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
        sensor_inputs = tf.expand_dims(inputs, axis=3)
        # sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
        acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

        acc_conv1 = layers.convolution2d(acc_inputs, CONV_NUM, kernel_size=[1, 2 * 3 * CONV_LEN],
                                         stride=[1, 2 * 3], padding='VALID', activation_fn=None, data_format='NHWC',
                                         scope='acc_conv1')
        acc_conv1 = batch_norm_layer(acc_conv1, train, scope='acc_BN1')
        acc_conv1 = tf.nn.relu(acc_conv1)
        acc_conv1_shape = acc_conv1.get_shape().as_list()
        acc_conv1 = layers.dropout(acc_conv1, CONV_KEEP_PROB, is_training=train,
                                   noise_shape=[acc_conv1_shape[0], 1, 1, acc_conv1_shape[3]], scope='acc_dropout1')

        acc_conv2 = layers.convolution2d(acc_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
                                         stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC',
                                         scope='acc_conv2')
        acc_conv2 = batch_norm_layer(acc_conv2, train, scope='acc_BN2')
        acc_conv2 = tf.nn.relu(acc_conv2)
        acc_conv2_shape = acc_conv2.get_shape().as_list()
        acc_conv2 = layers.dropout(acc_conv2, CONV_KEEP_PROB, is_training=train,
                                   noise_shape=[acc_conv2_shape[0], 1, 1, acc_conv2_shape[3]], scope='acc_dropout2')

        acc_conv3 = layers.convolution2d(acc_conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST],
                                         stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC',
                                         scope='acc_conv3')
        acc_conv3 = batch_norm_layer(acc_conv3, train, scope='acc_BN3')
        acc_conv3 = tf.nn.relu(acc_conv3)
        acc_conv3_shape = acc_conv3.get_shape().as_list()
        acc_conv_out = tf.reshape(acc_conv3,
                                  [acc_conv3_shape[0], acc_conv3_shape[1], 1, acc_conv3_shape[2], acc_conv3_shape[3]])

        gyro_conv1 = layers.convolution2d(gyro_inputs, CONV_NUM, kernel_size=[1, 2 * 3 * CONV_LEN],
                                          stride=[1, 2 * 3], padding='VALID', activation_fn=None, data_format='NHWC',
                                          scope='gyro_conv1')
        gyro_conv1 = batch_norm_layer(gyro_conv1, train, scope='gyro_BN1')
        gyro_conv1 = tf.nn.relu(gyro_conv1)
        gyro_conv1_shape = gyro_conv1.get_shape().as_list()
        gyro_conv1 = layers.dropout(gyro_conv1, CONV_KEEP_PROB, is_training=train,
                                    noise_shape=[gyro_conv1_shape[0], 1, 1, gyro_conv1_shape[3]], scope='gyro_dropout1')

        gyro_conv2 = layers.convolution2d(gyro_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
                                          stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC',
                                          scope='gyro_conv2')
        gyro_conv2 = batch_norm_layer(gyro_conv2, train, scope='gyro_BN2')
        gyro_conv2 = tf.nn.relu(gyro_conv2)
        gyro_conv2_shape = gyro_conv2.get_shape().as_list()
        gyro_conv2 = layers.dropout(gyro_conv2, CONV_KEEP_PROB, is_training=train,
                                    noise_shape=[gyro_conv2_shape[0], 1, 1, gyro_conv2_shape[3]], scope='gyro_dropout2')

        gyro_conv3 = layers.convolution2d(gyro_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
                                          stride=[1, 1], padding='VALID', data_format='NHWC', scope='gyro_conv3')
        gyro_conv3 = batch_norm_layer(gyro_conv3, train, scope='gyro_BN3')
        gyro_conv3 = tf.nn.relu(gyro_conv3)
        gyro_conv3_shape = gyro_conv3.get_shape().as_list()
        gyro_conv_out = tf.reshape(gyro_conv3, [gyro_conv3_shape[0], gyro_conv3_shape[1], 1, gyro_conv3_shape[2],
                                                gyro_conv3_shape[3]])

        sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
        senor_conv_shape = sensor_conv_in.get_shape().as_list()
        sensor_conv_in = layers.dropout(sensor_conv_in, CONV_KEEP_PROB, is_training=train,
                                        noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]],
                                        scope='sensor_dropout_in')

        sensor_conv1 = layers.convolution2d(sensor_conv_in, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN],
                                            stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC',
                                            scope='sensor_conv1')
        sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
        sensor_conv1 = tf.nn.relu(sensor_conv1)
        sensor_conv1_shape = sensor_conv1.get_shape().as_list()
        sensor_conv1 = layers.dropout(sensor_conv1, CONV_KEEP_PROB, is_training=train,
                                      noise_shape=[sensor_conv1_shape[0], 1, 1, 1, sensor_conv1_shape[4]],
                                      scope='sensor_dropout1')

        sensor_conv2 = layers.convolution2d(sensor_conv1, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN2],
                                            stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC',
                                            scope='sensor_conv2')
        sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
        sensor_conv2 = tf.nn.relu(sensor_conv2)
        sensor_conv2_shape = sensor_conv2.get_shape().as_list()
        sensor_conv2 = layers.dropout(sensor_conv2, CONV_KEEP_PROB, is_training=train,
                                      noise_shape=[sensor_conv2_shape[0], 1, 1, 1, sensor_conv2_shape[4]],
                                      scope='sensor_dropout2')

        sensor_conv3 = layers.convolution2d(sensor_conv2, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN3],
                                            stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC',
                                            scope='sensor_conv3')
        sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
        sensor_conv3 = tf.nn.relu(sensor_conv3)
        sensor_conv3_shape = sensor_conv3.get_shape().as_list()
        sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1],
                                                    sensor_conv3_shape[2] * sensor_conv3_shape[3] * sensor_conv3_shape[
                                                        4]])

        gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
        if train:
            gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

        gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
        if train:
            gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

        cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
        init_state = cell.zero_state(BATCH_SIZE, tf.float32)

        cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length,
                                                          initial_state=init_state, time_major=False)

        sum_cell_out = tf.reduce_sum(cell_output * mask, axis=1, keep_dims=False)
        avg_cell_out = sum_cell_out / avgNum

        logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

        return logits


def input_memory(file_list):
    features = []
    labels = []
    for file_path in file_list:
        csvfile = open(file_path, newline='')
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)

        for index, row in enumerate(reader):
            i_features = row[:WIDE * FEATURE_DIM]
            i_features = np.reshape(i_features, (WIDE, FEATURE_DIM))
            i_labels = row[WIDE * FEATURE_DIM:]
            features.append(i_features)
            labels.append(i_labels)

    features.extend(features)
    labels.extend(labels)
    return features, labels

# load train and test index

train_file_list = []
test_file_list = []

for i in range(DATA_NUM):
    train_file_list.append("hhar_data/data_" + str(i) + ".csv")

result_filename = "result/result.csv"
results_csv = open(result_filename, newline='')
result_reader = csv.reader(results_csv)

next(result_reader, None)
for idx, row in enumerate(result_reader):
    test_file_list.append(row[0])
    train_file_list.remove(row[0])

results_csv.close()

global_step = tf.Variable(0, trainable=False)

batch_feature, batch_label = input_pipeline(train_file_list, BATCH_SIZE)

eval_feature, eval_label = input_memory(test_file_list)

batch_eval_feature, batch_eval_label = input_pipeline(test_file_list, BATCH_SIZE, shuffle_sample=False)

X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 20, 120])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, OUT_DIM])

logits = deepSense(X, train=True, name='deepSense')
predict = tf.argmax(logits, axis=1)

batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(batchLoss)
correct_prediction = tf.equal(predict, tf.argmax(Y, 1))
train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

eval_logits = deepSense(X, train=False, reuse=True, name='deepSense')
eval_predict = tf.argmax(eval_logits, axis=1)
eval_accuracy = tf.reduce_mean(tf.cast(tf.equal(eval_predict, tf.argmax(Y, axis=1)), dtype=tf.float32))

t_vars = tf.trainable_variables()

regularizers = 0.
for var in t_vars:
    regularizers += tf.nn.l2_loss(var)
loss += 5e-4 * regularizers

discOptimizer = tf.train.AdamOptimizer(
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9
).minimize(loss, var_list=t_vars)


saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("RESTORE VARIABLE")
    saver.restore(sess, "tmp/model.ckpt")

    TEST_epoch = int(len(test_file_list) / BATCH_SIZE) + 1

    process_file = open("process.csv", "a")
    process_writer = csv.writer(process_file)
    for iteration in range(TOTAL_ITER_NUM):
        print(iteration)

        batch_x, batch_y = sess.run([batch_feature, batch_label])
        train_acc, opt = sess.run([train_accuracy, discOptimizer], feed_dict={X: batch_x, Y: batch_y})

    process_file.close()

    # _predict, test_acc = sess.run([eval_predict, eval_accuracy], feed_dict={X: eval_feature, Y: eval_label})
    acc = []
    rows = []
    label_arr = []
    predict_arr = []
    for eval_idx in range(TEST_epoch):
        # batch_x, batch_y = sess.run([batch_eval_feature, batch_eval_label])
        non_batch_x = np.array(eval_feature[eval_idx * BATCH_SIZE : (eval_idx + 1) * BATCH_SIZE])
        non_batch_y = np.array(eval_label[eval_idx * BATCH_SIZE : (eval_idx + 1) * BATCH_SIZE])
        _predict, test_acc = sess.run([eval_predict, eval_accuracy], feed_dict={X: non_batch_x, Y: non_batch_y})
        label_arr.extend(np.argmax(non_batch_y, axis=1))
        predict_arr.extend(_predict)
        acc.append(test_acc)
    test_acc = np.mean(acc)

    for idx, row in enumerate(test_file_list):
        rows.append([row, label_arr[idx], predict_arr[idx]])

    print("FINAL TEST ACCURACY : ", test_acc)

    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    result_file = open(result_filename, 'w', encoding='utf-8', newline='')
    result_writer = csv.writer(result_file)
    result_writer.writerow([test_acc])
    result_writer.writerows(rows)

    # save_path = saver.save(sess, "tmp/model.ckpt")

    coord.request_stop()
    coord.join(threads)
    sess.close()