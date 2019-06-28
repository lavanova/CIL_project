import os
import csv
import numpy as np
import tensorflow as tf
#import pandas as pd
data_path = '../data'
def create_dataloader_test(batch_size=256):
    row_col = []
    label = []
    rcstrs = []
    with open(os.path.join(data_path, "sampleSubmission.csv")) as f:
        reader = csv.reader(f, delimiter=',')
        for i, sample in enumerate(reader):
            if i == 0:
                continue
            if sample == None or sample == "":
                continue
            rcstrs.append(sample[0])
            row = int(sample[0].split('_')[0][1:])
            col = int(sample[0].split('_')[1][1:])
            row_col.append([row, col])
            rating = int(sample[1])
            label.append(rating)         
    
    row_col = np.asarray(row_col)
    label = np.asarray(label, dtype=np.float32).reshape((-1, 1)) #reshape??
    assert row_col.shape[0] == label.shape[0], "error sample number doesn't match with label number"

    sample_num = label.shape[0]

    print( "{} test samples, ".format(sample_num) )

    test_ds = tf.data.Dataset.from_tensor_slices((row_col, label))
    #test_ds = test_ds.shuffle(buffer_size=1500)
    #train_ds = train_ds.map()
    #test_ds = test_ds.repeat()
    test_ds = test_ds.batch(batch_size, drop_remainder=False)

    #iterator_test = test_ds.make_one_shot_iterator()
    iterator_test = test_ds.make_initializable_iterator()
    #dataloader_test = iterator_test.get_next()
    
    #template = pd.read_csv( os.path.join(data_path, "sampleSubmission.csv") )
    #return dataloader_test, row_col, rcstrs
    return iterator_test, row_col, rcstrs
def create_dataloader_train(valid_ratio=0.1, batch_size=256):
    row_col = []
    label = []
    rcstrs = []
    max_row = 1
    max_col = 1
    with open(os.path.join(data_path, "data_train.csv")) as f:
        reader = csv.reader(f, delimiter=',')
        for i, sample in enumerate(reader):
            if i == 0:
                continue
            if sample == None or sample == "":
                continue
            rcstrs.append(sample[0])
            row = int(sample[0].split('_')[0][1:])
            max_row = max(max_row, row)
            col = int(sample[0].split('_')[1][1:])
            max_col = max(max_col, col)
            row_col.append([row, col])
            rating = int(sample[1])
            label.append(rating)
    
    row_col = np.asarray(row_col)
    label = np.asarray(label, dtype=np.float32).reshape((-1, 1)) #reshape??
    #row_col_output = np.copy(row_col)
    #label_output = np.copy(label)
    assert row_col.shape[0] == label.shape[0], "error sample number doesn't match with label number"

    sample_num = label.shape[0]
    np.random.seed(1234)
    index = np.random.permutation(sample_num)
    row_col = row_col[index]
    label = label[index]
    rcstrs = list( np.asarray(rcstrs)[index] )
    row_col_output = np.copy(row_col)
    label_output = np.copy(label)

    valid_num = int(sample_num * valid_ratio)
    valid_sample = row_col[0: valid_num, :]
    valid_label = label[0: valid_num, :]
    row_col_output_valid = np.copy(row_col_output)[0: valid_num, :]
    label_output_valid = np.copy(label_output)[0: valid_num, :]
    #valid_sample = np.copy(row_col)
    #valid_label = np.copy(label)

    train_sample = row_col[valid_num:, :]
    train_label = label[valid_num:, :]
    #train_sample = np.copy(row_col)
    #train_label = np.copy(label)

    print( "{} train samples, ".format(sample_num - valid_num) + " {} valid samples.".format(valid_num) )
    #print(np.asarray(rcstrs)[index][0:10])
    output_ds = tf.data.Dataset.from_tensor_slices((row_col_output, label_output))
    output_ds = output_ds.batch(batch_size, drop_remainder=False)
    iterator_output = output_ds.make_initializable_iterator()
    #iterator_output = output_ds.make_one_shot_iterator()
    #dataloader_output = iterator_output.get_next()

    output_valid_ds = tf.data.Dataset.from_tensor_slices((row_col_output_valid, label_output_valid))
    output_valid_ds = output_valid_ds.batch(batch_size, drop_remainder=False)
    iterator_output_valid = output_valid_ds.make_initializable_iterator()
    #iterator_ouput_valid = output_valid_ds.make_one_shot_iterator()
    #dataloader_output_valid = iterator_ouput_valid.get_next()

    train_ds = tf.data.Dataset.from_tensor_slices((train_sample, train_label))
    train_ds = train_ds.shuffle(buffer_size=(sample_num-valid_num))
    #train_ds = train_ds.map()
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size, drop_remainder=False)

    iterator_train = train_ds.make_one_shot_iterator()
    dataloader_train = iterator_train.get_next()

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_sample, valid_label))
    valid_ds = valid_ds.shuffle(buffer_size=valid_num)
    valid_ds = valid_ds.repeat()
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True)

    iterator_valid = valid_ds.make_one_shot_iterator()
    dataloader_valid = iterator_valid.get_next()

    return dataloader_train, dataloader_valid, max_row, max_col, iterator_output, rcstrs, iterator_output_valid, rcstrs[0:valid_num]




    


            

