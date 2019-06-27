import numpy as np
import tensorflow as tf
import random as rd
import scipy.sparse as sp
from time import time
import os
import csv

class Data(object):
    def __init__(self, batch_size=1024, valid_ratio=0.1):
        self.path = './data'
        self.batch_size = batch_size

        #train_file = path + '/train.txt'
        #test_file = path + '/test.txt'
        train_file = '../data/data_train.csv'
        test_file = '../data/sampleSubmission.csv'
        #get number of users and items
        
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = list(range(10000))
        self.n_items = 1000
        self.n_users = 10000

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        #self.R_rating_train = np.zeros((10000, 1000))
        #self.R_rating_test = np.zeros
        #self.train_items, self.test_set = {}, {}
        #self.train_items = {}
        #self.test_set_row = []
        #self.test_set_col = []
        #self.test_set_rating = []
        row_col = []
        label = []
        self.rcstrs_output = []
        with open(os.path.join(train_file)) as f:
            reader = csv.reader(f, delimiter=',')
            for i, sample in enumerate(reader):
                if i == 0:
                    continue
                if sample == None or sample == "":
                    continue
                self.rcstrs_output.append(sample[0])
                row = int(sample[0].split('_')[0][1:]) - 1
                #max_row = max(max_row, row)
                col = int(sample[0].split('_')[1][1:]) - 1
                #max_col = max(max_col, col)
                row_col.append([row, col])
                self.R[row, col] = 1
                rating = int(sample[1])
                label.append(rating)
        row_col = np.asarray(row_col)
        label = np.asarray(label, dtype=np.float32).reshape((-1,)) #reshape??
        #row_col_output = np.copy(row_col)
        #label_output = np.copy(label)

        sample_num = label.shape[0]
        np.random.seed(1234)
        index = np.random.permutation(sample_num)
        row_col = row_col[index]
        label = label[index]
        self.rcstrs_output = list( np.asarray(self.rcstrs_output)[index] )
        row_col_output = np.copy(row_col)
        label_output = np.copy(label)
    
        valid_num = int(sample_num * valid_ratio)
        valid_sample = row_col[0: valid_num, :]
        valid_label = label[0: valid_num]
        row_col_output_valid = np.copy(row_col_output)[0: valid_num, :]
        label_output_valid = np.copy(label_output)[0: valid_num]
        self.rcstrs_output_valid = self.rcstrs_output[0: valid_num]
        #valid_sample = np.copy(row_col)
        #valid_label = np.copy(label)

        train_sample = row_col[valid_num:, :]
        train_label = label[valid_num:]
        #train_sample = np.copy(row_col)
        #train_label = np.copy(label)

        print( "{} train samples, ".format(sample_num - valid_num) + " {} valid samples.".format(valid_num) )

        output_ds = tf.data.Dataset.from_tensor_slices((row_col_output, label_output))
        output_ds = output_ds.batch(batch_size, drop_remainder=False)
        self.iterator_output = output_ds.make_initializable_iterator()

        output_valid_ds = tf.data.Dataset.from_tensor_slices((row_col_output_valid, label_output_valid))
        output_valid_ds = output_valid_ds.batch(batch_size, drop_remainder=False)
        self.iterator_output_valid = output_valid_ds.make_initializable_iterator()
        train_ds = tf.data.Dataset.from_tensor_slices((train_sample, train_label))
        train_ds = train_ds.shuffle(buffer_size=(sample_num-valid_num))
        #train_ds = train_ds.map()
        train_ds = train_ds.repeat()
        train_ds = train_ds.batch(batch_size, drop_remainder=False)

        iterator_train = train_ds.make_one_shot_iterator()
        #dataloader_train = iterator_train.get_next()
        self.row_col_train , self.label_train = iterator_train.get_next()

        valid_ds = tf.data.Dataset.from_tensor_slices((valid_sample, valid_label))
        valid_ds = valid_ds.shuffle(buffer_size=valid_num)
        valid_ds = valid_ds.repeat()
        valid_ds = valid_ds.batch(batch_size, drop_remainder=True)

        iterator_valid = valid_ds.make_one_shot_iterator()
        #dataloader_valid = iterator_valid.get_next()
        self.row_col_valid, self.label_valid = iterator_valid.get_next()

        row_col_test = []
        label_test = []
        self.rcstrs_test = []
        with open(os.path.join(test_file)) as f:
            reader = csv.reader(f, delimiter=',')
            for i, sample in enumerate(reader):
                if i == 0:
                    continue
                if sample == None or sample == "":
                    continue
                self.rcstrs_test.append(sample[0])
                row = int(sample[0].split('_')[0][1:]) - 1
                #max_row = max(max_row, row)
                col = int(sample[0].split('_')[1][1:]) - 1
                row_col_test.append([row, col])

                #max_col = max(max_col, col)
                #row_col.append([row, col])
                #R[row, col] = 1
                rating = int(sample[1])
                label_test.append(rating)
        row_col_test = np.asarray(row_col_test)
        label_test = np.asarray(label_test, dtype=np.float32).reshape((-1,))

        sample_num_test = label_test.shape[0]
        print("{} test samples, ".format(sample_num_test))
        test_ds = tf.data.Dataset.from_tensor_slices((row_col, label))
        test_ds = test_ds.batch(batch_size, drop_remainder=False)
        self.iterator_test = test_ds.make_initializable_iterator()
        
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)
    """
    def sample_test(self):
        num_batch = int(self.test_set_rating.shape[0]) // self.batch_size)
        users = np.zeros((num_batch, self.batch_size))
        pos_items = np.zeros((num_batch, self.batch_size))
        pos_items_rating = np.zeros((num_batch, self.batch_size))
        for i in range(num_batch):
            users[i, :] = self.test_set_row[i*self.batch_size : (i+1)*self.batch_size]
            pos_items[i,:] = self.test_set_col[i*self.batch_size : (i+1)*self.batch_size]
            pos_items_rating[i,:] = self.test_set_rating[i*self.batch_size : (i+1)*self.batch_size]
        return users, pos_items, pos_items_rating
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch, self.R_rating_train[u , pos_i_id]

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)
        #def sample_items_for_u(u, num):

        #pos_items, neg_items = [], []
        pos_items, pos_items_rating = [], []
        for u in users:
            select_item, select_rating = sample_pos_items_for_u(u, 1)
            pos_items += select_item
            pos_items_rating += select_rating
            #pos_items += sample_pos_items_for_u(u, 1)
            #neg_items += sample_neg_items_for_u(u, 1)

        #return users, pos_items, neg_items
        return users, pos_items, pos_items_rating

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
    """

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
