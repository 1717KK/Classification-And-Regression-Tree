import psycopg2
from Tree import *
from Forest import *
import pytest
import numpy as np
import math
import time
import cProfile, pstats, io
import sys
from sklearn.model_selection import KFold
from Tree import SQLTree


def test_cart(file_name, num_alpha, l_alpha, h_alpha, k, c, d):

    # initialize the instance

    print('\nTesting Classification Tree...')

    # load dataset
    start_load_data = time.time()
    X, y, labels = load_data(file_name)
    end_load_data = time.time()
    t_load_data = end_load_data - start_load_data
    print("time for loading data: ", t_load_data, "seconds")

    np.random.seed(10)
    alpha_range = np.random.uniform(l_alpha, h_alpha, num_alpha)
    criterion = c
    kFold = k
    maxDepth = d

    conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database="yiqizhou", user="yiqizhou", password="19960129zyq")
    cur = conn.cursor()
    print("connected!")
    cur.execute(open("Resources//impurity.sql", "r").read())
    print("opened!")

    dict_tree, dict_SQL_tree = random_pick(X, y, labels, kFold, alpha_range, criterion, maxDepth, cur)

    print("Done!")


def load_data(file_name):
    """load data and return features dataset and outcomes dataset"""
    with open(str(file_name)) as f:
        file = f.readlines()
        label = file[0].strip().split(",")[1:-1]
        file = file[1:]
        row = len(file)
        col = len(file[0].strip().split(","))-1
        mat = np.zeros((row, col))
        for i in range(len(file)):
            row = file[i].strip()
            cols = row.split(",")[1:]
            for j in range(len(cols)):
                mat[i][j] = int(cols[j])
    np.random.seed(10)
    np.random.shuffle(mat)
    X = mat[:, 0:-1]
    y = mat[:, -1]

    return X, y, label

def random_pick(X, y, labels, kFold, alpha_range, criterion, d, cur):
    rows = [10, 100, 300, 500, 700]
    d_tree =dict()
    d_SQL_tree = dict()
    for row in rows:
        sub_X = X[0:row, :]
        sub_y = y[0:row]
        error, alpha, avg_time_build_tree = implement_tree(sub_X, sub_y, labels, kFold, alpha_range, criterion, d)
        d_tree[row] = (error, alpha, avg_time_build_tree)
    for row in rows:
        sub_X = X[0:row, :]
        sub_y = y[0:row]
        sql_error, sql_alpha, sql_time_build_tree = implement_SQLtree(sub_X, sub_y, labels, kFold, alpha_range, criterion, d,
                                                                      cur)
        d_SQL_tree[row] = (sql_error, sql_alpha, sql_time_build_tree)

    return d_tree, d_SQL_tree



def test_reponse_format(y):
    """test if reponses are binary"""
    print("test_response_format()...", end = "")
    for item in y:
        assert (item == 0 or item == 1)
    print("Passed!")


def test_feature_format(X):
    """test if features are numerical"""
    print("test_feature_format()...", end = "")
    for row in range(len(X)):
        for col in range(len(X[0])):
            assert (isinstance(X[row][col], float) == True)
    print("Passed!")


def test_response_value(predict, y):
    """If responses are all the same value, no need to split and all predictions are the same as reponses"""
    print("test_response_value()...", end = "")
    if len(set(y)) == 1:
        assert (predict == y).all()
    print("Passed!")


def test_split_feature(tree):
    """Test if the feature with all same value is removed or not"""
    print("test_split_feature()...", end = "")
    assert (tree.process_split_feature() == True)
    print("Passed!")


def test_missing_value(tree):
    """Test if the missing value is replaced or not"""
    print("test_missing_feature()...", end = "")
    assert (tree.process_missing_value() == True)
    print("Passed!")


def test_all(train):
    test_reponse_format(train.y)
    test_feature_format(train.X)
    test_response_value(train.predict, train.y)
    test_split_feature(train)
    test_missing_value(train)


def implement_tree(X, y, labels, k, alpha_range, criterion, d):
    """generate a classification tree"""

    kf = KFold(n_splits=k)
    min_avg_error = math.inf
    best_alpha = None
    avg_time_build_tree = 0
    for alpha in alpha_range:
        total_error = 0
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train = Tree(criterion, X_train, y_train, labels)
            test = Tree(criterion, X_test, y_test, labels)
            start = time.time()
            train_tree = train.build_tree(train.root, train.X, train.y, 0, d, train.labels, train.root.X_idx, None)
            test_all(train)
            end = time.time()
            train.is_valid(train_tree)
            period = end - start
            avg_time_build_tree += period
            pruned_tree, pruned_alpha, error = train.prune_tree(train_tree, alpha, test.X, test.y, test.labels)
            total_error += error
        avg_error = total_error / k
        if avg_error < min_avg_error:
            min_avg_error = avg_error
            best_alpha = alpha
    avg_time_build_tree = avg_time_build_tree / (k*len(alpha_range))

    return min_avg_error, best_alpha, avg_time_build_tree

def implement_SQLtree(X, y, labels, k, alpha_range, criterion, d, cur):
    """generate a classification tree"""

    kf = KFold(n_splits=k)
    min_avg_error = math.inf
    best_alpha = None
    avg_time_build_tree = 0
    for alpha in alpha_range:
        total_error = 0
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            SQL_train = SQLTree(criterion, X_train, y_train, labels, "articles", cur)
            SQL_test = SQLTree(criterion, X_test, y_test, labels, "articles", cur)
            start = time.time()
            SQL_train_tree = SQL_train.build_tree(SQL_train.root, 0, d, SQL_train.labels, None, "")
            test_all(SQL_train)
            end = time.time()
            SQL_train.is_valid(SQL_train_tree)
            period = end - start
            pruned_SQLtree, pruned_alpha, error = SQL_train.prune_tree(SQL_train_tree, alpha, SQL_test.X, SQL_test.y,
                                                                       SQL_test.labels)
            total_error += error
            avg_time_build_tree += period
        avg_error = total_error / k
        if avg_error < min_avg_error:
            min_avg_error = avg_error
            best_alpha = alpha
        avg_time_build_tree = avg_time_build_tree / (k * len(alpha_range))

    return min_avg_error, best_alpha, avg_time_build_tree


if __name__ == "__main__":
    file_name = sys.argv[1]
    num_alpha = int(sys.argv[2])
    low_alpha = int(sys.argv[3])
    high_alpha = int(sys.argv[4])
    kFold = int(sys.argv[5])
    criterion = int(sys.argv[6])
    maxDepth = int(sys.argv[7])

    pr = cProfile.Profile()
    pr.enable()

    test_cart(file_name, num_alpha, low_alpha, high_alpha, kFold, criterion, maxDepth)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    ps.dump_stats("result_out")