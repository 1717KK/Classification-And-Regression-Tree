import numpy as np
import copy
import pandas as pd
import math
import collections
from Node import *


class Tree(object):
    def __init__(self, criterion, X, y, labels):
        """Initialize the root for classification tree
        criterion: criterion for impurity calculation
        X: a n*p matrix of observed covariates for each observation
        y: a vector of observed outcomes
        """
        self.root = Node(None, None, X, y, None, None, None, 0, None, [i for i in range(
            X.shape[0])])  # Node(label, split_var, data_points, outcomes, parent, left, right, depth = 0, value)
        self.criterion = criterion
        self.X = X
        self.y = y
        self.labels = labels
        self.predict = None
        self.depth = 0

    def majority_vote(self, X_idx):
        """Use majority vote to get observed outcome y
        """
        count_0 = 0
        count_1 = 0
        for i in range(len(X_idx)):
            if self.y[i] == 0:
                count_0 += 1
            elif self.y[i] == 1:
                count_1 += 1
        if count_0 > count_1:
            return 0
        else:
            return 1

    def split_region(self, idx_list, ft, mean, y):
        """Split a column of features into left and right part
        value < mean -> left part
        value >= mean -> right part
        index: index to record each value's position in original matrix
        ft: an array to store a kind of feature
        y: observed outcomes
        mean: mean of each two adjacent values
        """
        left_ft = []
        right_ft = []
        left_idx = []
        right_idx = []
        for i in range(len(ft)):
            if ft[i] < mean:
                left_ft.append(ft[i])
                left_idx.append(idx_list[i])
            else:
                right_ft.append(ft[i])
                right_idx.append(idx_list[i])
        left_y = y[left_idx]
        right_y = y[right_idx]
        return (left_ft, left_y, left_idx, right_ft, right_y, right_idx)

    def cal_prob(self, y):
        """Calculate the probability of the observed Y in region is 1
        region: splitted region
        idx_lst:index to record each value's position in original matrix
        y: observed outcomes
        """
        total = len(y)
        count = 0
        for i in range(len(y)):
            if y[i] == 1:
                count += 1
        prob = count / total
        return prob

    def cal_impurity(self, y, criterion):
        """Calculate the impurity for each region
        if criterion == 1: use bayes error to calculate
        if criterion == 2: use cross-entropy to calculate
        if criterion == 3: use gini index to calculate
        """
        prob = self.cal_prob(y)
        if criterion == 1:
            min_prob = 0
            max_prob = min(1 / 2, 1 - 1 / 2)
            impurity = min(prob, 1 - prob)
            assert min_prob <= impurity <= max_prob
        elif criterion == 2:
            min_prob = 0
            max_prob = -(1 / 2) * math.log10(1 / 2) - (1 - 1 / 2) * math.log10(1 / 2)
            impurity = -prob * math.log10(prob) - (1 - prob) * math.log10(prob)
            assert min_prob <= impurity <= max_prob
        elif criterion == 3:
            min_prob = 0
            max_prob = 1 / 2 * (1 - 1 / 2)
            impurity = prob * (1 - prob)
            assert min_prob <= impurity <= max_prob
        return impurity

    def split_single_feature(self, ft_idx, ft, y, idx, criterion):
        max_red = -math.inf
        l_idx = None
        r_idx = None
        split_var = None
        for j in range(len(ft) - 1):
            if ft[j] == ft[j + 1]:
                continue
            mean = (ft[j] + ft[j + 1]) / 2
            (left_ft, left_y, left_idx, right_ft, right_y, right_idx) = self.split_region(idx, ft, mean, y)
            impurity_left = self.cal_impurity(left_y, criterion)
            impurity_right = self.cal_impurity(right_y, criterion)
            impurity = self.cal_impurity(y, criterion)
            reduction = impurity - len(left_ft) / len(ft) * impurity_left - len(right_ft) / len(ft) * impurity_right
            if reduction > max_red:
                max_red = reduction
                l_idx = left_idx
                r_idx = right_idx
                split_var = mean
        info = [ft_idx, max_red, l_idx, r_idx, split_var]

        return info

    def split(self, X, y, fixed_ft_idx, criterion):
        """Find the appropriate feature and split the matrix of X into left and right part based on this feature"""
        info_list = []
        # loop each feature
        for i in range(X.shape[1]):

            # if the feature is fixed, skip it
            if i in fixed_ft_idx:
                continue

            ft = X[:, i]
            if np.all(ft == ft[0]):
                continue
            idx = np.argsort(ft)  # use idx to find corresponding y
            sort_ft = ft[idx]

            # find the best split point to split one single feature
            # info = [ft_idx, max_red, l_idx, r_idx, var]
            info = self.split_single_feature(i, sort_ft, y, idx, criterion)
            info_list.append(info)

        idx_lst = None
        max_reduction = -math.inf
        for i in range(len(info_list)):
            if info_list[i][1] > max_reduction:
                idx_lst = i
                max_reduction = info_list[i][1]

        target_info = info_list[idx_lst]
        [idx_ft, max_reduction, left_region_idx, right_region_idx, split_var] = target_info

        left_region = X[left_region_idx, :]
        right_region = X[right_region_idx, :]
        left_y = y[left_region_idx]
        right_y = y[right_region_idx]

        return (idx_ft, split_var, left_region, left_region_idx, right_region, right_region_idx, left_y, right_y)

    def build_tree(self, node, X, y, depth, maxDepth, labels, X_idx, parent, fixed_ft_idx=[], mark="root"):
        """build a classification tree
        :param node: node class
        :param X: data of attributes
        :param y: data of label
        :param depth: the depth of the tree
        :param labels: a list to store label of each feature
        :param X_idx: a list of index to store each value's position in original matrix
        :param parent: parent of current node
        :param fixed_ft_idx:
        :param mark: left child or right child
        :return: Returns a new tree with a new node
        """
        # print("-" * depth, mark, X.shape)
        # label, split_var, data_points, outcomes, parent = None, left = None, right = None, depth = 0, value = None
        node = Node(None, None, X, y, parent, None, None, 0, None, X_idx)

        # base case: data is unambiguous, no need to split further
        if len(set(y)) == 1:
            node.value = y[0]
            return node

        # base case: no more remaining features, cannot split further
        elif len(labels) == 1 or depth > maxDepth:
            node.value = self.majority_vote(X_idx)
            return node

        else:
            (idx_ft, split_var, left, left_idxs, right, right_idxs, left_y, right_y) = self.split(X, y, fixed_ft_idx,
                                                                                             self.criterion)
            label = self.labels[idx_ft]
            node.label = label
            node.split_var = split_var
            remain_labels = copy.deepcopy(labels)
            remain_labels.remove(label)
            update_fixed_ft_idx = copy.deepcopy(fixed_ft_idx)
            update_fixed_ft_idx.append(idx_ft)

            left_child = self.build_tree(None, left, left_y, depth + 1, maxDepth, remain_labels, left_idxs, node,
                                         update_fixed_ft_idx, "left")
            node.left = left_child
            left_child.parent = node

            right_child = self.build_tree(None, right, right_y, depth + 1, maxDepth, remain_labels, right_idxs, node,
                                          update_fixed_ft_idx, "right")
            node.right = right_child
            right_child.parent = node

            return Node(label, split_var, X, y, node.parent, node.left, node.right, depth, None, X_idx)

    def predict_helper(self, sample, node, labels):

        if node == None:
            return
        elif node.value != None:
            return node.value
        else:
            for i in range(len(sample)):
                if labels[i] == node.label and sample[i] < node.split_var:
                    return self.predict_helper(sample, node.left, labels)
                elif labels[i] == node.label and sample[i] >= node.split_var:
                    return self.predict_helper(sample, node.right, labels)

    def my_predict(self, root, X, labels):
        """Given a new X, determine the Y predicted by the tree
        root: trained classfication tree
        Returns a list of predicted output y
        """
        y_pred_list = []
        for i in range(X.shape[0]):
            y_pred = self.predict_helper(X[i], root, labels)
            y_pred_list.append(y_pred)
        return y_pred_list

    def cal_error(self, root, X, y, labels):
        """Given X, predicted y and true y, calculate the error rate"""
        y_pred = self.my_predict(root, X, labels)
        self.predict = y_pred
        count = 0
        total = len(y)
        for i in range(total):
            if y[i] != y_pred[i]:
                count += 1
        return count / total

    def cal_depth(self, tree):
        """Given a tree, calculate the depth of tree"""
        if (tree == None):
            return -1
        depth = max(self.cal_depth(tree.left), self.cal_depth(tree.right) + 1);
        return depth

    def store_node_lists(self, node, node_list):
        """Store all nodes of a tree
        node: root of the current tree
        Returns a list of all nodes
        """
        if node.label != None:
            node_list.append(node)
            self.store_node_lists(node.left, node_list)
            self.store_node_lists(node.right, node_list)
        return node_list

    def cal_R_t(self, node):
        """Given a node, calculate the training error of node"""
        pt = node.data_points.shape[0] / self.X.shape[0]
        idx = node.X_idx
        value = self.majority_vote(idx)
        y_true = self.y[idx]
        counter = collections.Counter(y_true)
        right_count = counter[value]
        rt = (len(y_true) - right_count) / len(y_true)
        return rt * pt

    def cal_R_Tt(self, node, error, mark = "root"):
        """Given a node, calculate the training error of a subtree Tt - a tree with root at node t"""
        if node.label == None: # leaf node
            y_true = self.y[node.X_idx]
            value = self.majority_vote(node.X_idx)
            counter = collections.Counter(y_true)
            wrong_count = len(y_true) - counter[value]
            return wrong_count
        else:
            error = self.cal_R_Tt(node.left, error, "left") + self.cal_R_Tt(node.right, error, "right")
            return error

    def cal_leaf(self, node, count):
        """Given a node, calculate the number of its leaves"""
        if node.value != None:
            count += 1
            return count
        else:
            count = self.cal_leaf(node.left, count)+ self.cal_leaf(node.right, count)
            return count

    def remove_node(self, node, label):
        """Given a tree and label of a target node, remove the target node from the tree"""
        if node.label != None:
            if node.label == label:
                new_value = self.majority_vote(node.X_idx)
                node.label = None
                node.split_var = None
                node.left = None
                node.right = None
                node.value = new_value
                return node
            else:
                node.left = self.remove_node(node.left, label)
                node.right = self.remove_node(node.right, label)
        return node

    def print_tree(self, node, depth=0, mark="root"):
        """Given a tree, print the tree using preorder"""
        if node.label != None:
            print("-" * depth, mark, node.label, node.split_var)
            self.print_tree(node.left, depth + 1, "left")
            self.print_tree(node.right, depth + 1, "right")
        # else:
        #     print(mark, node.value)


    def find_all_pruned_tree(self, tree, alpha):
        """Given a classification tree and find its all pruned tree"""
        alpha_star_list = [0]
        pruned_tree_list = [tree]
        T = tree
        while (self.cal_depth(T) > 0):
            nodes = self.store_node_lists(T, [])
            min_gt = math.inf
            target_node = None
            for node in nodes:
                # error of node t
                R_t = self.cal_R_t(node)
                # error of a subtree Tt
                R_Tt = (self.cal_R_Tt(node, 0))/self.root.data_points.shape[0]
                num_leaf = self.cal_leaf(node, 0)
                gt = (R_t - R_Tt) / (num_leaf - 1)
                if 0 <= gt < min_gt:
                    min_gt = gt
                    target_node = node
            alpha_star = min_gt
            if alpha_star <= alpha and target_node.label != tree.label:
                alpha_star_list.append(alpha_star)
                T = self.remove_node(T, target_node.label)
                pruned_tree_list.append(T)
            else:
                break

        return pruned_tree_list, alpha_star_list

    def prune_tree(self, tree, alpha, X_test, y_test, labels_test):
        """Given a tree and alpha, choose the optimal pruned tree making the test error smallest"""
        pruned_tree_list, alpha_list = self.find_all_pruned_tree(tree, alpha)
        min_error = math.inf
        best_alpha = None
        best_tree = None
        for i in range(len(alpha_list)):
            pruned_tree = pruned_tree_list[i]
            #self.print_tree(pruned_tree)
            error = self.cal_error(pruned_tree, X_test, y_test, labels_test)
            if error < min_error:
                min_error = error
                best_alpha = alpha_list[i]
                best_tree = pruned_tree

        return best_tree, best_alpha, min_error

    def search_tree(self, node, label):
        """Find the target node with target label
        root: root of the tree
        label: a node's label
        Returns the target node
        """
        if node.label != None:
            if label == node.label:
                return node
            else:
                self.search_tree(node.left)
                self.search_tree(node.right)

    def get_data(self, node):
        """for a given node, obtain all the data points that fall into that node of the tree
        root: root of current tree
        Return a list of all the data points that fall into that node
        """
        return node.data_points

    def is_empty(self, root):
        """Check if no nodes are empty
        root: root of the tree
        Returns True if no nodes are empty. Otherwise, return false
        """
        node_list = self.store_node_lists(root, [])
        for node in node_list:
            if node == None:
                return False
        return True

    def is_valid_split(self, node):
        """Check if a node has valid split
        Returns True if all data points in the left child have value less than split point
        and all points in the right child have value larger than split point
        """
        ft_idx = self.labels.index(node.label)
        left_check = True
        if node.left != None:
            if isinstance(node.data_points, np.ndarray):
                left_child = node.left.data_points
                left_check_ft = left_child[:, ft_idx]
                for i in range(len(left_check_ft)):
                    if left_check_ft[i] >= node.split_var:
                        left_check = False
                        break
                    break

        right_check = True
        if node.right != None:
            if isinstance(node.data_points, np.ndarray):
                right_child = node.right.data_points
                right_check_ft = right_child[:, ft_idx]
                for j in range(len(right_check_ft)):
                    if right_check_ft[j] < node.split_var:
                        right_check = False
                        break
                    break
        if left_check and right_check:
            return True
        else:
            return False

    def combine(self, node, generation):
        """Regenerate the dataset based on leaves"""
        if node.label == None:
            return node.data_points
        else:
            left_data = self.combine(node.left, generation)
            generation = np.row_stack((generation, left_data))

            right_data = self.combine(node.right, generation)
            generation = np.row_stack((generation, right_data))

            return generation[1:, :]

    def check_all_data(self, node, generation):
        """Check if the generated dataset matches the real dataset applied to the root node"""
        if isinstance(node.data_points, np.ndarray):
            data = self.combine(node, generation)
            for i in range(self.X.shape[1]):
                ft = self.X[:, i]
                true_ft = set(np.sort(ft).tolist())
                gene_ft = set(np.sort(data[:, i]).tolist())
                if not (true_ft == gene_ft):
                    return False
        return True

    def is_valid(self, node):
        """Check if a classification tree is valid
        Check if no nodes are empty
        Check if a node has valid split
        Check if the generated dataset matches the real dataset applied to the root node
        root: root of classification tree
        dataset: input dataset
        """
        assert self.is_empty(node) == True
        assert self.is_valid_split(node) == True
        assert self.check_all_data(node, np.zeros(self.X.shape[1]).reshape(1, -1)) == True

    def process_split_feature(self):
        """If a split feature is the same value, then ingore this feature,
        that is, this feature is not used to split data set"""
        is_processed = False
        # np_X = np.array(self.X)
        # assert(np_X == self.X)
        boolean_X = np.all(self.X == self.X[0, :], axis=0)
        index = []
        for i in range(len(boolean_X)):
            if boolean_X[i] == True:
                index.append(i)
        new_np_X = np.delete(self.X, index, axis=1)
        self.X = new_np_X
        is_processed = True

        return is_processed

    def process_missing_value(self):
        is_processed = False
        """replace missing value with mean value of this feature"""
        # np_X = np.array(self.X)
        df = pd.DataFrame(self.X)
        new_df = df.fillna(df.mean())
        new_np_X = new_df.values
        # self.X = new_np.X.tolist()
        self.X = new_np_X
        is_processed = True
        return is_processed


class SQLTree(Tree):
    def __init__(self, criterion, X, y, labels, table_name, cur):
        super().__init__(criterion, X, y, labels)
        #SQL Tree will not store the data, just use X_idx to store a series of splits
        self.root = Node(None, None, X, y, None, None, None, 0, None, "")  # Node(label, split_var, data_points, outcomes, parent, left, right, depth = 0, value, X_idx)
        self.predict = None
        self.depth = 0
        self.table_name = table_name
        self.cur = cur

    def split_region(self, idx_list, ft, mean, y):
        pass

    def split_single_feature(self, ft, ft_name, method, where):
        min_red = math.inf
        split_var = None
        where_right = ""
        where_left = ""
        for i in range(len(ft) - 1):
            if ft[i][0] == ft[i + 1][0]:
                continue
            mean = (ft[i][0] + ft[i + 1][0]) / 2

            if len(where) == 0:
                left = str(ft_name) + " < " + str(mean)
                right = str(ft_name) + " >= " + str(mean)
            else:
                left = where + " and " + str(ft_name) + " < " + str(mean)
                right = where + " and " + str(ft_name) + " >= " + str(mean)

            left_query = "select " + str(method) + "(fraction_ones(outcome)) from "+self.table_name+" where " + left + ";"
            self.cur.execute(left_query)
            impurity_left = self.cur.fetchall()[0][0]

            right_query = "select " + str(method) + "(fraction_ones(outcome)) from "+self.table_name+" where " + right + ";"
            self.cur.execute(right_query)
            impurity_right = self.cur.fetchall()[0][0]

            reduction = (i + 1) / len(ft) * impurity_left - (len(ft) - (i + 1)) / len(ft) * impurity_right
            if reduction < min_red:
                min_red = reduction
                split_var = mean
                where_left = left
                where_right = right

        info = [ft_name, min_red, split_var, where_left, where_right]

        return info

    def split(self, criterion, where):
            """Find the appropriate feature and split the matrix of X into left and right part based on this feature"""

            if criterion == 1:
                method = "bayer_error"
            elif criterion == 2:
                method = "cross_entropy"
            elif criterion== 3:
                method = "gini_index"

            info_list = []

            # loop each feature
            for ft_name in self.labels:
                if ft_name in where:
                    continue
                else:
                    if len(where) == 0:
                        sql = "select " + str(ft_name) +" from "+self.table_name+" order by " + str(ft_name)
                    else:
                        sql = "select " + str(ft_name) + " from "+self.table_name+" where " + where + " order by " + str(ft_name)
                    self.cur.execute(sql)
                    ft = self.cur.fetchall()
                    #info = [ft_name, min_red, split_var, where_left, where_right]
                    info = self.split_single_feature(ft, ft_name, method, where)
                    # find the best split point to split one single feature
                    info_list.append(info)

            idx_lst = None
            min_reduction = math.inf
            for i in range(len(info_list)):
                if info_list[i][1] < min_reduction:
                    idx_lst = i
                    ft_name = info_list[i][0]
                    min_reduction = info_list[i][1]

            target_info = info_list[idx_lst]
            [ft_name, min_red, split_var, where_left, where_right] = target_info

            return (ft_name, split_var, where_left, where_right)

    def majority_vote(self, where):
        """Use majority vote to get observed outcome y
        """
        if where != "":
            self.cur.execute("select count(outcome) from " + self.table_name + " where " + where + "and outcome = 1;")
            count_0 = self.cur.fetchall()[0][0]
            self.cur.execute("select count(outcome) from " + self.table_name + " where " + where + "and outcome = 0;")
            count_1 = self.cur.fetchall()[0][0]
        else:
            self.cur.execute("select count(outcome) from " + self.table_name + " where outcome = 1;")
            count_0 = self.cur.fetchall()[0][0]
            self.cur.execute("select count(outcome) from " + self.table_name + " where outcome = 0;")
            count_1 = self.cur.fetchall()[0][0]
        if count_0 >= count_1:
            return 0
        else:
            return 1


    def build_tree(self, node, depth, maxDepth, labels, parent, where):
        #(None, depth + 1, remain_labels, node, where_left)
        """build a classification tree
        :param node: node class
        :param depth: the depth of the tree
        :param labels: a list to store label of each feature
        :param parent: parent of current node
        :param where: string a series of splits
        :return: Returns a new tree with a new node
        """
        # label, split_var, data_points, outcomes, parent = None, left = None, right = None, depth = 0, value = None
        node = Node(None, None, None, None, parent, None, None, 0, None, where)

        # base case: data is unambiguous, no need to split further
        if depth == 0:
            self.cur.execute("select outcome from "+self.table_name+";")
        else:
            self.cur.execute("select outcome from "+self.table_name+" where " + where + ";")
        y_tuple = self.cur.fetchall()
        y = []
        for e in y_tuple:
            y.append(e[0])
        if len(set(y)) == 1:
            node.value = y[0]
            return node

        # base case: no more remaining features, cannot split further
        elif len(labels) == 1 or depth > maxDepth:
            node.value = self.majority_vote(where)
            return node

        else:
            (ft_name, split_var, where_left, where_right) = self.split(self.criterion, where)
            #print(ft_name, split_var, "|||", where_left, "|||", where_right)
            label = ft_name
            node.label = label
            node.split_var = split_var
            remain_labels = copy.deepcopy(labels)
            remain_labels.remove(label)

            left_child = self.build_tree(None, depth + 1, maxDepth, remain_labels, node, where_left)
            node.left = left_child
            left_child.parent = node

            right_child = self.build_tree(None, depth + 1, maxDepth, remain_labels, node, where_right)
            node.right = right_child
            right_child.parent = node

            return Node(label, split_var, None, None, node.parent, node.left, node.right, depth, None, where)


    def cal_R_t(self, node):
        """Given a node, calculate the training error of node"""
        if node.X_idx == "":
            pt = 1
            self.cur.execute("SELECT outcome FROM " + self.table_name + " ;")
            y_true_tuple = self.cur.fetchall()
            y_true = [y[0] for y in y_true_tuple]
        else:
            self.cur.execute("SELECT * FROM " + self.table_name+" ;")
            total_data = len(self.cur.fetchall())
            where = node.X_idx
            self.cur.execute("SELECT * FROM " + self.table_name + " where "+ where +" ;")
            current_data = len(self.cur.fetchall())
            pt = current_data / total_data
            self.cur.execute("SELECT outcome FROM " + self.table_name + " where " + where + " ;")
            y_true_tuple = self.cur.fetchall()
            y_true = [y[0] for y in y_true_tuple]

        idx = node.X_idx
        value = self.majority_vote(idx)
        counter = collections.Counter(y_true)
        right_count = counter[value]
        rt = (len(y_true) - right_count) / len(y_true)
        return rt * pt

    def cal_R_Tt(self, node, error, mark = "root"):
        """Given a node, calculate the training error of a subtree Tt - a tree with root at node t"""
        if node.label == None: # leaf node
            where = node.X_idx
            self.cur.execute("SELECT outcome FROM " + self.table_name + " where " + where + " ;")
            y_true_tuple = self.cur.fetchall()
            y_true = [y[0] for y in y_true_tuple]
            value = self.majority_vote(where)
            counter = collections.Counter(y_true)
            wrong_count = len(y_true) - counter[value]
            return wrong_count
        else:
            error = self.cal_R_Tt(node.left, error, "left") + self.cal_R_Tt(node.right, error, "right")
            return error