class Node(object):
    def __init__(self, label, split_var, data_points, outcomes, parent, left, right, depth, value, X_idx):
        """Initialize all elements of a node

        label: the label of a node
        split_var: value of splitting a node
        data_points: all data points of the node
        parent: the parent of the node
        left: the left child of the node
        right: the right child of the node
        depth: depth of the node
        """
        self.label = label
        self.split_var = split_var
        self.data_points = data_points
        self.outcomes = outcomes
        self.parent = parent
        self.left = left
        self.right = right
        self.depth = depth
        self.value = value
        self.X_idx = X_idx