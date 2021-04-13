class Tree(object):
    def __init__(self, item, index):
        self.left = None
        self.right = None
        self.input = item
        self.index = index ###from 0 to len(sentence - 1)
        self.state = None


def cal_node(root):
    if root.left == None:
        if root.right != None:
            return 10000;

        return 1
    else:
        return 1 + cal_node(root.left) + cal_node(root.right)