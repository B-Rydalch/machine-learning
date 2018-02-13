class Node(object):
    def __init__(self, name = "", children = {}):
        self.name = name
        self.children = children

    def isLeaf(self):
        return not self.children

    def appendChild(self, attribute, value):
        self.children[attribute] = value