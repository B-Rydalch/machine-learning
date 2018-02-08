class Node():
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.children = []

    def isLeaf(self):
        return len(self.children) == 0

    def appendChild(self, name, value):
        self.children.append({
            'name': name,
            'value': value
        })

    def getNextChild(self, node_value):
        if self.isLeaf():
            return self
        else:
            for child in self.children:
                if (child == node_value):
                    return self.name

    def display(self):
        if self.isLeaf():
            print("{}".format(self.name))
        else:
            for child in self.children:
                print("{} -> {}".format(child.name, child.value))
