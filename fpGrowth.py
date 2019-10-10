class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.NodeLink = None
        self.parent = parentNode
        self.children = {}

    def increase(self, numOccur):
        self.count += numOccur


    def display(self, index=1):
        print(' '*index, self.name, '  ', self.count)
        for child in self.children.values():
            child.display(index + 1)

rootNode = treeNode('pyramid', 9, None)
rootNode.children['eye'] = treeNode('eye', 13, None)
rootNode.display()
rootNode.children['ophoenix'] = treeNode('phoenix', 3, None)
rootNode.display()