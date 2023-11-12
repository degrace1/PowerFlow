'''
Classes file for TET 4205 Group 2 - Fall 2023
2 Classes
VarMat:
    - matrix variable
    - includes a name and a value for easier printing and analyzing
JacElem:
    - Jacobian element
    - includes a name, value, and type
    - type has 8 categories for easier calculation of the elements
'''


class VarMat:
    def __init__(self, name = "NA", val = 0.0):
        self.name = name
        self.val = val

class JacElem:
    def __init__(self, name = "NA", val = 0.0, type = "NA"):
        self.name = name
        self.val = val
        self.type = type
