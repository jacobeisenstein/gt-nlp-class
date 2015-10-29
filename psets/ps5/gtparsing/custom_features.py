from dependency_features import DependencyFeatures
from numpy import sign

class LexFeats(DependencyFeatures):
    def create_arc_features(self,instance,h,m,add=False):
        """ Notes about the code
        - You start by calling the same function, using the parent class. 
          You can build a chain of feature functions in this way.
        - h provides the index of the head word of the dependency arc
        - m provides the index of the modifier word of the dependency arc
        - You can access the part of speech tags in the instance as instance.pos[i], 
          where i indexes any word token.
        - You can access the words themselves as instance.words[i], 
          where i again indexes the token
        - To create a feature, you call getF(), with two arguments:
          - A feature tuple, which includes an index k, and any other information 
            you want -- it need not be a tuple of exactly three items
          - An argument "add", which you don't need to worry about 
            (but you do need to include)
          - Make sure to keep k up-to-date. 
            This prevents collisions in the space of features.
        """
        ff = super(LexFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.words[m]),add)
        ff.append(f)
        return ff


# For Deliverable 1a
class LexDistFeats(LexFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        return None

# For Deliverable 1b
class LexDistFeats2(LexDistFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        return None

# For Deliverable 1c
class ContextFeats(LexDistFeats2):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        return None

# For Deliverable 2c
class DelexicalizedFeats(DependencyFeatures):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        return None
