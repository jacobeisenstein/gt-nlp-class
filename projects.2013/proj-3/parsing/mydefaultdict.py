from collections import defaultdict

class MyDefaultDict(defaultdict):
    def __init__(self,*args,**kwargs):
        super(MyDefaultDict,self).__init__(*args,**kwargs)
        self.addkey = True
    def __missing__(self,key):
        if self.default_factory is None: raise KeyError((key,))
        value = self.default_factory()
        if self.addkey: self[key] = value
        return value

