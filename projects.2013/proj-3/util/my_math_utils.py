import numpy as np
from scipy import *
from scipy.sparse import *

from itertools import izip
import operator

def sort_dic_by_value (dic,reverse=False):
    return sorted(dic.iteritems(), key=operator.itemgetter(1),reverse=reverse)


## Maximum value of a dictionary
def dict_max(dic):
  aux = dict(map(lambda item: (item[1],item[0]),dic.items()))
  if aux.keys() == []:
      return 0
  max_value = max(aux.keys())
  return max_value,aux[max_value]


############
## Dot products that works for sparse matrix as well
## Taken from:
## http://old.nabble.com/Sparse-matrices-and-dot-product-td30315992.html
############
def spdot(A, B): 
  "The same as np.dot(A, B), except it works even if A or B or both might be sparse." 
  if issparse(A) and issparse(B): 
    return A * B 
  elif issparse(A) and not issparse(B): 
    return (A * B).view(type=B.__class__) 
  elif not issparse(A) and issparse(B): 
    return (B.T * A.T).T.view(type=A.__class__) 
  else: 
    return np.dot(A, B) 


##############
### Gets a perpendicualar line in 2D
##############
def perp_2d(a):
   res = 1./a
   res = res[:,] * [-1,1]
   return res



def l2norm(a):
  value = 0
  for i in xrange(a.shape[1]):
    value += np.dot(a[:,i],a[:,i])
  return np.sqrt(value)

def l2norm_squared(a):
  value = 0
  for i in xrange(a.shape[1]):
    value += np.dot(a[:,i],a[:,i])
  return value

#######
## Normalizes an array to sum to one, either column wize, or row wize or the full array.
## Column wize - 0 default
## Rown wize - 1 default
## All - 2 default
########
def normalize_array(a,direction="column"):
  b = a.copy()
  if(direction == "column"):
    sums = np.sum(b,0)
    return np.nan_to_num(b/sums)
  elif(direction == "row"):
    sums =np.sum(b,1)
    return  np.nan_to_num((b.transpose() / sums).transpose())
  elif(direction == "all"):
    sums = np.sum(b)
    return np.nan_to_num(b / sums)
  else:
    print "Error non existing normalization"
    return b
