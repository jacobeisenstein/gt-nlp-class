from nose.tools import eq_, assert_almost_equals
from gtnlplib import logreg

def setup_module():
    pass

def test_lazy_regularization():
    x_fake = [{'foo':1,'bar':1},{'bar':1,'baz':1}]
    y_fake = ['ugly','pretty']
    theta_lr_lazy,_ = logreg.estimate_logreg(x_fake,y_fake,10,lazy_reg=True,learning_rate=.02)
    theta_lr_not_lazy,_ = logreg.estimate_logreg(x_fake,y_fake,10,lazy_reg=False,learning_rate=.02)
    eq_(len(theta_lr_lazy),len(theta_lr_not_lazy))
    for key,weight in theta_lr_lazy.iteritems():
        assert_almost_equals(weight,theta_lr_not_lazy[key])
