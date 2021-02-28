
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (12.0, 8.0)

from sklearn.metrics import mean_squared_error
import numpy as np
import random
from matplotlib import pyplot as plt
from numpy import arange
import bisect


# ### Generate Bound Walk Examples
# 
# In this section, we will generate our random walk examples.
# 
# In order to obtain statistically reliable, results, 100 training sets, each consisting of 10 sequences, were constructed for use by all learning procedures.

# In[2]:

def generateBoundedWalk():
    ret = ""
    d = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'}
    state = 3
    ret += d[state]
    count = 1
    while((d[state] is not 'A') and(d[state] is not 'G')):
        flip = random.randint(0, 1)
        if(flip == 0):
            state += -1
            ret += d[state]
            count = count + 1
        else:
            state += 1
            ret += d[state]
            count = count + 1
    if((ret[-1] == 'A') or (ret[-1] == 'G')):
        return ret
    
    return ""
          
    
    


# In[3]:

def generateBoundedWalks(size_of_set = 10):
    ret = []
    while(len(ret) < size_of_set):
        walk = generateBoundedWalk()
        ret.append(walk)
     
    return(ret)
        


# In[4]:

def generateSetsofBoundedWalks(number_of_sets = 100, size_of_set = 10):
    ret = []
    for i in range(number_of_sets):
        ret.append(generateBoundedWalks(size_of_set))
    
    return ret
        


# In[5]:

walks = generateBoundedWalks()
print(len(walks))
for item in walks:
    print(item)


# ### Implement TD Lambda algorithm
# 
# In this section, we will implement TD lambda.

# In[6]:

def convertAlphaStateToVector(state):
    #d = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'}
    switcher = {
        'A': np.array([0.]),
        'B': np.array([1.,0.,0.,0.,0.]),
        'C': np.array([0.,1.,0.,0.,0.]),
        'D': np.array([0.,0.,1.,0.,0.]),
        'E': np.array([0.,0.,0.,1.,0.]),
        'F': np.array([0.,0.,0.,0.,1.]),
        'G': np.array([1.]),
    }
    # Get the function from switcher dictionary
    func = switcher[state]
    # Execute the function
    return func

def convertAlphaSeqToVectorSeq(walk):
    ret = []
    for i in range(len(walk)):
        ret.append(convertAlphaStateToVector(walk[i]))
        
    #print(ret)
    return ret


# In[7]:

def prediction(w, x):
    if((x == np.array([1.])).all()):
        #print('G')
        return 1
    if((x == np.array([0.])).all()):
        #print('A')
        return 0
    return w.T.dot(x)

def gradient(w, P, x):
    return x


def cal_delta_w(X, t, w_t, alpha, vlambda):    
        P_t = prediction(w_t, X[t-1]) #t -1 is t, array nonsense
        P_t_1 = prediction(w_t, X[t])   # t is t+1, in the array
        pred = (P_t_1 - P_t)

 
        sum_w = np.array([0.,0.,0.,0.,0.])
        for k in range(1, t+1):
            cal_lambda = pow(vlambda, t-k)
            g = gradient(w_t, P_t, X[k-1]) #k -1 is k, array nonsense
            sum_w += cal_lambda * g
            
        return(alpha*pred*sum_w)
    
def td_lambda(w, X, alpha, vlambda): 
    delta_w = np.array([0.,0.,0.,0,0.])
    for t in range(1,len(X)):
        tmp = cal_delta_w(X, t, w, alpha, vlambda)
        delta_w += tmp
    ret = w + delta_w
    return ret

def td_lambda_ex2(w, X, alpha, vlambda): 
    delta_w = np.array([0.,0.,0.,0,0.])
    for t in range(1,len(X)):
        tmp = cal_delta_w(X, t, w, alpha, vlambda)
        delta_w = tmp
        w = w + delta_w
    ret = w 
    return ret


# ## Experiment 1
# 
# ### Figure 3
# Average error on the random-walk problem under repeated presentations.
# All data are from TD(A) with different values of A. The dependent measure
# used is the RMS error between the ideal predictions and those found by the
# learning procedure after being repeatedly presented with the training set
# until convergence of the weight vector. This measure was averaged over
# 100 training sets to produce the data shown. The A = 1 data point is
# the performance level attained by the Widrow-Hoff procedure. For each
# data point, the standard error is approximately ~ = 0.01, so the differences
# between the Widrow-Hoff procedure and the other procedures are highly
# significant.
# 
# 

# In[8]:

walks = generateSetsofBoundedWalks()


# In[64]:

def exp_1():
    actual = [1./6., 1./3., 1./2., 2./3., 5./6.]
    w = np.array([0.5,0.5,0.5,0.5,0.5])
    lambdas = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    alpha = 0.10
    #training_sets = generateSetsofBoundedWalks()
    training_sets = walks  
    ret = []
    for lambda_e in lambdas:
        cal = []
        for train in  training_sets: 
            w_e = w
            for seqence in train:
                #print(seqence)
                X = convertAlphaSeqToVectorSeq(seqence)
                #print(X[-1])
                w_e = td_lambda(w_e, X, alpha,lambda_e)

            RMSE = mean_squared_error(actual, w_e)**0.5
            cal.append(RMSE)
        
        cal = np.array(cal)
        print("Lambda: {0}, avg: {1}, std:{2}".format(lambda_e, np.average(cal), np.std(cal)))
        ret.append((lambda_e, np.average(cal)))
    return ret

v = exp_1()

print(v)


# In[65]:

def figure3_plot_1(x,y, ext_x = 0.1, ext_y = 0.02):
    fig = plt.figure()
    fig.suptitle('Figure 3', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title('Average error on the random-walk problem under repeated presentations.')

    ax.set_xlabel('Lambda')
    ax.set_ylabel('Error using Best Alpha')

    
    ax.plot(x,y,marker='o')
    plt.xlim(min(x)-ext_x,max(x)+ext_x)
    plt.ylim(min(y)-ext_y,max(y)+ext_y)
    plt.show()
    
    
print(v)    
x,y = zip(*v)
print(x)
figure3_plot_1(x, y)


# ## Experiment 2
# 
# The second experiment concerns the question of learning rate when the
# training set is presented just once rather than repeatedly until convergence.
# Although it is difficult to prove a theorem concerning learning rate, it is easy to
# perform the relevant computational experiment. We presented the same data
# to the learning procedures, again for several values of lambda, with the following
# procedural changes. First, each training set was presented once to each procedure.
# Second, weight updates were performed after each sequence, as in (1),
# rather than after each complete training set. Third, each learning procedure
# was applied with a range of values for the learning-rate parameter a. Fourth,
# so that there was no bias either toward right-side or left-side terminations, all
# components of the weight vector were initially set to 0.5.
# 
# 
# ### Figure 4
# Average error on random walk problem after experiencing 10 sequences.
# All data are from TD(lambda) with different values of alpha and Lambda. The dependent
# measure is the RMS error between the ideal predictions and those found
# by the learning procedure after a single presentation of a training set.
# This measure was averaged over 10O training sets. The lamba = 1 data points
# represent performances of the Widrow-Hoff supervised-learning procedure.
# 
# ### Figure 5
# Average error at best Alpha value on random-walk problem. Each data point
# represents the average over 100 training sets of the error in the estimates
# found by TD(Lambda)~), for particular A and a values, after a single presentation
# of a training set. The Lambda value is given by the horizontal coordinate. The alpha
# value was selected from those shown in Figure 4 to yield the lowest error
# for that lambda, value.
# 

# In[ ]:

walks = walks  


# ### Figure 4

# In[66]:

def exp_2():
    actual = [1./6., 1./3., 1./2., 2./3., 5./6.]
    w = np.array([0.5,0.5,0.5,0.5,0.5])
    #lambdas = [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    lambdas = [0.8]
    alpha = 0.30
    training_sets = walks  
    ret = []
    
    for lambda_e in lambdas:
        cal = []     
        for train in  training_sets:             
            w_e = w
            for sequence in train:
                X = convertAlphaSeqToVectorSeq(sequence)
                #print(sequence)
                w_e = td_lambda_ex2(w_e, X, alpha, lambda_e)
            #print(w_e)
            RMSE = mean_squared_error(actual, w_e)**0.5
            cal.append(RMSE)     
        cal = np.array(cal)
        print("Lambda: {0}, avg: {1}, std:{2}".format(lambda_e, np.average(cal), np.std(cal)))
        ret.append((lambda_e, np.average(cal)))
    return ret

g = exp_2()

print(g)


# In[67]:

def exp_2_4():
    actual = [1./6., 1./3., 1./2., 2./3., 5./6.]
    w = np.array([0.5,0.5,0.5,0.5,0.5])
    lambdas = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #alpha = 0.10
    alphas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    #training_sets = generateSetsofBoundedWalks()
    training_sets = walks  
    ret = []
    for lambda_e in lambdas:
        alp_arr = []
        for alpha in alphas:
            cal = []
            for train in  training_sets: 
                w_e = w
                for seqence in train:
                    #print(seqence)
                    X = convertAlphaSeqToVectorSeq(seqence)
                    #print(X[-1])               
                    w_e = td_lambda_ex2(w_e, X, alpha,lambda_e)

                RMSE = mean_squared_error(actual, w_e)**0.5
                cal.append(RMSE)

            cal = np.array(cal)
            print("Lambda: {0}, alpha:{1}, avg: {2}, std:{3}".format(lambda_e, alpha, np.average(cal), np.std(cal)))
            avg = np.average(cal)
            alp_arr.append((alpha, avg  if avg < 1.0 else np.nan))
        
        ret.append((lambda_e, alp_arr))
        
        #ret.append((lambda_e, np.average(cal)))
            
    return ret

v = exp_2_4()

#print(v)


# In[68]:

def figure4_plot_1(lambda_arr, tup, ext_x = 0.1, ext_y = 0.02):
    fig = plt.figure()
    fig.suptitle('Figure 4', fontsize=14, fontweight='bold')
    
    
    for i in range(0,len(lambda_arr)):
        ax = fig.add_subplot(111)
        
        if(lambda_arr[i] in [1, 0.0, 0.8, 0.3]):
            #print(lambda_arr[i])
            x,y = zip(*tup[i])
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Error')
            ax.plot(x,y,marker='o', label ='Lambda:{0}'.format(lambda_arr[i]))
            
    
    ax.set_title('Average error on random walk problem after experiencing 10 sequences..')
    plt.xlim(-0.1,0.7)
    plt.ylim(0.05,0.7)
    plt.legend(loc='upper left')
    plt.show()
    
    
#print(v)    
lambda_arr,tup = zip(*v)
#print(lambda_arr)
#print(tup)
x,y = zip(*tup[0])
#print(x)
#print(y)
figure4_plot_1(lambda_arr,tup)


# ### Figure 5

# In[94]:

def getBestAlpha(lambda_arr, tup):
    ret = []
    for i in range(0, len(lambda_arr)):
        x,y = zip(*tup[i])
        y = np.array(y)
        y= y[numpy.logical_not(numpy.isnan(y))]
        index = np.argmin(y)
        ret.append((lambda_arr[i], x[index]))  
    return ret


lambda_arr,tup = zip(*v)
alphas = getBestAlpha(lambda_arr, tup)

print(alphas)


# In[97]:

def exp_2_5(alphas):
    actual = [1./6., 1./3., 1./2., 2./3., 5./6.]
    w = np.array([0.5,0.5,0.5,0.5,0.5])
   
    #training_sets = generateSetsofBoundedWalks()
    training_sets = walks  
    ret = []
    for lambda_e, alpha in alphas:
        cal = []
        for train in  training_sets: 
            w_e = w
            for seqence in train:
                #print(seqence)
                X = convertAlphaSeqToVectorSeq(seqence)
                #print(X[-1])
                w_e = td_lambda_ex2(w_e, X, alpha,lambda_e)

            RMSE = mean_squared_error(actual, w_e)**0.5
            cal.append(RMSE)
        
        cal = np.array(cal)
        print("Lambda: {0}, alpha:{1}, avg: {2}, std:{3}".format(lambda_e, alpha, np.average(cal), np.std(cal)))
        ret.append((lambda_e, np.average(cal)))
    return ret

v = exp_2_5(alphas)

print(v)


# In[99]:

def figure5_plot_1(x,y, ext_x = 0.1, ext_y = 0.02):
    fig = plt.figure()
    fig.suptitle('Figure 5', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title('Average error at best alpha value on random-walk problem..')

    ax.set_xlabel('Lambda')
    ax.set_ylabel('Error using Best Alpha')

    
    ax.plot(x,y,marker='o')
    plt.xlim(min(x)-ext_x,max(x)+ext_x)
    plt.ylim(min(y)-ext_y,max(y)+ext_y)
    plt.show()
    
    
print(v)    
x,y = zip(*v)
print(x)
figure5_plot_1(x, y)


# In[ ]:



