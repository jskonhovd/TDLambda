
# coding: utf-8

# In[1]:

import math

def findPossibleMedians(xPresent, yPresent, k):
    #magic = sys.maxint
    x_a = list(xPresent)
    y_a = list(yPresent)
    #x.append(magic)
    #y.append(magic)
    x_a.sort()
    y_a.sort()
    xl_index = int(math.floor((len(x_a) - 1) / 2))
    xh_index = xl_index + 1
    yl_index = int(math.floor((len(y_a) - 1) / 2))
    yh_index = yl_index + 1
    
    print xl_index, xh_index, yl_index, yh_index
    
    xl_bounds = x_a[xl_index]
    xh_bounds = x_a[xh_index]
    
    
    yl_bounds = y_a[yl_index]
    yh_bounds = y_a[yh_index]
    
    #xl_maxlow = x_a[0]
    #xh_maxhigh = x_a[len(x_a) - 1]
    
    #yl_maxlow = y_a[0]
    #yh_maxhigh = y_a[len(y_a) - 1]
    
    ret = []
    print xl_bounds, xh_bounds, yl_bounds, yh_bounds
    #print xl_maxlow, xh_maxhigh, yl_maxlow, yh_maxhigh
    
    for x in range(xl_bounds, xh_bounds):
        for y in range(yl_bounds, yh_bounds):
            if(abs(x-y) == k):
                ret.append((x, y, k))
    return ret         


# In[2]:

def getDifferenceVector(x_a,y_a):
    ret = []
    
    for i in range(len(x_a)):
        value = x_a[i] - y_a[i]
        ret.append(abs(value))
    
    return ret


# In[3]:

def getLInfinityDistance(x_a,y_a):
    v = getDifferenceVector(x_a,y_a)
    
    return max(v)


# In[4]:

def generateXY(x_a, y_a, xMissing, yMissing, x_value, y_value):
    x = list(x_a)
    y = list(y_a)
    
    x.insert(xMissing, x_value)
    y.insert(yMissing, y_value)
    
    return (x,y)
    


# In[16]:

import sys

def getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k):
    
    print xPresent
    print yPresent
    ret = sys.maxint
    medians = findPossibleMedians(xPresent, yPresent, k)
    ret_tuple = (0,0,0)
    for median in medians:
        x = median[0]
        y = median[1]
        x_a, y_a = generateXY(xPresent, yPresent, xMissing, yMissing, x, y)
        value = getLInfinityDistance(x_a,y_a)
        if(value <= ret):
            ret = min(ret, getLInfinityDistance(x_a,y_a))
            ret_tuple = (x,y,ret)
        
    return ret_tuple
    


# In[17]:

n=3
xPresent=[-70, 110]
yPresent=[32, -240]
xMissing=1
yMissing=1
k=115


print getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k)


# In[18]:

n=9
xPresent=[-167, -204, 195, 255, -206, -135, 165, 239]
yPresent=[89, -141, 77, 133, -106, 85, -78, 91]
xMissing=3
yMissing=5
k=44

print getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k)


# In[19]:

n=7
xPresent=[212, -190, -93, 189, -211, 130]
yPresent=[-6, 213, -144, 60, -216, 172]
xMissing=1
yMissing=3
k=108

print getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k)


# In[20]:

n=15
xPresent=[28, 189, -206, -14, -183, 129, 193, -103, 247, -120, -235, -28, -68, 139]
yPresent=[-141, 196, -10, -148, 3, -239, -117, 112, -190, 192, -128, -246, 31, 24]
xMissing=13
yMissing=8
k=24 

print getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k)


# In[21]:

n=9
xPresent=[-120, 66, 84, -102, 235, -226, -193, 111] 
yPresent=[-88, 98, 233, 101, 128, 7, 127, -226]
xMissing=1
yMissing=3
k=170 

print getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k)


# In[22]:

n=3
xPresent=[-224, -142]
yPresent=[218, -252]
xMissing=2
yMissing=0
k=58 

print getValueNonExpansion(n, xPresent, yPresent, xMissing, yMissing, k)


# In[ ]:



