#!/usr/bin/env python
# coding: utf-8

# In[1]:


words = open('names.txt', 'r').read().splitlines()


# In[2]:


len(words)


# In[3]:


min(len(w) for w in words)


# In[4]:


max(len(w) for w in words)


# In[5]:


#Bigram Language Model


# In[6]:


#To predict the stats of one chars coming after in biagram (uses two character at a time) model is to count for that we can use dictionary.
b = {} #dictionary to maintain the count .
for w in words:
    chs = ['<S>']+list(w)+['<E>'] #array with special start token,with end token which is list of one element, 'w (emma initally) is one element in words dataset(names.txt)'
    for ch1, ch2 in zip(chs, chs[1:]):
        biagram = (ch1, ch2) # biagram stores two characters
        b[biagram] = b.get(biagram, 0)+1 #if biagram doesn't contain any character then by default it will be 0 or biagram +1  to store count


# In[7]:


b.items() #b.items will return the tuples of key are chars biagrams and value are counts.


# In[8]:


sorted(b.items(), key = lambda kv: -kv[1]) #it will take key value (kv) and return kv at 1 which is count.


# In[9]:


#storing info. intp 2D array the rows will be first char of biagram and the column will be the second char.
#And each entry will tell how often second char follow the first char.


# In[10]:


import torch


# In[11]:


a = torch.zeros((3,5), dtype = torch.int32) #array of 3x5
a


# In[12]:


a.dtype 


# In[13]:


a[1,3]+=1 #at row 1 and column 3 make the element 1
a


# In[14]:


N = torch.zeros((27, 27), dtype=torch.int32) #26 chars of alphabat and two special character <S> , <E> show 2D grid wil be 28X28


# In[15]:


#creating lookup table from chars to ints.
#we will take words which is list of string and concatinate into a massive string basically making entire dataset a massive string.
#and pass that massive string to 'set' data structure which will remove all the duplicates in dataset and past it to list,
# to make list out of it.

chars = sorted(list(set(''.join(words))))
#making look up table from string to integer mapping
stoi = {s:i+1 for i,s in enumerate(chars)} #enumerate over character and make a mapping for string to integer.
stoi ['.'] = 0
itos = {i:s for s, i in stoi.items()}


# In[16]:


#To predict the stats of one chars coming after in biagram (uses two character at a time) model is to count for that we can use dictionary.
for w in words:
    chs = ['.']+list(w)+['.'] #array with special start token,with end token which is list of one element, 'w (emma initally) is one element in words dataset(names.txt)'
    for ch1, ch2 in zip(chs, chs[1:]): 
        ix1 = stoi[ch1] # mapping character to value in lookup table stoi
        ix2 = stoi[ch2] # mapping character to value in lookup table stoi
        N[ix1,ix2] +=1 #Our 2D array with [ix1(row), ix2(column)] and indexing(movement of pinter) +=1


# In[17]:


N


# In[ ]:





# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va = "bottom", color="gray")
        plt.text(j, i, N[i,j].item(), ha="center", va = "top", color="gray")
plt.axis('off');


# In[19]:


N[0,:] #gives 1D array of first row, These are raw count now we will covert it into probability.


# In[20]:


p = N[0].float() # converting N into float, The e=reason why we are converting it int float because we are normalizing these count.
p = p/p.sum() # This will create a probability distribution.
p # now this will output the probability.


# In[21]:


g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() #This will draw sample from it . 
itos[ix] # 1: a mapping key is index 1 while value is char a


# In[22]:


#Tourch.Multinomial will return the sample form the multinomial distribution.Which simply means that 
#"You give me probability and i will give you integer which are samples according to probability distribution"
#and to make everything deterministic we will use an object "generator."

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g) #generate three random number.
p = p/p.sum() #normalizing the generator
p # this will give us probability distribution 3 of it.


# In[23]:


torch.multinomial(p, num_samples=20, replacement=True, generator=g) #This will draw sample from it . 
#It will take torch tensor of probability distribution "p = p/p.sum()", number of sample.  
#replacement : It means that when we draw a element we can draw it and we can put back into a list of eligible indics to draw again.
#generator of deterministic result or same result.


# In[24]:


#P = N.float() #convert everything from our 'N' to float . N is our matrix of 27x27 with all the chars which we made above.

P = (N+1).float() 


#Adding 1 to N (27X27 GRID) because we want that atleast very word comes with every atleast one time.This is known as 'Model Smoothing.'
#eg: before P = N.float() 'j comes with q '0' times but now it will atleaset come '1' time.We can add any number if we want.
#It will also insure that there will be no '0' in our probability matrix 'P'.

#we can add any number we want . the more number we will add the more 'Uniform Model we will have'.
#convert everything from our 'N' to float . N is our matrix of 27x27 with all the chars which we made above.

P.sum(1, keepdim=True).shape#0 is sum across the row , 1  is sum across the column,


# In[25]:


P/=P.sum(1, keepdim=True)
#Broadcasting rules (Basically the rule which let us know that is it possible to divide 27X27 dimension array to 27X1 dimension array at binary level)
# - Each tensor has atleast one dimension.
# - When iterating over dimension size starting trom trailing dimension, 1. The dimension size must be equal, one of them is 1, or when 
#of them doesn't exist. 
#27 X 27
#27 X 1 is able to brocaste and have this opertion perform without any error P = P/P.sum(1, keepdim=True)
# because by rule on the dimension size be equal which it is 27 in first and 27 in second, and one of them should be 1 so it is also there.


# In[26]:


g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix  = 0 # begin at index '0'
    while True:
        p = P[ix]
        
        #p = N[ix].float() # we are always fetching a row of 'N' and then converting it to float.
        #p = p/p.sum() # normalize it 
        #p = torch.ones(27)/ 27.0 #This is uniform distribution which will make everything equally likely.
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() #this wil tell us what index will be the next.
        out.append(itos[ix])
        if ix==0:
            break
    print(' '.join(out))


# In[27]:


# log(a*b*c) = log(a) + log(b) + log(c)


# In[28]:


#GOAL : maximize likelihood of the data with respect to 'Model parameter (statistical modeling).'
#equivalent to maximizing the log likelihood (because log is monotonic).
#equivalent to minimizing the negative log likelihood.
#equivalent to minimizing the average negative log likelihood.


# In[29]:


#Summery of quality of model into a single number . How good it is predicting a training set . In training set we can evaluvate 
#training loss it will let us quality of this model in single number.

#code which we previously used for counting above.

#Trainging 

log_likelihood = 0.0
n = 0 #for count

for w in words:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip (chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        #probabilty which model assign to each biagram.
        
        prob = P[ix1,ix2]
        
        #summerise every probability in a single number that measure the quality of model. 'Likeli9hood is product of all probability. '
        #Generally people work with Log Likelihood rather than Likelyhood.
        
        logprob = torch.log(prob)
        log_likelihood +=logprob
        n+=1
        
        #print(f'{ch1}{ch2}: {prob: 4f} {logprob:4f}')
        #The job of our trainingis to find parameter that minimize the negative log likelihood

print(f'{log_likelihood =}')
nll = -log_likelihood #negative log likelihood will give us a very nice loss function.
print(f'{nll=}')
print(f'{nll/n}') 


#normalizin negative log likelihood by count. It will give us 'Average of negative log likelihood'
#how high the log likelihood can get, 'It can go to '0' ' so when all the probability are '1' the log likelihood will be '0'.
#when all probability are lower it will grow more and more negative. we don't like this because we like 'Loss function because'
#loss function has sementic that 'low is good because we are minimizing the loss.' So we actully need to invert this that's what
#give us negative log likelihood


# Alternate Way : Casting the problem of biagram character level language model into 'Neural Network Framework.'

# In[30]:


#STEPS :

#Neural network still going biagram language model but it will receive a single character as an input then then there is 
#neural network with some weight/parameter (W),the the output the probability distribution over the next character in a sequence.
#It is going to make guess what is likely to follow 'Character that was input to the model' and addition to that we will able to
#evaluvate any setting of parameter of neural net because we have 'loss function (negative log likelihood)'.
#We will take a look of probability distribution and use labels which are just identity of the next or second character in that biagram.
#So knowing the second character of biagram allow us to then look at how high the probability the model assign to that character.
#We want probability to be high which is other way to say loss is low .

#We will use 'Gradient base optimization to tune the parameter/ weights of the network so that neural net is correctly predicting'


# In[31]:


#create a training set of all the biagram.
#iterate over all the biagrams
#bigram denotes the (x, y) - first charcter and next to predict.
#training set will be made up of two lists.
xs = [] #Inputs list
ys = [] #targets/labels  lists

for w in words[:1]:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip (chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append (ix1) #these from xs are integers.
        ys.append(ix2) #these value from ys are integers.
        
xs = torch.tensor(xs) #creating tensor from integers of xs
ys = torch.tensor(ys) #creating tensor from integers of ys


# In[32]:


xs #input 0 have label 5 prediction of next word which is 'e' in emma


# In[33]:


ys # for input 0 we have label 5 prediction of next word which is 'e' in emma


# In[34]:


#Now we have to feed in these input and label(xs, ys) into neural network.we can not plug integer index into neural next we we have right now.
#So, The most common way to encode integer to feed to neural net is 'one hot encoding'.
#In one hot encoding we take a integer let's say '13' and create a vector that is all zero except the '13th' dimension which is 
#turned to '1 then we can feed it to neural net'

import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes = 27).float() #num_classes is how long you want your vector to be., coverting into floats because integers can't be feed to neural net but floats can.
xenc


# In[35]:


xenc.shape


# In[36]:


plt.imshow(xenc)


# In[37]:


xenc.dtype


# In[38]:


#Constructing our first neuron

W =torch.randn((27,1)) #intial weight of neuron. '1' represents the presence of single neuron.
#weights are multiplied by inputs so for matrix multiplication.

xenc @ W #xencoding vector matrix with W weights

#output is 5x1 becasue we took the xencoding which is 5x27(5 rows, 27 columns ) and multiplied it by 27X1 which is (27 rows, 1 column)
#27 and 27 will multiply and add and will give us 5 x1 in matrix multiplication.

#The output is 'Activation'


# In[39]:


#Constructing our first neuron

W =torch.randn((27,27)) #intial weight of neuron. '27' represents the presence of single neuron.
#weights are multiplied by inputs so for matrix multiplication.

xenc @ W #xencoding vector matrix with W weights

#output is 5x1 becasue we took the xencoding which is 5x27(5 rows, 27 columns ) and multiplied it by 27X27 which is (27 rows, 1 column)
#27 and 27 will multiply and add and will give us 5 x 27 in matrix multiplication.

#The output is 'Activation.'


# In[40]:


#(5,27) @ (27,27) -> (5,27) the 27 in output tells what is the firing rate of those neurons on every one of those 5 examples.
#for example 
(xenc @ W)[3, 13] # this is giving us a firing rate of "13th neuron looking at 3rd input." this is done by a "dot product"between 
#the 3rd input and 13 column of 'W'matrix.


# In[41]:


#so now we have 27 numbers for every neuron which are right now giving us log counts but we what to get probability out of it for next character to happen. 
#to get count we will take log count and exponetiate it.
logits = (xenc @ W) #log counts.
counts = logits.exp() #this will give us count equvilent to 'N' matrix.

#probability is just count normalized.
prob = counts/counts.sum(1, keepdims = True)
prob
#taking count and expontiate it and then normalize it for probability is also known as 'Softmax Function'
#counts = logits.exp()
#prob = counts/counts.sum(1, keepdims = True)


# In[42]:


nlls = torch.zeros(5)
for i in range (5):
    x = xs[i].item()
    y = ys[i].item()
    print('------------')
    print(f'biagram example {i+1} :{itos[x]}{itos[y]}(indexes {x},{y})')
    print ('Input to nneural net :',x)
    print('output probabilities from the neural net:', prob[i])
    print('label (actual next character):', y)
    p = prob[i,y]
    print('probability assigned by the net to the correct character:', p.item())
    logp = torch.log(p)
    print('Log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:',nll.item())
    nlls[i] = nll
    
    print('===========')
    print('average negative log likelihood, i.e. loss = ', nlls.mean().item())


# In[43]:


prob.shape


# In[ ]:




