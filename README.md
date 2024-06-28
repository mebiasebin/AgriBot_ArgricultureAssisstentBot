# AgriBot_ArgricultureAssisstentBot

GPT - Generative Pre-Trained Transformation

Prompt -> Trained on large text data -> O/p Text

6 layer model

Self attention mechanism - find the imp word
E.g , I LOVE DHOSA
        1    5         10
DHOSA -> imp word
It generates an o/p -> That’s great to hear! Dosa is a South Indian dish.

Self Attention = softmax((QKTdk)V)
Q - Query or Question (Dosa)
K = Key - feature (South Indian Dish)
V = Vector - in depth features
dk - dimensionality constant
softmax =  [1 2 3 4]^T = 
	e1e1+e2+e3+e4 e2e1+e2+e3+e4 e3e1+e2+e3+e4 e4e1+e2+e3+e4T
			Probability Distribution

Architecture:

I/p Embedding -> Positional Encoding -> Self Attention -> Feed Forward -> Normalization -> O/p
Input Embedding:
Words converted to numbers or vectors
Tokenizing
E.g , The quick brown fox jump over the lazy dog.
Each word assign with e^i values in order.There are 9 words, hence it is e^1, e^2, e^3, e^4, e^5, e^6, e^7, e^8, e^9
Positional Encoding:
PE=sin(pos100002idmodel) for even
PE=cos(pos100002idmodel) for odd
dmodel - size of matrix (vector) 
E.g , PE1 = [ sin(11000004),  sin(11000014),  sin(11000024),  sin(11000034) ] 
Self Attention:
Feature extraction, i.e  K and V
Self attention  = softmax((QKTdk)V)
Q=XWQ,K=KWK, V=VWiV
E.g , Self attention for ‘quick’ :
softmax(QquickKquickTdk)Vquick
Feed Forward:
In depth feature extraction
FFN(x)= h=max(0,xw1+b1)w2+b2
w1-> for first layer
w2 -> for second layer
b1,b2 -> biases
RELU activation function is used (non-linear activation fn)
h=(0.1,0.2,0.9 ) represents the hidden space
Normalization:
To convert to a range of values
Norm(x)=(x-)+
x=n ,    =mean    ,  =s.d ,  =for scale ,  =shift in mean & variance of features    
Output:
Final result
Multihead(Q,K,V)=Concat(head1,head2,...)
head=Attention(XWiQ,KWiK,VWiV)
Projections imp for fine tuning - projects vector to higher dimensional space
Q proj, K proj,  proj, geek proj, up proj, down proj

Standard Text Generation

Eg: “A quick brown fox”
Tokenization:
process of splitting the text into smaller units called tokens.

[‘A’,’quick’,’brown’,’fox’]

Token id
 	Each token is assigned a unique identifier, called a token ID. This is done using a    
vocabulary, which is a mapping of tokens to IDs.

{‘A’:0,’quick’:1..}

Embedding vector
Each token ID is mapped to an embedding vector, which is a fixed-size continuous vector representation of the token. These vectors are learned during the training process.

[[0.1,0.3,0.2,0.4],[. . . .]]

Normalisation
The embedding vectors are processed through neural network layers, often involving transformations like normalisation. This can involve operations like layer normalisation or batch normalisation, but in the context of logits, we usually mean the output of the network layers before applying softmax.
Logits->[3.1,0.5,1.5   . . .]
Softmax is a function that converts the logits into a probability distribution. 
provides probability distribution


Token selection
The final step is to select the next token based on the probabilities from the softmax function. 
