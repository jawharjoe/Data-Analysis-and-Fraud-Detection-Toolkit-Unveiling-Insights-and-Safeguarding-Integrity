import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy import linalg as LA2
from sklearn.neighbors import KNeighborsClassifier

print("*****************Q1*********************")
df = pd.read_csv("NormalSample.csv")
sample = df["x"]
samplelist=[]
for x in sample:
    samplelist.append(x)
samplelist.sort()
N = len(samplelist)
mid = int(N/2)+1
midval = samplelist[mid]
medianval = stat.median(samplelist)
minval = min(samplelist)
maxval = max(samplelist)
print("min",minval)
print("max",maxval)
a = int(minval)
if type(maxval) == float:
    b = int(maxval) + 1
print("a",a)
print("b",b)

#calculate izenman value with the previously calculated mid and N values
def izenman(samplelist, mid, N):
    firstmidval1 = samplelist[ int(mid/2) ]
    firstmidval2 = samplelist[ int(mid/2)+1 ]
    lowerqaud = (firstmidval1 + firstmidval2)/2
    print("q1",lowerqaud)
    lastmidval1 = samplelist[ int((mid+N)/2) ]
    higherquad = lastmidval1
    print("q2 ",higherquad)
    print("sample", sample.describe())
    IQR = higherquad - lowerqaud
    print("IQR",IQR)
    izenmannh = ( 2 * IQR )/ math.pow(N,1/3)
    print("izenmannh",izenmannh)
izenman(samplelist, mid, N)

#display histogram with a,b, bin-width and dataset. the number of bins and p(x) values are calculated
def histo(a, b, h, samplelist):
    x = a
    bins = []
    while x < b+h:
        bins.append(round(x,5))
        x+= h
    print(list(bins))
    bin_array = np.array(bins)
    avg_array = (bin_array[1:] + bin_array[:-1]) / 2

    px =[]
    for x in avg_array:
        ycount=0
        for y in samplelist:
            if y>=(x - (h/2)) and y<=(x + (h/2)):
                ycount+=1
        px.append(ycount/(N*h))
    print("bins" , len(bins))
    print("px",len(px))
    print(px)
    plt.hist(sample, len(bins))
    plt.title(h)
    plt.xlabel("m")
    plt.ylabel("p(m)")
    plt.show()
histo(a,b,0.1,samplelist)
histo(a,b,0.5,samplelist)
histo(a,b,1,samplelist)
histo(a,b,2,samplelist)

print("*****************Q2*********************")
X = df["x"]
x_1 = df[ df.group==1 ]
x_0 = df[ df.group==0 ]
X_1 = x_1["x"]
X_0 = x_0["x"]

def findvals(sample):
    samplelist=[]
    for x in sample:
        samplelist.append(x)
    samplelist.sort()

    #calculate five point summaries for the whole list(X), and X by group. Also, finds the outliers
    N = len(samplelist)
    print("N",N)
    mid = int(N/2)+1
    medianval = stat.median(samplelist)
    print("median ",medianval)
    minval = min(samplelist)
    maxval = max(samplelist)
    print("min",minval)
    print("max",maxval)
    a = int(minval)
    if type(maxval) == float:
        b = int(maxval) + 1
    print("a",a)
    print("b",b)
    firstmidval1 = samplelist[ int(mid/2) ]
    firstmidval2 = samplelist[ int(mid/2)+1 ]
    lowerqaud = (firstmidval1 + firstmidval2)/2
    print("q1",lowerqaud)
    lastmidval1 = samplelist[ int((mid+N)/2) ]
    higherquad = lastmidval1
    print("q2 ",higherquad)
    print("sample", sample.describe())
    IQR = higherquad - lowerqaud
    print("IQR",IQR)
    W1 = lowerqaud - 1.5 * IQR
    W2 = higherquad + 1.5 * IQR
    print("W1",W1)
    print("W2",W2)
    outlier1=[]
    outlier2 =[]
    for x in sample:
        if x<W1:
            outlier1.append(x)
        elif x>W2:
            outlier2.append(x)
    print("out1",outlier1)
    print("out2",outlier2) 

print("x",X)
print("X_1",X_1)
print("X_2",X_0)
findvals(X)
plt.boxplot(X, vert=False)
plt.title("X")
findvals(X_1)
findvals(X_0)
#subplots for the X , X_1 and X_0
fig, xbxplt = plt.subplots()
X_plt = xbxplt.boxplot(X, positions = [1])
X1_plt = xbxplt.boxplot(X_1, positions = [2])
X2_plt = xbxplt.boxplot(X_0, positions = [3])
plt.show()

print("*****************Q3*********************")
#data preparation for boxplots
df = pd.read_csv("fraud.csv")
freq = df["FRAUD"]
spend = df[["FRAUD", "TOTAL_SPEND"]]
dr_visits = df[["FRAUD", "DOCTOR_VISITS"]]
num_claim = df[["FRAUD", "NUM_CLAIMS"]]
mem_dur = df[["FRAUD", "MEMBER_DURATION"]]
opt_pres = df[["FRAUD", "OPTOM_PRESC"]]
num_mem = df[["FRAUD", "NUM_MEMBERS"]]
interval_matrix = df[["FRAUD", "TOTAL_SPEND", "DOCTOR_VISITS", "NUM_CLAIMS", "MEMBER_DURATION", "OPTOM_PRESC", "NUM_MEMBERS"]]

freqlist=[]
for x in freq:
    freqlist.append(x)
freqlist.sort()   
N = len(freqlist) 
print("N",N)
minval = min(freqlist)
maxval = max(freqlist)
countf=0
#calculate fraudulant percentage
for x in freqlist:
    if x == 1:
        countf+=1
fraudperc = ( countf/N ) * 100
print("fraudulant percentage", round(fraudperc,4))
#boxplots for all intervals
spend.boxplot(by="FRAUD", vert=False)
dr_visits.boxplot(by="FRAUD", vert=False)
num_mem.boxplot(by="FRAUD", vert=False)
mem_dur.boxplot(by="FRAUD", vert=False)
opt_pres.boxplot(by="FRAUD", vert=False)
num_claim.boxplot(by="FRAUD", vert=False)
plt.show()

f = np.genfromtxt("fraud.csv", delimiter=',', skip_header=True, usecols=[1])
m = np.genfromtxt("fraud.csv", delimiter=',', skip_header=True, usecols=[2,3,4,5,6,7])
print(m)

#Find eigenvalues and eigen vectors, transformation matrix, transformed matrix and check oerthonormality
mtm = np.matmul(m.transpose(),m)
eigenval, eigenvect = LA2.eigh(mtm)
print("Eigenvalues", eigenval)
trans = eigenvect * LA2.inv(np.sqrt(np.diagflat(eigenval)))
print("Transformation matrix:",trans)
trans_m = np.matmul(m,trans)
print("Transformed Matrix:",trans_m)
IMat = trans_m.transpose().dot(trans_m)
print("I", IMat)

#find nearest neighbours and print the score
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trans_m, f)
sco = nbrs.score(trans_m,f)
print(sco)
new=[[7500, 15, 3, 127, 2, 2]]
trans_new = np.matmul(new,trans)
dist, myNeighbors_t = nbrs.kneighbors(trans_new)
print("My Neighbors = \n", myNeighbors_t)
print("My dist = \n", dist)

#calculate input variable values
print(df.loc[588])
print(df.loc[1199])
print(df.loc[2264])
print(df.loc[1246])
print(df.loc[3809])
proba = nbrs.predict_proba(trans_new)
print("Probability", proba)