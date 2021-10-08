import numpy as np
from scipy.special import jv
import time

latt = np.array([[3/2,np.sqrt(3)/2],[3/2,-np.sqrt(3)/2]])


#floquet hamiltonian
def H (k,a,b,phi):
    def A(n):
        if n==3:
            A = a
        else:
            o = (-1)**(np.mod(1,n)+1)
            A = 0.5*np.sqrt(a**2+3*b**2+o*2*np.sqrt(3)*a*b*np.cos(phi))
           
        return A
    def psi(n):
        if n==3:
            psi = 0
        else:
            o = (-1)**(np.mod(1,n)+1)
            psi = o*np.arctan(np.sqrt(3)*b*np.sin(phi)/(a+o*np.sqrt(3)*b*np.cos(phi)))
            
        return psi
    h = np.exp(-1j*np.dot(k,latt[0]))*jv(0,A(1))+np.exp(-1j*np.dot(k,latt[1]))*jv(0,A(2))+jv(0,A(3))
    H = np.linalg.qr(np.array(((0,h),(h.conj(),0))))[0]
    #H = np.array(((0,h),(h.conj(),0)))
    #H = H/np.linalg.norm(H)
    return H

#k space points
kd = 10
k = np.linspace(0,2*np.pi,kd)
kk = np.meshgrid(k,k)
kk = np.array(kk)

kk = np.reshape(kk,[2,kd**2]).T

#vary driving parameters
abd = 40
a = np.linspace(0,5,abd)
b = np.linspace(0,5,abd)
ab = np.meshgrid(a,b)
ab = np.array(ab)
ab = np.reshape(ab,[2,abd**2]).T

N = abd**2


#get hamiltonians
HH = []
for i in range(abd**2):
    h = []
    for j in range(kd**2):
        h.append(H(kk[j],ab[i,0],ab[i,1],np.pi/2))
    HH.append(h)


HH = np.array(HH)
"""
def dd(H):
    A = np.tile(np.reshape(H,[N,1,kd**2,2,2]),[1,N,1,1,1])
    B = np.tile(np.reshape(H,[1,N,kd**2,2,2]),[N,1,1,1,1])
    h = np.matmul(np.transpose(A.conj(),axes=(0,1,2,4,3)),B)
    dists = np.sum(np.trace(h+h.conj(),axis1=3,axis2=4),axis=2)
    dd = 1/N*abs(dists)
    return dd
"""

#distance metric
def dd(a,b):
    
    h = np.matmul(np.transpose(a.conj(),axes=(0,2,1)),b)
    dists = np.sum(np.trace(h+h.conj(),axis1=1,axis2=2))
    dd = 1/(2*N)*abs(dists)
    return dd


dar = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        dar[i,j] = dd(HH[i],HH[j])


#%%

eps = 0.01
ex = 0.5
dar = np.nan_to_num(dar)

def K(a,b):
    #K = dar[a,b]
    #K = np.exp(-1/(2*eps)*(1-((1+k/2)**0.5))
    #K = dd(HH[a],HH[b])
    K = np.exp(-(1-((1+dar[a,b])/2)**ex/eps))
    return K

Kk = np.zeros([N,N])

for i in range(N):
    for j in range(N):
        Kk[i,j] = K(i,j)

Kk = np.nan_to_num(Kk)

def z(K):
    z = np.sum(K,axis=1)
    return z



Zz = z(Kk)

Zz = np.nan_to_num(Zz)

def P(K,z):
    P = K/np.tile(np.reshape(z,[N,1]),[1,N])
    return P

Pp = P(Kk,Zz)

Pp = np.nan_to_num(Pp)

#%%
def D(a,b,z,P):

    D = np.sum((P[a,:]-P[b,:])**2)
    return D
#%%
def A(z,p):
    Zz = np.tile(np.reshape(z,[N,1]),[1,N])
    zZ = np.tile(np.reshape(z,[1,N]),[N,1])
    Aa = p*np.sqrt(Zz/zZ)
    return Aa

a = A(Zz,Pp)

e = np.linalg.eigvalsh(a)
#%%
"""
#generate diffusion map
ds = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        ds[i,j] = D(i,j,Zz,Pp)
"""
#%%
from sklearn.cluster import KMeans, AgglomerativeClustering



means = KMeans(3)
vals = means.fit_predict(Pp)

import matplotlib.pyplot as plt
import pickle
#pickle.dump([ab,vals],open("PhaseKmeans2.p",'wb'))


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_aspect('equal')
plt.title('Phases')
plt.scatter(ab[:,0],ab[:,1],c=vals,marker=',')

plt.figure()
plt.scatter(np.arange(10),r[-10:])
plt.title('10 largest eigvals')
