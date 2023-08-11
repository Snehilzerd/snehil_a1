import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#vertices
A = np.array([1, -1])
B = np.array([-4, 6])
C = np.array([-3, -5])
omat = np.array([[0, 1], [-1, 0]])

def dir_vec(A, B):
    return B - A

def norm_vec(A, B):
    return omat @ dir_vec(A, B)

k1=1
k2=1

p = np.zeros(2)
t = norm_vec(B, C)
n1 = t / np.linalg.norm(t)
t = norm_vec(C, A)
n2 = t / np.linalg.norm(t)
t = norm_vec(A, B)
n3 = t / np.linalg.norm(t)

p[0] = n1 @ B - k1 * n2 @ C
p[1] = n2 @ C - k2 * n3 @ A

N = np.block([[n1 - k1 * n2],[ n2 - k2 * n3]])
I = np.linalg.inv(N)@p
r = n1 @ (B-I)

print("Coordinates of point I:", I)
print(f"Distance from I to AB=Distance from I to CA= {r}")

def alt_foot(A,B,C):
  m = B-C
  n = omat@m 
  N=np.block([[m],[n]])
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P

D =  alt_foot(I,B,C)
E =  alt_foot(I,B,A)
F =  alt_foot(I,C,A)

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
x_DI = line_gen(I,D)
x_CI = line_gen(I,E)
x_BI = line_gen(I,F)



#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_DI[0,:],x_DI[1,:],label='$DI$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')

#Labeling the coordinates
tri_coords = np.block([[A],[B],[C],[I],[D],[E],[F]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.grid() #minor
plt.savefig("figure")       
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig("Incentre.png",bbox_inches='tight')
