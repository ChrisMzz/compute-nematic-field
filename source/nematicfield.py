import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from copy import deepcopy
from tqdm import tqdm
#import napari

h,w = 300, 400
n = 800

sigma = 0.5

def loop_over_positions(shape, k=0): # all positions
        if k == len(shape)-1: 
            temp_list = []
            for j in range(shape[-1]): 
                temp = list(shape)
                temp[-1] = j
                temp_list.append(tuple(temp))
            return temp_list
        temp_list = []
        for j in range(shape[k]):
            temp = list(shape)
            temp[k] = j
            temp_list += loop_over_positions(tuple(temp), k+1)
        return temp_list

def make_random_line_image(h,w,n):
    img = np.zeros((h,w))
    for r0, c0 in zip(np.random.randint(0,h,n),np.random.randint(0,w,n)):
        dr,dc = np.random.randint(0,min(int(h/10),int(w/10)),2)
        r1, c1 = r0+dr, c0+dc
        if r0+dr >= h or c0+dc >= w: continue
        rr, cc = sk.draw.line(r0,c0,r1,c1)
        img[rr,cc] = 1
    return img

def tesselate(img:np.ndarray, th:int, tw:int, debug=False):
    h,w = img.shape
    steph, stepw = int(h/th), int(w/tw)
    if steph*th != h or stepw*tw != w: raise Exception("Dimensions provided aren't divisors of image shape.")
    if debug:
        return [(img[dh:dh+steph, dw:dw+stepw], (dh,dh+steph, dw,dw+stepw)) for dw in range(0,w,stepw) for dh in range(0,h,steph)]
    return [img[dh:dh+steph, dw:dw+stepw] for dw in range(0,w,stepw) for dh in range(0,h,steph)]

def kernel(dr:np.ndarray):
    if np.linalg.norm(dr) == 0: return 0
    phi = np.angle(complex(*dr))
    return np.exp(-(np.linalg.norm(dr)**2)/2*sigma**2)*np.array([[np.cos(2*phi),  np.sin(2*phi)],
                                                                 [np.sin(2*phi), -np.cos(2*phi)]])


def integrate_over_area(a:np.ndarray, radius):
    positions = loop_over_positions(a.shape)
    s = np.zeros((2,2)) # initial nematic
    for r in positions:
        circle = sk.draw.circle_perimeter(*r, radius, shape=a.shape)
        for p in zip(*circle): # p = r + dr for biggest dr
            line = sk.draw.line(*r, *p) # to include midways in lines and avoid gaps, despite making dr non constant
            s += sum([np.linalg.norm(np.array(pr)-np.array(r))*a[r]*a[pr]*kernel(np.array(pr)-np.array(r)) for pr in zip(*line)])
            # /(a.shape[0]*a.shape[1])
    return s

def extract(Q:np.ndarray):
    Qxx, Qxy = Q[0,0], Q[0,1]
    Qnorm = np.sqrt(Qxx**2 + Qxy**2)
    if Qnorm == 0: return 0, 0
    return Qnorm, np.arccos(Qxx/Qnorm)/2
    #      Qnorm, np.arcsin(Qxy/Qnorm)
    
def nematic_field(img:np.ndarray, x, y, r=2):
    imghelper = np.zeros(img.shape)
    for a,pos in tqdm(tesselate(img, x,y, debug=True)):
        nem = integrate_over_area(a,r)
        Qnorm, phi = extract(nem)
        imghelper[pos[0]:pos[1],pos[2]:pos[3]] = Qnorm
        # lines drawn don't ever point up-right or down-left, may be a bug in code
        t1, t2 = int((pos[0]+pos[1])/2), int((pos[2]+pos[3])/2)
        left, right = (t1+t2*1j) - 3*np.exp(1j*phi), (t1+t2*1j) + 3*np.exp(1j*phi)
        imghelper[sk.draw.line(int(left.real), int(left.imag), int(right.real), int(right.imag))] = 0
    return imghelper
    
    

#IMG = make_random_line_image(h,w,n)
#plt.imshow(IMG)
#plt.show()
#sk.io.imsave('test4.png',IMG)
#A = tesselate(IMG, 3,4, debug=True)

"""
for a,pos in A:
    fig, (ax1, ax2) = plt.subplots(1,2)
    imghelper = deepcopy(IMG)
    imghelper[pos[0]:pos[1],pos[2]:pos[3]] += 1
    ax1.imshow(imghelper)
    ax2.imshow(a)
    plt.show()
"""

IMG = 1*(sk.io.imread('dessin.png') > 0)
nem_field = nematic_field(IMG, 30, 40)

#fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.imshow(IMG)
#ax2.imshow(nem_field)
#plt.show()

sk.io.imsave('nemfield_dessin10.tif',np.array([IMG, nem_field/np.max(nem_field)]))
