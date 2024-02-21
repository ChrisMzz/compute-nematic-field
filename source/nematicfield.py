from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from copy import deepcopy
from tqdm import tqdm
#import napari
import sphinx

h,w = 300, 400
n = 400

sigma = 0.5

INTENSITY_THRESHOLD = 1e-3


def loop_over_positions(shape, k=0) -> list[tuple]: 
    """Provides list of all positions in an ndarray of given shape.

    Args:
        shape (tuple): shape of ndarray
    """
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

def make_random_line_image(h:int, w:int, n:int) -> np.ndarray:
    """Generate a mask of random lines for testing. Lines are of length somewhere in [0 to h/10] in height and somewhere in [0 to w/10] in width.

    Args:
        h (int): Height of image in px.
        w (int): Width of image in px.
        n (int): Maximum number of lines.

    Returns:
        ndarray: Array with values 0 and 1 representing mask of image.
    """
    img = np.zeros((h,w))
    for r0, c0 in zip(np.random.randint(0,h,n),np.random.randint(0,w,n)):
        dr,dc = np.random.randint(max(-int(h/10),-int(w/10)),min(int(h/10),int(w/10)),2)
        r1, c1 = r0+dr, c0+dc
        if (r0+dr >= h or c0+dc >= w) or (r0+dr < 0 or r0+dc < 0): continue
        rr, cc = sk.draw.line(r0,c0,r1,c1)
        img[rr,cc] = 1
    return img

def tesselate(img:np.ndarray, th:int, tw:int, debug=False) -> list[np.ndarray]:
    """Make list of rectangle subdivisions of an array, such that every element in original in array appears in one single subdivision.

    Args:
        img (np.ndarray): Initial array to subdivide.
        th (int): number of subdivisions along array height.
        tw (int): number of subdivisions along array width.
        debug (bool, optional): Whether to also provide positions of subdivisons in array. Defaults to False.

    Raises:
        Exception: If provided th and tw aren't divisors of initial array, it's impossible to return such a subdivision array.

    Returns:
        list of ndarray representing each subdivision
    """
    h,w = img.shape
    steph, stepw = int(h/th), int(w/tw)
    if steph*th != h or stepw*tw != w: raise Exception("Dimensions provided aren't divisors of image shape.")
    if debug:
        return [(img[dh:dh+steph, dw:dw+stepw], (dh,dh+steph, dw,dw+stepw)) for dw in range(0,w,stepw) for dh in range(0,h,steph)]
    return [img[dh:dh+steph, dw:dw+stepw] for dw in range(0,w,stepw) for dh in range(0,h,steph)]



def kernel(dr:np.ndarray) -> np.ndarray:
    """Kernel used in integration.
     
    Args:
        dr (np.ndarray): Vector along which the kernel is computed.

    Returns:
        ndarray: (2,2)-array representing the resulting nematic.
    """
    if np.linalg.norm(dr) == 0: return np.zeros((2,2))
    phi = np.angle(complex(*dr))
    phi = (phi + np.pi)*(phi < 0) + phi*(phi > 0) # angle in [0,pi)
    return np.exp(-(np.linalg.norm(dr)**2)/2*sigma**2)*np.array([[np.cos(2*phi),  np.sin(2*phi)],
                                                                 [np.sin(2*phi), -np.cos(2*phi)]])


def integrate_over_area(a:np.ndarray, radius:int) -> np.ndarray:
    """Run integration over a given area array `a`. Each element in the array will be summed over and an "average nematic" 
    will be computed by integrating over a circle around it, of a given radius. The bigger the circle, the more varied will 
    be the radii in the circle and therefore the more precise will be the computation.

    Args:
        a (np.ndarray): area over which integral is calculated
        radius (int): radius of mini-integration circles

    Returns:
        ndarray: (2,2)-array representing the resulting nematic.
    """
    positions = loop_over_positions(a.shape)
    s = np.zeros((2,2)) # "empty" nematic
    for r in positions:
        if a[r] < INTENSITY_THRESHOLD: continue
        circle = sk.draw.circle_perimeter(*r, radius, shape=a.shape)
        for p in zip(*circle): # p = r + dr
            line = sk.draw.line(*r, *p) # to include midways in lines and avoid gaps, despite making dr non constant
            s += np.linalg.norm(np.array(p)-np.array(r))*kernel(np.array(p)-np.array(r))*sum([a[r]*a[pr] for pr in zip(*line)])
            # /(a.shape[0]*a.shape[1])
    return s

def extract(Q:np.ndarray):
    """Get norm and angle from a (2,2)-array nematic.

    Args:
        Q (np.ndarray): (2,2)-array representing the nematic

    Returns:
        tuple: (norm, angle) where the angle is in [0, pi)
    """
    Qxx, Qxy = Q[0,0], Q[0,1]
    Qnorm = np.sqrt(Qxx**2 + Qxy**2)
    if Qnorm == 0: return 0, 0
    return Qnorm, np.arctan2(Qxy, Qxx)/2
    
def nematic_field(img:np.ndarray, x, y, r=2, **kwargs) -> np.ndarray:
    """Compute nematic field of an image, with given resolution, integration circle radius, and extra parameters.

    Args:
        img (np.ndarray): Initial image.
        x (int): number of subdivisions along array height.
        y (int): number of subdivisions along array width.
        r (int, optional): Radius of integration circles. Defaults to 2.

    Kwargs:
        uniform_bg (bool): Whether to hide squares and instead opt to colour the lines themselves. Defaults to True.

    Returns:
        ndarray: Image of nematic field according to given parameters.
    """
    uniform_bg = True if 'uniform_bg' not in kwargs.keys() else kwargs['uniform_bg']
    
    imghelper = np.zeros(img.shape)
    h,w = img.shape
    rl, cl = max((min(h/x,w/y)//2),1), max((min(h/x,w/y)//8),1)
    if rl==1 or cl==1:
        def drawer(t1,t2):
            left, right = complex((t1+t2*1j) - (min(h/x,w/y)//4)*np.exp(1j*phi)), complex((t1+t2*1j) + (min(h/x,w/y)//4)*np.exp(1j*phi))
            return sk.draw.line(int(left.real), int(left.imag), int(right.real), int(right.imag))
    else:
        def drawer(t1,t2):
            return sk.draw.ellipse(t1,t2,rl,cl,img.shape,rotation=phi)
        
    if uniform_bg:
        for a,pos in tqdm(tesselate(img, x,y, debug=True)):
            nem = integrate_over_area(a,r)
            Qnorm, phi = extract(nem)
            t1, t2 = int((pos[0]+pos[1])/2), int((pos[2]+pos[3])/2)
            imghelper[drawer(t1,t2)] = Qnorm
    else:
        for a,pos in tqdm(tesselate(img, x,y, debug=True)):
            nem = integrate_over_area(a,r)
            Qnorm, phi = extract(nem)
            imghelper[pos[0]:pos[1],pos[2]:pos[3]] = Qnorm
            t1, t2 = int((pos[0]+pos[1])/2), int((pos[2]+pos[3])/2)
            imghelper[drawer(t1,t2)] = 0
    return imghelper
    

if __name__ == '__main__':

    IMG = make_random_line_image(h,w,n)
    #plt.imshow(IMG)
    #plt.show()
    sk.io.imsave('test2.png',IMG)
    #A = tesselate(IMG, 3,4, debug=True)


    #IMG = 1*(sk.io.imread('dessin.png') > 0)
    nem_field = nematic_field(IMG, 15, 20)

    #fig, (ax1,ax2) = plt.subplots(1,2)
    #ax1.imshow(IMG)
    #ax2.imshow(nem_field)
    #plt.show()

    sk.io.imsave('test2_uniformbg.tif',np.array([IMG, nem_field/np.max(nem_field)]))
