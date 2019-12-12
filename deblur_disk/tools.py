import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
import scipy.ndimage

def SingleChannelConvolution(input_image, kernel, pad):    
    padded = F.pad(input_image.unsqueeze(0).unsqueeze(0), pad=pad, mode='replicate')
    output = F.conv2d(padded, kernel.unsqueeze(0).unsqueeze(0), padding = 0)
    return output.data.squeeze()

def RGBConvolution(input_image, kernel, pad):
    output1 = SingleChannelConvolution(input_image[0], kernel, pad)
    output2 = SingleChannelConvolution(input_image[1], kernel, pad)
    output3 = SingleChannelConvolution(input_image[2], kernel, pad)
    output = torch.cat((output1.unsqueeze(0), output2.unsqueeze(0), output3.unsqueeze(0)), 0)
    return output

def complex_multiplication(t1, t2):
    real1, imag1 = t1.transpose(0,4)
    real2, imag2 = t2.transpose(0,4)
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = 0).transpose(0,4)

def complex_division(t1, t2):
    real2, imag2 = t2.transpose(0,4)
    temp = complex_multiplication(t1, torch.stack([real2, -imag2], dim=0).transpose(0, 4))
    norm = (real2**2 + imag2**2).unsqueeze(0).transpose(0, 4)
    return temp/norm

# pad a kernel to imagesize and shift it
def pad_and_shift(k, image_size):
    a, b = image_size
    w, l = k.shape
    a = int(a/2)
    b = int(b/2)
    w = int((w - 1)/2)
    l = int((l - 1)/2)
    kpad = np.zeros(image_size)
    kpad[a-w:a+w+1, b-l:b+l+1] = k
    kshift = np.fft.fftshift(kpad)
    return kshift

'''
Create predefined 2-D blur kernel filters

'disk'      circular averaging filter
'motion'    motion filter

k = fspecial('disk',RADIUS) returns a circular averaging filter
(pillbox) within the square matrix of side 2*RADIUS+1.
The default RADIUS is 5.

k = fspecial('motion',LEN,THETA) returns a filter to approximate, once
convolved with an image, the linear motion of a camera by LEN pixels,
with an angle of THETA degrees in a counter-clockwise direction. The
filter becomes a vector for horizontal and vertical motions.  The
default LEN is 9, the default THETA is 0, which corresponds to a
horizontal motion of 9 pixels.
'''

def fspecial(type, p1=None, p2=None):

    if type == 'disk':
        if p1: rad = p1 
        else: rad = 5
        crad = np.ceil(rad-0.5).astype('int')
        x, y = np.meshgrid(np.arange(-crad, crad+1), np.arange(-crad, crad+1))
        maxxy = np.maximum(abs(x),abs(y))
        minxy = np.minimum(abs(x),abs(y))
        m1 = (rad**2 < (maxxy+0.5)**2 + (minxy-0.5)**2)*(minxy-0.5) + \
            (rad**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2)* \
            np.nan_to_num(np.sqrt(rad**2 - (maxxy + 0.5)**2)) 
        m2 = (rad**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2)*(minxy+0.5) + \
            (rad**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2)* \
            np.nan_to_num(np.sqrt(rad**2 - (maxxy - 0.5)**2))
        sgrid = (rad**2*(0.5*(np.arcsin(m2/rad) - np.arcsin(m1/rad)) + \
            0.25*(np.sin(2*np.arcsin(m2/rad)) - np.sin(2*np.arcsin(m1/rad)))) - \
            (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5)) \
            *((((rad**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & \
            (rad**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | \
            ((minxy==0)&(maxxy-0.5 < rad)&(maxxy+0.5>=rad))))
        sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < rad**2)
        sgrid[crad, crad] = min(np.pi*rad**2, np.pi/2)
        if ((crad>0) and (rad > crad-0.5) and (rad**2 < (crad-0.5)**2+0.25)):
            m1  = np.sqrt(rad**2 - (crad - 0.5)**2)
            m1n = m1/rad
            sg0 = 2*(rad**2*(0.5*np.arcsin(m1n) + 0.25*sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
            sgrid[2*crad, crad] = sg0
            sgrid[crad, 2*crad] = sg0
            sgrid[crad, 0]        = sg0
            sgrid[0, crad]        = sg0
            sgrid[2*crad-1, crad]   = sgrid[2*crad-1,crad] - sg0
            sgrid[crad, 2*crad-1]   = sgrid[crad,2*crad-1] - sg0
            sgrid[crad, 1]        = sgrid[crad,1]      - sg0
            sgrid[1, crad]        = sgrid[1,crad]      - sg0
        sgrid[crad, crad] = min(sgrid[crad,crad],1)
        h = sgrid/sgrid.sum()
        
    elif type == 'motion':
        if p1: len = max(1, p1)
        else: len = 9
        half = (len-1)/2        # rotate half length around center
        if p2: phi = p2%180/180*np.pi
        else: phi = 0
        
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xsign = np.sign(cosphi)
        linewdt = 1
        eps = np.finfo('float').eps
        
        # define mesh for the half matrix, eps takes care of the right size
        # for 0 & 90 rotation
        sx = np.fix(half*cosphi + linewdt*xsign - len*eps)
        sy = np.fix(half*sinphi + linewdt - len*eps)
        x, y = np.meshgrid(np.arange(0,sx+xsign,xsign),np.arange(0,sy+1))

        # define shortest distance from a pixel to the rotated line
        dist2line = (y*cosphi-x*sinphi)     # distance perpendicular to the line
        
        rad = np.sqrt(x**2 + y**2)
        # find points beyond the line's end-point but within the line width
        lastpix = (rad >= half) & (abs(dist2line)<=linewdt)
        # distance to the line's end-point parallel to the line
        x2lastpix = half - abs((x[lastpix] + dist2line[lastpix]*sinphi)/cosphi)
        
        dist2line[lastpix] = np.sqrt(dist2line[lastpix]**2 + x2lastpix**2)
        dist2line = linewdt + eps - abs(dist2line)
        dist2line[dist2line<0] = 0          # zero out anything beyond line width
        
        # unfold half-matrix to the full size
        hm = np.rot90(dist2line,2)
        a,b = hm.shape
        fm = np.zeros((2*a-1, 2*b-1))
        fm[0:a, 0:b] = hm
        fm[a-1:2*a,b-1:2*b] = dist2line
        h = fm/(fm.sum() + eps*len*len)
        
        if cosphi > 0:
            h = np.flipud(h)  
    else:
        raise Exception('input type only accepts \'disk\' and \'motion\'')

    return h

# convolve a 3 tunnel image with a given disk kernel 
def conv_disk(x, r):
    k = fspecial('disk', r)
    blurred = x.copy()
    if len(blurred.shape) == 4:
        for i in range(blurred.shape[0]):
            for j in range(3):
                blurred[i,j,:,:] = scipy.ndimage.convolve(x[i,j,:,:], k, mode='nearest')
    else:
        for i in range(3):
            blurred[i,:,:] = scipy.ndimage.convolve(x[i,:,:], k, mode='nearest')
    
    return blurred


def conv_motion(x, r, theta):
    k = fspecial('motion', r, theta)
    blurred = np.zeros(x.shape)
    if len(blurred.shape) == 4:
        for i in range(blurred.shape[0]):
            for j in range(3):
                blurred[i,j,:,:] = scipy.ndimage.convolve(x[i,j,:,:], k, mode='nearest')
    else:
        for i in range(3):
            blurred[i,:,:] = scipy.ndimage.convolve(x[i,:,:], k, mode='nearest')
    
    return blurred