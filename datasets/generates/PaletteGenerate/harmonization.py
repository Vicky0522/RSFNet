import math
import numpy as np
import cv2 

type_m = ['i type', 'L type', 'mirror-L type', 'V type', 'I type', 'Y type', 'X type', 'T type'] 
type_name = ['i-type', 'L-type', 'mirror-L', 'V-type', 'I-type', 'Y-type', 'X-type', 'T-type'] 
Tm = np.array([[0,9,0,9],[0,39.6,90,9],[90,39.6,0,9],[0,46.8,0,46.8],[0,9,180,9],[0,9,180,46.8],[0,46.8,180,46.8],[0,90,0,90]])
phi = 0.38197

def shift_hue(img, deg):
    #deg = 180 * deg / 360
    #deg = (deg + 180) % 180
    deg = fit_hue(deg)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i][j][0] = (img[i][j][0] + deg) % 180
def G(sigma, x):
    return math.exp(-1 * x**2 / (2 * sigma**2))
def arc_dist(a1, a2):
    return min((a1 - a2) % 360, (a2 - a1) % 360)
def arc_dist_180(a1, a2):
    return real_hue(min((a1 - a2) % 180, (a2 - a1) % 180))
def dtr(t):
    return t * math.pi / 180
def rth(t): ## radians to hue 
    return t * 90 / math.pi
def fit_hue(h):
    return 180 * (h % 360) / 360 ## 0 < h < 180
def real_hue(h):
    return 360 * (h % 180) / 180 ## 0 < h < 360
def hue_border_dist(m, alpha, hue): ## 0 < alpha, hue < 360
    ret1 = 180
    delta = abs((hue - (Tm[m][0] + alpha) + 180) % 360 - 180) # 0 < d < 180
    if delta < Tm[m][1]: 
        return 0
    else:
        ret1 = delta - Tm[m][1]
    ret2 = 180
    delta = abs((hue - (Tm[m][2] + alpha) + 180) % 360 - 180) # 0 < d < 180
    if delta < Tm[m][3]: 
        return 0
    else:
        ret2 = delta - Tm[m][3]
    return dtr(ret1) if ret1 < ret2 else dtr(ret2) ## ret > 0 (radians)
def direction(m, hue, C1, C2): ## 0 < alpha, hue < 180
    ret1 = 90
    delta = abs((hue - C1 + 90) % 180 - 90) # 0 < d < 180
    if delta < Tm[m][1]: 
        return 1
    else:
        ret1 = delta - Tm[m][1]
    ret2 = 180
    delta = abs((hue - C2 + 90) % 180 - 90) # 0 < d < 180
    if delta < Tm[m][3]: 
        return 2
    else:
        ret2 = delta - Tm[m][3]
    return 1 if ret1 < ret2 else 2 

def determine_F(img, m, alpha, weight):
    f = 0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            f += hue_border_dist(m, alpha, real_hue(img[i][j][0])) \
                * img[i][j][1] * weight[i][j]
    return f ## f > 0
def angle_mid(a0, a1):
    r = 0.5
    ret = 0
    if (a0 - a1) % 360 < (a1 - a0) % 360:
        ret = (a0 - ((a0 - a1) % 360) * r) % 360
    else:
        ret = (a0 + ((a1 - a0) % 360) * r) % 360
    return ret
def brent(img, m, w):
    ##### brent's method #####
    a0, a1, a2 = 0, 120, 240
    #f0, f1, f2
    f0 = determine_F(img, m, a0, w)
    f1 = determine_F(img, m, a1, w)
    f2 = determine_F(img, m, a2, w)
    if f0 < f1 and f0 < f2:
        a0, a1, a2 = a2, a0, a1
        f0, f1, f2 = f2, f0, f1
    elif f2 < f1 and f2 < f0:
        a0, a1, a2 = a1, a2, a0 
        f0, f1, f2 = f1, f2, f0 
    while (arc_dist(a2,a0) > 0.5):
        ##if abs((a1 - a0) % 360) > abs((a2 - a1) % 360):
        if arc_dist(a1, a0) > arc_dist(a2, a1):
            x = angle_mid(a0,a1)
            fx = determine_F(img, m, x, w)
            #print(a0,'\t',x,'\t',a1,'\t',a2)
            if fx < f1:
                a2, f2 = a1, f1
                a1, f1 = x, fx
            else:
                a0, f0 = x, fx
        else:
            x = angle_mid(a2,a1)
            fx = determine_F(img, m, x, w)
            #print(a0,'\t',a1,'\t',x,'\t',a2)
            if fx < f1:
                a0, f0 = a1, f1
                a1, f1 = x, fx
            else:
                a2, f2 = x, fx
    return f1, m, a1
    ##### brent's method #####
def harm_shift(H_img, m, alpha, s_w):
    w1 = 2 * dtr(Tm[m][1])
    w2 = 2 * dtr(Tm[m][3])
    ##print(w1,w2)
    sigma1 = w1 * s_w
    sigma2 = w2 * s_w
    C1 = fit_hue(Tm[m][0] + alpha)
    C2 = fit_hue(Tm[m][2] + alpha)
    for i in range(0,H_img.shape[0]):
        for j in range(0,H_img.shape[1]):
            #print(H_img[i][j])
            #if arc_dist_180(H_img[i][j][0], C1) < arc_dist_180(H_img[i][j][0], C2) :
            #print(m, H_img[i][j][0], C1, C2)
            if direction(m, H_img[i][j][0], C1, C2) == 1:
                if (C1 - H_img[i][j][0]) % 180 < (H_img[i][j][0] - C1) % 180 :
                    H_img[i][j][0] = \
                        (C1 - rth(w1)/2 * (1-G(sigma1, dtr(arc_dist_180(H_img[i][j][0],C1))))) % 180
                else:
                    H_img[i][j][0] = \
                        (C1 + rth(w1)/2 * (1-G(sigma1, dtr(arc_dist_180(H_img[i][j][0],C1))))) % 180
            else:
                if (C2 - H_img[i][j][0]) % 180 < (H_img[i][j][0] - C2) % 180 :
                    H_img[i][j][0] = \
                        (C2 - rth(w2)/2 * (1-G(sigma2, dtr(arc_dist_180(H_img[i][j][0],C2))))) % 180
                else:
                    H_img[i][j][0] = \
                        (C2 + rth(w2)/2 * (1-G(sigma2, dtr(arc_dist_180(H_img[i][j][0],C2))))) % 180
            #print(H_img[i][j])
    return H_img

def auto_palette(palette_rgb, weight): # K x 3 numpy array
    k = palette_rgb.shape[0]
    palette_rgb = palette_rgb.reshape(1,k,3)
    weight = weight.reshape(1,k)
    palette_hsv = cv2.cvtColor(palette_rgb.astype(np.ubyte), cv2.COLOR_RGB2HSV)

    Result = np.zeros((8,3))
    for m in range(0,8):
        Result[m] = brent(palette_hsv,m,weight)
        print('F =',Result[m][0],'\talpha =',Result[m][2])
    
    Result = Result[np.argsort(Result[:,0])]
    print(Result)
    m = 0
    print(m)
    M, Alpha = int(Result[m][1]), Result[m][2]
    sigma_w_ratio = 0.5
    palette_hsv = harm_shift(palette_hsv, M, Alpha, sigma_w_ratio)
    palette_rgb = cv2.cvtColor(palette_hsv,cv2.COLOR_HSV2RGB)##back to RGB
    return palette_rgb[0]


#################### Start of Program ####################
def main():
    name = str(input('File Name: '))
    Img = cv2.imread(name,cv2.IMREAD_COLOR)
    while Img is None :
        name = str(input('File Name: '))
        Img = cv2.imread(name,cv2.IMREAD_COLOR)
    Img = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)#0 Hue #1 Sat #2 Value
    img = Img.copy()
    ratio = max(100 / img.shape[0], 300 / img.shape[1])
    ratio = ratio if ratio < 1 else 1
    img = cv2.resize(img,None,fx = ratio,fy = ratio)
    height, width = img.shape[:2]
    ##print(width, height )
    print('Calculating')
    print('estimated',round(img.shape[0] * img.shape[1] / 45000), 'minutes')
    
    Result = np.zeros((8,3))
    weight = np.ones((image.shape[0],image.shape[1]))
    for m in range(0,8):
        print(type_m[m])
        Result[m] = brent(img,m,weight)
        print('F =',Result[m][0],'\talpha =',Result[m][2])
    
    Result = Result[np.argsort(Result[:,0])]
    print(Result)
    
    m = 0
    M, Alpha = int(Result[m][1]), Result[m][2]
    sigma_w_ratio = 1/2
    while True:
        print(type_m[M], Alpha)
        tmp_img = Img.copy()
        ratio = 300 / tmp_img.shape[1]
        tmp_img = cv2.resize(tmp_img,None,fx = ratio,fy = ratio)
        harm_shift(tmp_img, M, Alpha, sigma_w_ratio)
        tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_HSV2BGR) ##back to BGR
        tmp_img = cv2.resize(tmp_img,(width * 5,height * 5))
        print('[A(accept)] [N(next)] [P(previous)] [T(type)]')
        cv2.imshow('Preview',tmp_img)
        cv2.waitKey(3000)
        c = input()
        if c == 'A' or c == 'Accept' or c == 'a' or c == 'accept' :
            cv2.destroyWindow('Preview')
            break
        elif c == 'N' or c == 'Next' or c == 'n' or c == 'next' :
            m = (m + 1) % 8
            M, Alpha = int(Result[m][1]), Result[m][2]
        elif c == 'P' or c == 'Previous' or c == 'p' or c == 'previous' :
            m = (m - 1) % 8
            M, Alpha = int(Result[m][1]), Result[m][2]
        elif c == 'T' or c == 'Type' or c == 't' or c == 'type' :
            c = input('[i] [V] [L] [I] [T] [Y] [X] [M(mirror-L)]')
    
    print('Processing')
    print('estimated',round(Img.shape[0] * Img.shape[1] / 2500000), 'minutes')
    harm_shift(Img, M, Alpha, sigma_w_ratio)
    Img = cv2.cvtColor(Img,cv2.COLOR_HSV2BGR) ##back to BGR
    name_prefix = name.rsplit('.')[0]
    name_suffix = '.' + name.rsplit('.')[1] if name.count('.') > 0 else ''
    name_harmonized = name_prefix + '_harmonized_' + type_name[M] +  name_suffix
    print(name_harmonized)
    cv2.imwrite(name_harmonized,Img)
    print('Finished')
    cv2.imshow('Harmonized',Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
