from PIL import Image, ImageCms
import numpy as np

def distance(p1, p2):
	p1, p2 = np.array(p1), np.array(p2)
	return np.sum((p1 - p2) ** 2) ** 0.5

def rgb2lab(image):
	RGB_p = ImageCms.createProfile('sRGB')
	LAB_p = ImageCms.createProfile('LAB')
	return ImageCms.profileToProfile(image, RGB_p, LAB_p, outputMode='LAB')

def lab2rgb(image):
	RGB_p = ImageCms.createProfile('sRGB')
	LAB_p = ImageCms.createProfile('LAB')
	return ImageCms.profileToProfile(image, LAB_p, RGB_p, outputMode='RGB')

def RegularLAB(LAB):
    return (LAB[0] / 255 * 100, LAB[1] - 128, LAB[2] - 128)

def ValidLAB(LAB):
    L, a, b = LAB
    return 0 <= L <= 100 and -128 <= a <= 127 and -128 <= b <= 127

def ValidRGB(RGB):
    return False not in [0 <= x <= 255 for x in RGB]

def LABtoXYZ(LAB):
	# convert one lab pixel to xyz
	def f(n):
		return n**3 if n > 6/29 else 3 * ((6/29)**2) * (n - 4/29)

	assert(ValidLAB(LAB))

	L, a, b = LAB
	X = 95.047 * f((L+16)/116 + a/500)
	Y = 100.000 * f((L+16)/116)
	Z = 108.883 * f((L+16)/116  - b/200)
	return (X, Y, Z)

def XYZtoRGB(XYZ):
	# convert one xyz pixel to RGB
	def f(n):
		return n*12.92 if n <= 0.0031308 else (n**(1/2.4)) * 1.055 - 0.055

	X, Y, Z = [x/100 for x in XYZ]
	R = f(3.2406*X + -1.5372*Y + -0.4986*Z) * 255
	G = f(-0.9689*X + 1.8758*Y + 0.0415*Z) * 255
	B = f(0.0557*X + -0.2040*Y + 1.0570*Z) * 255
	return (R, G, B)

def LABtoRGB(LAB):
	return XYZtoRGB(LABtoXYZ(LAB))

def ByteLAB(LAB):
    return (int(LAB[0] / 100 * 255), int(LAB[1] + 128), int(LAB[2] + 128))
