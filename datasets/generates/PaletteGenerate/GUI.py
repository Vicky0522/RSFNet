from PIL import Image, ImageEnhance
from PIL.ImageQt import ImageQt, toqpixmap
from palette import *
from util import *
#from test import *
from transfer import *
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from harmonization import auto_palette
#import cv2

html_color = lambda color : '#%02x%02x%02x' % (color[0],color[1],color[2])
color_np = lambda color : np.array([color.red(),color.green(),color.blue()])

class Window(QWidget):
    K = 0 
    palette_button = []
    Source = ''
    image_label = ''
    #cv2Image = []
    img = []
    img_lab = []
    palette_color = (np.zeros((7,3)) + 239).astype(int) #initial grey
    sample_level = 16
    sample_colors = sample_RGB_color(sample_level)
    sample_weight_map = []
    means = []
    means_weight = []
    ###### means <-- lab ; palette_color <-- rgb ######
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pallete Based Photo Recoloring')
        self.UiComponents()
        self.show()

    def palette2mean(self):# rgb to lab
        mean = np.zeros(self.palette_color.shape)
        for i in range(0, self.palette_color.shape[0]):
            rgb = Image.new('RGB',(1,1),html_color(self.palette_color[i]))
            mean[i] = np.array(rgb2lab(rgb).getpixel((0,0)))
        return mean.astype(int)
    def mean2palette(self):# lab to rgb
        palette = np.zeros(self.means.shape)
        for i in range(0, self.means.shape[0]):
            lab = Image.new('LAB',(1,1),html_color(self.means[i].astype(int)))
            palette[i] = np.array(lab2rgb(lab).getpixel((0,0)))
        return palette.astype(int)

    def calc_palettes(self, k):
        self.K = k
        colors = self.img_lab.getcolors(self.img_lab.width * self.img_lab.height)
        bins = {}
        for count, pixel in colors:
            bins[pixel] = count
        bins = sample_bins(bins)
        self.means, self.means_weight = \
            k_means(bins, k=self.K, init_mean=True)
        print(self.means)
        #####means_rgb = cv2.cvtColor(
        #####    self.means.astype(np.ubyte)[None,:,:],cv2.COLOR_Lab2RGB)
        #####print(means_rgb[0])
        ######self.palette_color[:k] = means_rgb[0]
        #####self.palette_color = means_rgb[0]
        self.palette_color = self.mean2palette()
        self.set_palette_color()
        print('original palette')
        print(self.palette_color)
        #print('auto:') 
        #print(auto_palette(self.palette_color, self.means_weight))
    def pixmap_open_img(self, k):
        # load image
        self.img = Image.open(self.Source)
        print(self.Source, self.img.format, self.img.size, self.img.mode)
        # transfer to lab
        self.img_lab = rgb2lab(self.img)
        # get palettes
        self.calc_palettes(k)
        pixmap = toqpixmap(self.img)
        #self.set_number_of_palettes('5') # default 5 palettes
        return pixmap
    def style_transfer(self):
        print('style transfer in development')

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,"QFileDialog.getOpenFileName()", "", \
            "Images (*.jpg *.JPG *jpeg *.png *.webp *.tiff *.tif *.bmp *.dib);;All Files (*)", options=options)
        if len(file_name) == 0:
            return
        # load image
        style_img = Image.open(file_name)
        # transfer to lab
        style_img_lab = rgb2lab(style_img)
        # get palettes
        colors = style_img_lab.getcolors(style_img.width*style_img.height)
        bins = {}
        for count, pixel in colors:
            bins[pixel] = count
        bins = sample_bins(bins)
        style_means, _ = \
            k_means(bins, k=self.K, init_mean=True)
        print('style',style_means)

        # rbf weights
        # style_sample_weight_map = rbf_weights(style_means, self.sample_colors)

        # change GUI palette color
        style_palette = np.zeros(style_means.shape)
        for i in range(0, self.means.shape[0]):
            lab = Image.new('LAB',(1,1),html_color(style_means[i].astype(int)))
            style_palette[i] = np.array(lab2rgb(lab).getpixel((0,0)))
        self.palette_color = style_palette.astype(int)
        self.set_palette_color()

        #transfer
        self.img = img_color_transfer(
            self.img_lab, self.means, style_means, \
            self.sample_weight_map, self.sample_colors, self.sample_level)
        print('Done')
        resized = toqpixmap(self.img).scaledToHeight(512)
        self.image_label.setPixmap(resized)

    def auto(self):
        print(self.palette_color)
        self.palette_color = \
            auto_palette(self.palette_color, self.means_weight)
        self.set_palette_color()
        print(self.palette_color)
        # modify image
        ####palette_color_lab = cv2.cvtColor(
        ####    self.palette_color[None,:,:],cv2.COLOR_RGB2Lab)[0]
        ####print(palette_color_lab)
        self.img = img_color_transfer(
            self.img_lab, self.means, self.palette2mean(), \
            self.sample_weight_map, self.sample_colors, self.sample_level)
        print('Done')
        ## for testing
        #enhancer = ImageEnhance.Brightness(self.img)
        #self.img = enhancer.enhance(1.1)
        # show image
        #self.image_label.setPixmap(pixmap)
        resized = toqpixmap(self.img).scaledToHeight(512)
        self.image_label.setPixmap(resized)
    def clicked(self, N):
        if N >= self.K:
            print('invalid palette')
            return
        print('change palette', N, 'to', end='\t')
        # choose new color
        curr_clr = self.palette_color[N]
        current = QColor(curr_clr[0],curr_clr[1],curr_clr[2])
        color = QColorDialog.getColor(initial=current, 
            options=QColorDialog.DontUseNativeDialog)
        print(color_np(color))
        #new_palette = Image.new('RGB',(1,1),html_color(color_np(color)))
        #palette_lab = np.array(rgb2lab(new_palette).getpixel((0,0)))
        #print(palette_lab)
        ###### test #####
        print(self.palette2mean())
        ###### test #####
        self.palette_color[N] = color_np(color)
        self.set_palette_color()
        # modify image
        ####palette_color_lab = cv2.cvtColor(
        ####    self.palette_color[None,:,:],cv2.COLOR_RGB2Lab)[0]
        ####print(palette_color_lab)
        self.img = img_color_transfer(
            self.img_lab, self.means, self.palette2mean(), \
            self.sample_weight_map, self.sample_colors, self.sample_level)
        print('Done')
        ## for testing
        #enhancer = ImageEnhance.Brightness(self.img)
        #self.img = enhancer.enhance(1.1)
        # show image
        #self.image_label.setPixmap(pixmap)
        resized = toqpixmap(self.img).scaledToHeight(512)
        self.image_label.setPixmap(resized)
    def init_palette_color(self):
        for i in range(7):
            attr = 'background-color:'+html_color(
                self.palette_color[i])+';border:0px'
            self.palette_button[i].setStyleSheet(attr)
    def set_palette_color(self):
        for i in range(self.K):
            attr = 'background-color:'+html_color(
                self.palette_color[i])+';border:0px'
            self.palette_button[i].setStyleSheet(attr)
    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,"QFileDialog.getOpenFileName()", "", \
            "Images (*.jpg *.JPG *jpeg *.png *.webp *.tiff *.tif *.bmp *.dib);;All Files (*)", options=options)
        if len(file_name) == 0:
            return
        self.Source = file_name
        resized = self.pixmap_open_img(5).scaledToHeight(512)
        self.image_label.setPixmap(resized)
        #self.image_label.setPixmap(self.pixmap_open_img(5))
        # rbf weights
        self.sample_weight_map = rbf_weights(self.means, self.sample_colors)
    def reset(self):
        resized = self.pixmap_open_img(self.K).scaledToHeight(512)
        self.image_label.setPixmap(resized)
        #self.image_label.setPixmap(self.pixmap_open_img(self.K))
    def save_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,"QFileDialog.getOpenFileName()", "", \
            "PNG (*.png);;JPG (*.jpg);;Images (*.jpg *.JPG *jpeg *.png *.webp *.tiff *.tif *.bmp *.dib);;All Files (*)", options=options)
        if len(file_name) == 0:
            return
        print('Saving to',file_name)
        if file_name.find('.') == -1:
            file_name += '.png'
        self.img.save(file_name)
        #cv2.imwrite(file_name,self.cv2Image)
        print('Saved to',file_name)
    def set_number_of_palettes(self, text):
        self.K = int(text)
        #self.set_palette_color()
        for i in range(self.K, 7):
            attr = 'background-color:#EFEFEF;border:0px'
            self.palette_button[i].setStyleSheet(attr)
        self.calc_palettes(self.K)
    def UiComponents(self):
        self.main_layout = QVBoxLayout()

        Image_section = QWidget()
        image_section_layout = QHBoxLayout()
        self.image_label = QLabel()
        #label.setPixmap(self.pixmap_img())
        image_section_layout.addWidget(self.image_label)
        Color_wheel = QWidget()
        color_wheel_layout = QVBoxLayout()
        Color_wheel.setLayout(color_wheel_layout)
        image_section_layout.addWidget(Color_wheel)

        self.Palette = QWidget()
        #self.palette_layout = QHBoxLayout()
        self.palette_layout = QVBoxLayout()
        for i in range(7):
            self.palette_button.append(QPushButton())
            self.palette_button[i].clicked.connect(
                lambda state,x=i: self.clicked(x))
            self.palette_layout.addWidget(self.palette_button[i])
        self.init_palette_color()
        self.Palette.setLayout(self.palette_layout)
        image_section_layout.addWidget(self.Palette)
        #self.main_layout.addWidget(self.Palette)

        Image_section.setLayout(image_section_layout)
        self.main_layout.addWidget(Image_section)

        Image_button = QWidget()
        image_button_layout = QHBoxLayout()
        combo_box = QComboBox()
        for i in range(3,8):
            combo_box.addItem(str(i))
        combo_box.activated[str].connect(self.set_number_of_palettes)
        image_button_layout.addWidget(combo_box)
        #combo_box.setCurrentText(str(self.K))
        combo_box.setCurrentText('5')
        open_image = QPushButton('Open')
        reset_image = QPushButton('Reset')
        save_image = QPushButton('Save')
        open_image.clicked.connect(self.open_file)
        reset_image.clicked.connect(self.reset)
        save_image.clicked.connect(self.save_file)
        image_button_layout.addWidget(open_image)
        image_button_layout.addWidget(reset_image)
        image_button_layout.addWidget(save_image)
        Image_button.setLayout(image_button_layout)
        self.main_layout.addWidget(Image_button)
        
        Auto_button = QWidget()
        auto_button_layout = QHBoxLayout()
        auto = QPushButton('Auto')
        style = QPushButton('Style Transfer')
        auto_button_layout.addWidget(auto)
        auto_button_layout.addWidget(style)
        auto.clicked.connect(self.auto)
        style.clicked.connect(self.style_transfer)
        Auto_button.setLayout(auto_button_layout)
        self.main_layout.addWidget(Auto_button)

        self.setLayout(self.main_layout)

app = QApplication([])
window = Window()
app.exec_()
