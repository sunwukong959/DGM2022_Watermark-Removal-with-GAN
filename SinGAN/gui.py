import os
import sys

from PIL.ImageEnhance import Color
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QAction, QFileDialog, QInputDialog, QLabel, QMainWindow, QApplication, QPushButton, \
    QVBoxLayout, QWidget
from PyQt5.QtGui import QBitmap, QPixmap, QPainter, QPen

import inpainting


class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        self.drawing = False
        self.lastPoint = QPoint()
        self.image = QPixmap(720, 576)
        self.image_location = ''
        self.resize(720, 576)
        self.mask = QBitmap(720, 576)

        self.pen_size = 8

        # menu bar
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')
        tools_menu = main_menu.addMenu('Tools')

        # load image
        load_image_action = QAction('Load Image', self)
        load_image_action.setStatusTip('loads an image from the folder')
        load_image_action.triggered.connect(self.load_image_handler)
        file_menu.addAction(load_image_action)

        # save image
        save_image_action = QAction('Safe Image', self)
        save_image_action.setStatusTip('safes the shown image')
        save_image_action.triggered.connect(self.save_image_handler)
        file_menu.addAction(save_image_action)

        # remove watermarks
        remove_watermark_action = QAction('Remove Watermarks', self)
        remove_watermark_action.setStatusTip('removes marked areas from the image and in-paints them')
        remove_watermark_action.triggered.connect(self.remove_watermarks_handler)
        tools_menu.addAction(remove_watermark_action)

        # resize pen
        resize_pen_action = QAction('Resize Pen', self)
        resize_pen_action.setStatusTip('resizes pen that is used to draw')
        resize_pen_action.triggered.connect(self.resize_pen_handler)
        tools_menu.addAction(resize_pen_action)

        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)
        painter.drawPixmap(self.rect(), self.mask)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.mask)
            painter.setPen(QPen(Qt.red, self.pen_size, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def load_image_handler(self):
        self.image_location = QFileDialog.getOpenFileNames()[0][0]
        self.image = QPixmap(self.image_location)
        self.mask = QBitmap(self.image.width(), self.image.height())
        self.mask.clear()

        # resize
        if self.image.width() > 50 and self.image.height() > 50:
            self.resize(self.image.width(), self.image.height())

        self.update()

    def save_image_handler(self):
        save_name = QFileDialog.getSaveFileName()[0]
        self.image.save(save_name, 'png')

    def resize_pen_handler(self):
        self.pen_size = QInputDialog.getInt(self, 'Set Pen Size', 'Size', 8, 0, 100, 1)[0]

    def remove_watermarks_handler(self):
        # convert mask
        mask = self.mask.toImage()
        mask.invertPixels()

        # save mask
        image_base_name = os.path.basename(self.image_location)
        split_name = image_base_name.split('.')
        mask_filename = str('.'.join(split_name[:-1])) + '_mask.' + str(split_name[-1])
        mask_path = os.path.join(os.getcwd(), 'Input', 'Inpainting', mask_filename)
        mask.save(mask_path, 'png')

        # in-painting
        args = [
            '--input_name', str(image_base_name),
            '--inpainting_start_scale', '3',
            '--input_dir', str(os.path.dirname(self.image_location)),
            '--not_cuda'
        ]

        inpainting.main(args)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainMenu = Menu()
    sys.exit(app.exec_())
