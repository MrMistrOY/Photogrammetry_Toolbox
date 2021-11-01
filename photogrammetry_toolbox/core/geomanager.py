import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from photogrammetry_toolbox.tools.read_midf import ParserMidf


class GeoManager:
    def __init__(self):
        self.focus = 0
        self.principal_point = {'x': 0, 'y': 0}
        self.pixel_size = {'x': 0, 'y': 0}
        self.sensor_dims = {'x': 0, 'y': 0}
        self.external_orient = None
        self.list_object = None

    @property
    def get_images(self):
        return self.external_orient['NAME'].values

    def add_ext_orient_cam(self, external_orient):
        """
        Добавление параметров внешнего ориентирования камеры
        Формат файла: NAME,X,Y,Z,OMEGA_DEG,PHI_DEG,KAPPA_DEG
        :param external_orient: Путь до файла
        :return: None
        """
        self.external_orient = pd.read_csv(external_orient)

    def add_desc_cam(self, description_cam):
        """
        Добавление характеристик регистрирующего аппарата
        Формат файла: xml
        :param description_cam: Путь до файла
        :return: None
        """
        tree = ET.parse(description_cam)
        root = tree.getroot()
        self.focus = float(root.find('d').get('v'))
        x = float(root.findall('x')[1].findall('d')[0].get('v'))
        y = float(root.findall('x')[1].findall('d')[1].get('v'))
        self.principal_point = {'x': x, 'y': y}
        x = float(root.findall('x')[2].findall('x')[0].findall('d')[0].get('v'))
        y = float(root.findall('x')[2].findall('x')[0].findall('d')[0].get('v'))
        self.pixel_size = {'x': x, 'y': y}
        x = float(root.findall('x')[2].findall('x')[1].findall('i')[0].get('v'))
        y = float(root.findall('x')[2].findall('x')[1].findall('i')[1].get('v'))
        self.sensor_dims = {'x': x, 'y': y}

    def add_vector(self, filename, format='midf'):
        """
        Добавление информационного вектора
        :param filename: Путь до файла с объектами
        :param format: Формат данных в файле
                midf - формат для чтения из файлов mid|mif
        :return:
        """
        if format == 'midf':
            vector = ParserMidf(filename).get_data()
        else:
            vector = None

        return vector

    def get_external_orient(self, filename):
        assert self.external_orient is not None, 'Не заданы параметры внешнего ориентирования снимков, ' \
                                                 'функция add_ext_orient_cam'

        df_img = self.external_orient.loc[self.external_orient['NAME'].isin([filename])].iloc[0]

        omga = np.deg2rad(df_img['OMEGA_DEG'])
        alpha = np.deg2rad(df_img['PHI_DEG'])
        kappa = np.deg2rad(df_img['KAPPA_DEG'])

        a11 = np.cos(alpha) * np.cos(kappa)
        a12 = -np.cos(alpha) * np.sin(kappa)
        a13 = np.sin(alpha)
        a21 = np.sin(omga) * np.sin(alpha) * np.cos(kappa) + np.cos(omga) * np.sin(kappa)
        a22 = -np.sin(omga) * np.sin(alpha) * np.sin(kappa) + np.cos(omga) * np.cos(kappa)
        a23 = -np.sin(omga) * np.cos(alpha)
        a31 = -np.cos(omga) * np.sin(alpha) * np.cos(kappa) + np.sin(omga) * np.sin(kappa)
        a32 = np.cos(omga) * np.sin(alpha) * np.sin(kappa) + np.sin(omga) * np.cos(kappa)
        a33 = np.cos(omga) * np.cos(alpha)

        A = np.array([[a11, a12, a13],
                      [a21, a22, a23],
                      [a31, a32, a33]])

        Xs = df_img['Y']
        Ys = df_img['X']
        Zs = df_img['Z']
        Rs = np.array([Xs, Ys, Zs]).reshape(-1, 1)

        return A, Rs

    def geo2pixel(self, geo_point, filename):
        """
        Метод связи координат соответственных точек снимка и местности (уравнениями коллинеарности)
        :param geo_point: Точка в формате [[X1, X2, ...], [Y1, Y2, ...], [Z1, Z2, ...]]
        :param filename: Название снимка
        :return:
        """
        A, Rs = self.get_external_orient(filename)
        r = A.T @ (geo_point - Rs)
        x_pixel = self.sensor_dims['x'] // 2 - (self.focus * (r[0, :] / r[2, :]) -
                                                self.principal_point['x']) / self.pixel_size['y']

        y_pixel = self.sensor_dims['y'] // 2 + (self.focus * (r[1, :] / r[2, :]) -
                                                self.principal_point['y']) / self.pixel_size['y']
        return x_pixel, y_pixel

    def pixel2geo(self, local_point1, filename1, local_point2, filename2):
        """
        Метод связи координат точек местности и их изображений на стереопаре снимков (прямая фотограмметрическая засечка)
        :param local_point1: Пиксельные координаты объекта на первом изображении
        :param filename1: Название первого снимка
        :param local_point2: Пиксельные координаты объекта на втором изображении
        :param filename2: Название второго снимка
        :return:
        """
        p1 = np.array([(local_point1[0] - self.sensor_dims['x'] // 2) * self.pixel_size['x'] + self.principal_point['x'],
                       (self.sensor_dims['y'] // 2 - local_point1[1]) * self.pixel_size['y'] + self.principal_point['y'],
                       -self.focus]).reshape(-1, 1)
        A1, Rs1 = self.get_external_orient(filename1)
        r1 = A1 @ p1

        p2 = np.array([(local_point2[0] - self.sensor_dims['x'] // 2) * self.pixel_size['x'] + self.principal_point['x'],
                       (self.sensor_dims['y'] // 2 - local_point2[1]) * self.pixel_size['y'] + self.principal_point['y'],
                       -self.focus]).reshape(-1, 1)
        A2, Rs2 = self.get_external_orient(filename2)
        r2 = A2 @ p2

        B = Rs2 - Rs1

        N = np.sqrt(
            (B[1] * r2[2] - B[2] * r2[1]) ** 2 + (B[0] * r2[2] - B[2] * r2[0]) ** 2 + (
                    B[0] * r2[1] - B[1] * r2[0]) ** 2) / \
            np.sqrt((r1[1] * r2[2] - r2[1] * r1[2]) ** 2 + (r1[0] * r2[2] - r2[0] * r1[2]) ** 2 + (
                    r1[0] * r2[1] - r2[0] * r1[1]) ** 2)
        geo_point = Rs1 + N * r1
        return geo_point

    def vector_geo2pixel(self, vector):
        pass

    def find_obj_in_image(self, filename, vector):
        pass