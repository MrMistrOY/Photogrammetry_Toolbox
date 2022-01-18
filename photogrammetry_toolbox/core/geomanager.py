import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm

from osgeo import gdal

from photogrammetry_toolbox.tools.imaging import create_grid, init_feature, filter_matches
from photogrammetry_toolbox.tools.parser_midf import ParserMidf, writer
from photogrammetry_toolbox.tools.parser_midf import ParserMidf, writer
from photogrammetry_toolbox.tools.visualization import draw_obj_image


class GeoManager:
    def __init__(self):
        self.focus = 0
        self.principal_point = {'x': 0, 'y': 0}
        self.pixel_size = {'x': 0, 'y': 0}
        self.sensor_dims = {'x': 0, 'y': 0}
        self.external_orient = None
        self.dir_images = None
        self.list_vector = {}
        self.list_images = []
        self.fix_kalininv = False

    @property
    def get_images(self):
        return self.external_orient['NAME'].values

    @property
    def fix_kaliningrad(self):
        self.fix_kalininv = not self.fix_kalininv
        return self.fix_kalininv

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

    def add_vector(self, name, filename, transform_pixel=False, format='midf'):
        """
        Добавление информационного вектора
        :param name: Имя вектора
        :param filename: Путь до файла с объектами
        :param transform_pixel: Преобразование в локальные координаты
        :param format: Формат данных в файле
                midf - формат для чтения из файлов mid|mif
        :return:
        """
        if format == 'midf':
            vector = ParserMidf(filename).get_data()
        else:
            vector = None

        if vector is not None:
            self.list_vector[name] = vector

        if transform_pixel:
            self.transform_vector(name)

    def add_images(self, rood_dir, mask='*.tif'):
        filenames = glob(os.path.join(rood_dir, mask))
        if self.external_orient is not None:
            images = self.get_images
            filenames = list(filter(lambda x: os.path.splitext(os.path.basename(x))[0] in images, filenames))
        self.dir_images = filenames
        self.list_images = [dict(name=os.path.splitext(os.path.basename(filename))[0],
                                 gdal_obj=gdal.Open(filename)) for filename in filenames]

    def transform_vector(self, name):
        vector = np.array([obj['geom'] for obj in self.list_vector[name]]).T

        for filename in tqdm(self.get_images):
            pixel_points = self.geo2pixel(vector, filename)
            idx_x = np.bitwise_and(pixel_points[0] > 0, pixel_points[0] < self.sensor_dims['x'])
            idx_y = np.bitwise_and(pixel_points[1] > 0, pixel_points[1] < self.sensor_dims['y'])
            idx = np.argwhere(np.bitwise_and(idx_x, idx_y))
            if len(idx) != 0:
                for i in idx.reshape(-1):
                    if 'transform' in self.list_vector[name][i]:
                        self.list_vector[name][i]['transform'] += [{'pixel': [round(pixel_points[0][i]),
                                                                              round(pixel_points[1][i])],
                                                                    'image_id': filename}]
                    else:
                        self.list_vector[name][i]['transform'] = [{'pixel': [round(pixel_points[0][i]),
                                                                             round(pixel_points[1][i])],
                                                                   'image_id': filename}]

    @staticmethod
    def __filter_transform(x, filename):
        item = list(filter(lambda y: y['image_id'] == filename, x['transform']))
        if item:
            return item
        else:
            return False

    def get_obj_image(self, filename):
        objects = []
        for vector_name in self.list_vector:
            vector = self.list_vector[vector_name]
            for v in vector:
                item = self.__filter_transform(v, filename)
                if item:
                    objects.append({'geom': v['geom'], 'pixel': item[0]['pixel'],
                                    'id_obj': v['id_obj'], 'type_obj': v['type_obj']})
        return objects

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

        a = -1 if self.fix_kalininv else 1
        x_pixel = self.sensor_dims['x'] // 2 - (self.focus * (r[0, :] / r[2, :]) -
                                                self.principal_point['x']) / self.pixel_size['y'] * a

        y_pixel = self.sensor_dims['y'] // 2 + (self.focus * (r[1, :] / r[2, :]) -
                                                self.principal_point['y']) / self.pixel_size['y'] * a
        return x_pixel, y_pixel

    def pixel2geo(self, local_point1, filename1, local_point2, filename2):
        """
        Метод связи координат точек местности и их изображений на стереопаре снимков
        (прямая фотограмметрическая засечка)
        :param local_point1: Пиксельные координаты объекта на первом изображении
        :param filename1: Название первого снимка
        :param local_point2: Пиксельные координаты объекта на втором изображении
        :param filename2: Название второго снимка
        :return:
        """
        p1 = np.array([(local_point1[0] - self.sensor_dims['x'] // 2) * self.pixel_size['x'] +
                       self.principal_point['x'],
                       (self.sensor_dims['y'] // 2 - local_point1[1]) * self.pixel_size['y'] +
                       self.principal_point['y'],
                       -self.focus]).reshape(-1, 1)
        A1, Rs1 = self.get_external_orient(filename1)
        r1 = A1 @ p1

        p2 = np.array([(local_point2[0] - self.sensor_dims['x'] // 2) * self.pixel_size['x'] +
                       self.principal_point['x'],
                       (self.sensor_dims['y'] // 2 - local_point2[1]) * self.pixel_size['y'] +
                       self.principal_point['y'],
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

    def point2txt(self, filenames, labels, patch_shape, dx, dy, type_obj=None):
        xmins, ymins, xmaxs, ymaxs = create_grid(im_shape=(self.sensor_dims['x'], self.sensor_dims['y']),
                                                 patch_shape=patch_shape, dx=dx, dy=dy)
        file = open(filenames, 'w', encoding='utf-8')
        # for id_image in tqdm(self.get_images):
        for id_image in tqdm(self.dir_images):
            for vname in self.list_vector:
                name = os.path.splitext(os.path.basename(id_image))[0]
                objects = self.get_obj_image(name)
                if objects:
                    if type_obj is not None:
                        objects = list(filter(lambda i: i['type_obj'] == type_obj, objects))
                        if len(objects) == 0:
                            continue
                    x = np.array([item['pixel'][0] for item in objects])
                    y = np.array([item['pixel'][1] for item in objects])
                    for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
                        idx_x = np.bitwise_and(x > xmin, x < xmax)
                        idx_y = np.bitwise_and(y > ymin, y < ymax)
                        idx = np.bitwise_and(idx_x, idx_y)
                        if np.any(idx):
                            x_ = x[idx] - xmin
                            y_ = y[idx] - ymin

                            line = f"{id_image},{xmin} {ymin} {xmax} {ymax},"
                            line += f"{';'.join([f'{q_x} {q_y} {labels[vname]}' for q_x, q_y in zip(x_, y_)])}\n"

                            file.write(line)

        file.close()

    def calc_mutual_orient(self):
        detector, matcher = init_feature('orb')
        new_list = []
        for image in tqdm(self.list_images):
            kp, desc = detector.detectAndCompute(image['gdal_obj'].ReadAsArray().transpose([1, 2, 0]), None)
            image['kp'] = kp
            image['desc'] = desc
            new_list.append(image)
        self.list_images = new_list

    def export_vector(self, name: str, filename):
        if not self.list_vector.get(name):
            raise ValueError('Нет такого вектора с таким именем')

        vector = self.list_vector.get(name)
        writer(filename, vector)

    def draw_obj_image(self, id_image, color='green'):
        assert self.list_images, 'Снимки не добавлены'
        objects = self.get_obj_image(id_image)

        list_images = [image['name'] for image in self.list_images]
        img = self.list_images[id_image].ReadAsArray()
        img = img.transpose([1, 2, 0])
        draw_obj_image(img, objects, color)

    def annotation_vector(self):
        pass
