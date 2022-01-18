import os
import csv
import torch
import numpy as np
import skimage.color

from osgeo import gdal
from torch.utils.data import Dataset

from photogrammetry_toolbox.tools.imaging import create_grid


class CSVDataset(Dataset):
    def __init__(self, params, transform=None):

        self.transform = transform

        self.width = params.get('dimensions_object')[0]
        self.height = params.get('dimensions_object')[1]

        self.classes = params.get('classes')

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        with open(os.path.join(params.get('root_dir'), params.get('annotation_file')), 'r') as file:
            self.image_data = self._read_annotations(csv.reader(file, delimiter=','))
        self.image_names = list(self.image_data.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        filename = self.image_names[image_index]
        filename, roi = filename.split('-')
        roi = list(map(lambda x: int(x), roi.split(' ')))
        gdal_img = gdal.Open(filename)
        img = gdal_img.ReadAsArray(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1])
        img = img.transpose([1, 2, 0])
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        if len(annotation_list) == 0:
            return annotations

        for idx, a in enumerate(annotation_list):
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    @staticmethod
    def str_to_list(text):
        return list([int(i) for i in text[1:-1].split(', ')])

    def _read_annotations(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            filename, roi, objects = row
            filename += f'-{roi}'
            objects = objects.split(';')

            for obj in objects:
                if filename not in result:
                    result[filename] = []

                x, y, class_id = list(map(lambda a: int(a), obj.split(' ')))

                x1 = x - self.width // 2
                y1 = y - self.height // 2
                x2 = x + self.width // 2
                y2 = y + self.height // 2

                x1 = np.clip(x1, a_min=0, a_max=512)
                y1 = np.clip(y1, a_min=0, a_max=512)
                x2 = np.clip(x2, a_min=0, a_max=512)
                y2 = np.clip(y2, a_min=0, a_max=512)

                class_name = self.label_to_name(class_id)

                result[filename].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    image_batch = torch.stack(imgs)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    image_batch = image_batch.permute(0, 3, 1, 2)

    return {'img': image_batch, 'annot': annot_padded}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': torch.from_numpy(image.copy()), 'annot': torch.from_numpy(annots)}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': (2 * image.astype(np.float32) - 1), 'annot': annots}


class BigImage(Dataset):
    def __init__(self, gdal_image, patch_shape, dx, dy):

        self._gdal_image = gdal_image
        self._image_size = (self.gdal_image.RasterXSize, self.gdal_image.RasterYSize)
        self._patch_shape = patch_shape

        self.grid = create_grid(im_shape=self._image_size,
                                patch_shape=patch_shape,
                                dx=dx, dy=dy)

    def __getitem__(self, index):
        xmin = self.grid[0][index]
        ymin = self.grid[1][index]
        xmax = self.grid[2][index]
        ymax = self.grid[3][index]

        sample = self.gdal_image.ReadAsArray(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))
        sample = sample.transpose([1, 2, 0])
        sample = sample.astype(np.float32) / 255
        sample = 2 * sample - 1

        return (xmin, ymin), sample

    @property
    def size(self):
        return self._image_size

    def __len__(self):
        return len(self.grid[0])


def collater_pred(patchlist):
    batch = torch.cat([torch.from_numpy(patch[np.newaxis, :].transpose(0, 3, 1, 2)) for lc, patch in patchlist],
                      dim=0).float()
    lcs = [lc for lc, patch in patchlist]
    return lcs, batch
