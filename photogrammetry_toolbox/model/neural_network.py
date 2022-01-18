import os

import photogrammetry_toolbox as pt

from photogrammetry_toolbox.model.retina.retina import Retina


def train(manager: pt.GeoManager, params: dict):
    assert params.get('model'), 'Не задана модель нейронной сети'
    if not os.path.exists(params.get('root_dir')):
        os.makedirs(params.get('root_dir'), exist_ok=True)

    if not params.get('classes'):
        params['classes'] = {key: i for i, key in enumerate(manager.list_vector.keys())}

    if not params.get('num_classes'):
        params['num_classes'] = len(manager.list_vector)

    if not params.get('annotation_file') or not os.path.exists(os.path.join(params.get('root_dir'),
                                                                            params.get('annotation_file'))):
        manager.point2txt(filenames=os.path.join(params.get('root_dir'), params.get('annotation_file')),
                          labels=params['classes'],
                          patch_shape=params.get('patch_shape'),
                          dx=params.get('patch_shift')[0],
                          dy=params.get('patch_shift')[1],
                          type_obj='point')

    if params.get('model') == 'retina':
        model = Retina(params)
    else:
        model = None
        raise ValueError('Не корректно указана модель нейронной сети')

    model.train()


def predict(images, params, manager=None):
    # TODO: обобщить для батча
    params['batch_size'] = 1

    if not params.get('classes'):
        params['classes'] = {key: i for i, key in enumerate(manager.list_vector.keys())}

    if params.get('model') == 'retina':
        model = Retina(params)
    else:
        model = None
        raise ValueError('Не корректно указана модель нейронной сети')

    if isinstance(images, str):
        images = [images]

    for image in images:
        if not os.path.exists(image):
            if manager is None:
                raise ValueError('Не корректный путь до изображения')
            else:
                list_images = [image['name'] for image in manager.list_images]
                image = list_images[image]
        if params.get('sliding_window'):
            result = model.predict_big_image(image)
        else:
            result = model.predict(image)
