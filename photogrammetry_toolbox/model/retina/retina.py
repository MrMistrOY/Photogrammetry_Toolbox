import os
import warnings
import collections
import numpy as np

from osgeo import gdal
from datetime import datetime

import torch
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms
from torchvision.ops.boxes import nms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

from photogrammetry_toolbox.model.retina.model import RetinaNet
from photogrammetry_toolbox.model.dataloader import CSVDataset, BigImage, collater, collater_pred, \
    Augmenter, ToTensor, Normalizer

warnings.filterwarnings('ignore')


class Retina:
    def __init__(self, params):
        self.params = params

        if not os.path.exists(os.path.join(self.params.get('root_dir'), 'ckpt')):
            os.makedirs(os.path.join(self.params.get('root_dir'), 'ckpt'), exist_ok=True)

        self.model = None
        self.build_model()
        if self.params.get('write_log'):
            self.writer = SummaryWriter(log_dir=os.path.join(self.params.get('root_dir'),
                                                             'runs',
                                                             datetime.now().strftime('%Y-%m-%d %H-%M')),
                                        flush_secs=30, comment=self.params.get('backbone'))
        else:
            self.writer = None

    def build_model(self):
        self.model = RetinaNet(backbone=self.params.get('backbone'),
                               pretrained_backbone=True,
                               config={'num_classes': self.params.get('num_classes')})

        if self.params.get('use_gpu'):
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.params['use_gpu'] = False

        if self.params.get('use_multi_gpu'):
            self.model = torch.nn.DataParallel(self.model)
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()

    def train(self):
        self.model.training = True

        dataset_train = CSVDataset(self.params,
                                   transform=transforms.Compose([Augmenter(), Normalizer(), ToTensor()]))

        sampler = BatchSampler(RandomSampler(dataset_train), batch_size=self.params.get('batch_size'), drop_last=False)
        dataloader_loader = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)

        optimizer = optim.Adam(self.model.parameters(), lr=self.params.get('lr'))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        it = 0
        iter_warmup = 0
        iter_max = self.params.get('train_epoch')
        progress_bar = tqdm(initial=it, total=iter_max, desc='training stage')
        loss_hist = collections.deque(maxlen=500)

        while it < iter_max:
            self.model.train()
            if self.params.get('use_multi_gpu'):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
            iter_loss = []
            for data in dataloader_loader:
                progress_bar.update()
                optimizer.zero_grad()
                images = data['img']
                annotations = data['annot']
                # images -= torch.tensor([[[[0.4959]], [[0.4809]], [[0.4650]]]])
                # images /= torch.tensor([[[[0.1654]], [[0.1516]], [[0.1267]]]])

                if torch.cuda.is_available() and self.params.get('use_gpu'):
                    images = images.cuda()
                    annotations = annotations.cuda()

                cls_loss, reg_loss = self.model([images, annotations])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()

                loss = cls_loss + reg_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                loss_hist.append(float(loss))
                iter_loss.append(float(loss))

                progress_bar.postfix = "metric: {value:.6f}".format(value=np.mean(loss_hist))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

                optimizer.step()

                it += 1
                # scheduler.step(loss)
                if self.params.get('write_log') and it > iter_warmup:
                    self.writer.add_scalar(tag='losses/loss_cls',
                                           scalar_value=cls_loss.item(),
                                           global_step=it)
                    self.writer.add_scalar(tag='losses/loss_reg',
                                           scalar_value=reg_loss.item(),
                                           global_step=it)

                    self.writer.add_scalar(tag='params/lr',
                                           scalar_value=optimizer.param_groups[0]['lr'],
                                           global_step=it)

                if (it % self.params.get('ckpt_interval')) == 0:
                    scheduler.step(np.mean(iter_loss))
                    ckpt_filename = os.path.join(self.params.get('root_dir'),
                                                 'ckpt',
                                                 "retinanet_{backbone}_{it:08}.ckpt".format(
                                                     backbone=self.params.get('backbone'),
                                                     it=it))
                    if isinstance(self.model, torch.nn.DataParallel):
                        torch.save(self.model.module.state_dict(), ckpt_filename)
                    elif isinstance(self.model, RetinaNet):
                        torch.save(self.model.state_dict(), ckpt_filename)

        if self.params.get('write_log'):
            self.writer.close()

        ckpt_filename = os.path.join(self.params.get('root_dir'),
                                     'ckpt',
                                     "retinanet_{backbone}_final.ckpt".format(
                                         backbone=self.params.get('backbone'), it=it))
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), ckpt_filename)
        elif isinstance(self.model, RetinaNet):
            torch.save(self.model.state_dict(), ckpt_filename)

    def predict(self, image):
        if isinstance(image, gdal.Open()):
            image = image.ReadRaster()
        else:
            image = gdal.Open(image).ReadRaster()

        image = image.transpose([1, 2, 0])
        image = image.astype(np.float32) / 255
        image = 2 * image - 1

        if torch.cuda.is_available() and self.params.get('use_gpu'):
            image = image.cuda()

        self.model.eval()
        with torch.no_grad():
            p_scores, p_classid, p_boxes = self.model(image)
            scores_result = p_scores.detach().cpu()
            id_result = p_classid.detach().cpu()
            boxes_result = p_boxes.detach().cpu()
            keep = nms(boxes_result, scores_result, iou_threshold=0.3)
            scores_result = scores_result[keep]
            boxes_result = boxes_result[keep]
            id_result = id_result[keep]

            ind = scores_result > self.params.get('confidence_thresh')
            scores_result = scores_result[ind].numpy()
            id_result = id_result[ind].numpy()
            boxes_result = boxes_result[ind].numpy()

            return boxes_result, id_result, scores_result

    def predict_big_image(self, image):
        if not isinstance(image, gdal.Open()):
            image = gdal.Open(image)

        self.model.eval()
        with torch.no_grad():
            scores_result = torch.tensor([]).float().cuda()
            id_result = torch.tensor([]).long().cuda()
            boxes_result = torch.tensor([]).cuda()

            big_image = BigImage(image, patch_shape=self.params.get('patch_shape'),
                                 dx=self.params.get('patch_shift')[0],
                                 dy=self.params.get('patch_shift')[1])

            sampler = BatchSampler(SequentialSampler(big_image),
                                   batch_size=self.params.get('batch_size'),
                                   drop_last=False)
            dataloader_loader = DataLoader(big_image, num_workers=0, collate_fn=collater_pred, batch_sampler=sampler)

            for data in dataloader_loader:
                lcs, images = data

                if torch.cuda.is_available() and self.params.get('use_gpu'):
                    images = images.cuda()

                p_scores, p_classid, p_boxes = self.model(images)
                for i in range(self.params.get('batch_size')):
                    xmin, ymin = lcs[i]
                    ind = p_scores[i] != -1
                    scores = p_scores[i, ind]
                    classid = p_classid[i, ind]
                    boxes = p_boxes[i, ind] + torch.tensor([xmin, ymin, xmin, ymin]).cuda()
                    scores_result = torch.cat([scores_result, scores])
                    id_result = torch.cat([id_result, classid])
                    boxes_result = torch.cat([boxes_result, boxes])

            scores_result = scores_result.detach().cpu()
            id_result = id_result.detach().cpu()
            boxes_result = boxes_result.detach().cpu()
            keep = nms(boxes_result, scores_result, iou_threshold=0.3)
            scores_result = scores_result[keep]
            boxes_result = boxes_result[keep]
            id_result = id_result[keep]

            ind = scores_result > self.params.get('confidence_thresh')
            scores_result = scores_result[ind].numpy()
            id_result = id_result[ind].numpy()
            boxes_result = boxes_result[ind].numpy()

            return boxes_result, id_result, scores_result