import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Union, List, Tuple, Optional, ClassVar

import utils
from models import define_model
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from misc.metric_tool import ConfuseMatrixMeter

from misc.logger_tool import Logger, Timer
from misc.torchutils import norm_tensor

from utils import de_norm
from utils import write_dict_to_excel
# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VIS_FLAG = False


class BaseEvaluator():
    def __init__(self,
                 logger: Logger = None,
                 dataloader: DataLoader = None,
                 model_name: str = 'base_fcn_resnet18',
                 batch_size=8,
                 gpu_ids: Union[List, Tuple] = (0,),
                 checkpoint_dir: str = '',
                 vis_dir: Optional[str] = 'vis',
                 pred_dir: Optional[str] = None,
                 **kwargs):
        self.kwargs = kwargs
        self.checkpoint_dir = checkpoint_dir
        self.vis_dir = vis_dir
        self.pred_dir = pred_dir
        if self.pred_dir:
            os.makedirs(self.pred_dir, exist_ok=True)
        self.dataloader = dataloader
        self.gpu_ids = gpu_ids
        self.device = torch.device("cuda:%s" % gpu_ids[0]
                                   if torch.cuda.is_available() and len(gpu_ids) > 0
                                   else "cpu")
        print(self.device)
        self.batch_size = batch_size
        self.model_name = model_name
        # logger file
        if logger is None:
            logger_path = os.path.join(checkpoint_dir, 'log_eval.txt')
            logger = Logger(logger_path)
            self.__dict__.update(kwargs)
            logger.write_dict_str(self.__dict__)
        self.logger = logger

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        if VIS_FLAG:
            os.makedirs(self.vis_dir, exist_ok=True)
        # define model
        self.configure_model()
        self.configure_metric()

        self.n_class = kwargs.get('n_class', 2)
        self.loss_name = kwargs.get('loss_name', 'ce')
        self.eval_samples_num = kwargs.get('eval_samples_num', 10e8)

    def configure_model(self):
        # To be defined in the children class
        self.model = None

    def configure_metric(self):
        # To be defined in the children class
        self.running_metric = None
        self.log_score_name = 'mf1'

    def _forward_pass(self, batch):
        # To be defined in the children class
        return

    def save_vis_batch_data(self):
        # To be defined in the children class
        return

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            # update model states
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # model2_dict = self.model.state_dict()
            # state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
            # model2_dict.update(state_dict)
            # self.model.load_state_dict(model2_dict)

            self.model.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint.get('best_val_acc', 0)
            self.best_epoch_id = checkpoint.get('best_epoch_id', 0)

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def transfer_batch_to_device(self, batch: Dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(self.device)

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)
        if VIS_FLAG:
            if np.mod(self.batch_id, 100) == 1:
               self.save_vis_batch_data()

    def _update_metric(self):
        target = self.batch['mask'].detach().long()
        G_pred = self.G_pred.detach()
        if self.n_class == 1:
            if self.loss_name == 'bcl':
                G_pred = (G_pred > 1).long()

            else:
                G_pred = torch.sigmoid(G_pred)
                G_pred = (G_pred > 0.5).long()
        else:
            G_pred = torch.argmax(G_pred, dim=1)
        # G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _clear_cache(self):
        self.running_metric.clear()

    def _logger_write_dict(self, scores: dict):
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_acc = scores_dict[self.log_score_name]
        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass
        self._logger_write_dict(scores_dict)
        write2xls = self.kwargs.get('write2xls')
        if write2xls is not False:
            print(f'write scores to xls.')
            contents_dict = self.get_content_dict(scores_dict)
            out_xls_name = f'{write2xls}.xls' if isinstance(write2xls, str) else 'config_score.xls'
            out_path = os.path.join(self.checkpoint_dir, out_xls_name)
            write_dict_to_excel(out_path, contents_dict)

    def get_content_dict(self, scores_dict):
        out_keys = ['pretrained',  'data_name', 'split', 'split_val']
        contents_dict = {}
        contents_dict['model_name'] = self.model_name
        for key in out_keys:
            contents_dict[key] = self.kwargs[key]
        out_keys = ['mf1', 'precision_1', 'recall_1', 'F1_1', 'iou_1', 'acc']
        for key in out_keys:
            contents_dict[key] = f'{scores_dict[key]*100:.3f}'
        return contents_dict

    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.model.eval()
        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            if self.batch_id > (self.eval_samples_num // self.batch_size):
                break
            with torch.no_grad():
                self.transfer_batch_to_device(batch)
                self._forward_pass(batch)

            self._collect_running_batch_states()
        self._collect_epoch_states()


class SegEvaluator(BaseEvaluator):
    def __init__(self,
                 *args,
                 **kwargs):
        self.n_class = kwargs.get('n_class', 2)
        super().__init__(*args, **kwargs)

    def configure_metric(self):
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class if self.n_class!=1 else 2)

        self.log_score_name = 'mf1'

    def configure_model(self):
        self.model = define_model(model_name=self.model_name, gpu_ids=self.gpu_ids,
                                  n_class=self.n_class)

    def _visualize_pred(self):
        if self.n_class == 1:
            if self.loss_name == 'bcl':
                pred = (self.G_pred > 1).long()
            else:
                G_pred = torch.sigmoid(self.G_pred)
                pred = (G_pred > 0.5).long()
        else:
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def save_vis_batch_data(self):
        vis_input = utils.make_numpy_grid(utils.de_norm(self.batch['A']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['mask'])
        vis = np.concatenate([vis_input, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
                          str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')
        plt.imsave(file_name, vis)

    def _forward_pass(self, batch):
        self.batch = batch
        self.G_pred = self.model(batch['A'])


class CDEvaluator(BaseEvaluator):
    def __init__(self,
                 *args,
                 **kwargs):
        self.n_class = kwargs.get('n_class', 2)
        super().__init__(*args, **kwargs)

    def configure_metric(self):
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class if self.n_class!=1 else 2)
        self.log_score_name = 'mf1'

    def configure_model(self):
        self.model = define_model(model_name=self.model_name, gpu_ids=self.gpu_ids,
                                  n_class=self.n_class)

    def _visualize_pred(self):
        if self.n_class == 1:
            if self.loss_name == 'bcl':
                pred = (self.G_pred > 1).long()
            else:
                G_pred = torch.sigmoid(self.G_pred)
                pred = (G_pred > 0.5).long()
        else:
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def save_vis_batch_data(self):
        vis_input = utils.make_numpy_grid(utils.de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(utils.de_norm(self.batch['B']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['mask'])
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
                          str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')
        plt.imsave(file_name, vis)

    def _forward_pass(self, batch):
        self.batch = batch
        self.G_pred = self.model(batch['A'], batch['B'])
        if self.pred_dir:
            self._save_predictions()

    def _save_predictions(self):
        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            from misc.imutils import save_image
            save_image(pred, file_name)

    def _get_vis(self, batch_id=0):

        vis = self.model.vis
        vis[0] = list(vis[0])
        vis[1] = list(vis[1])
        mode = 'all'
        self.model.vis = []
        from misc.torchutils import norm_tensor
        for level in [0]:
            if mode == 'mean':
                vis[0][level] = torch.mean(vis[0][level], dim=1, keepdim=True)
                vis[1][level] = torch.mean(vis[1][level], dim=1, keepdim=True)
            elif mode == 'several':
                vis[0][level] = vis[0][level][:, 37:38]
                vis[1][level] = vis[1][level][:, 37:38]
            dif = torch.abs(vis[0][level] - vis[1][level])
            vis[0][level] = norm_tensor(vis[0][level])
            vis[1][level] = norm_tensor(vis[1][level])
            # features = torch.concat([vis[0][level], vis[1][level]], dim=0)
            features = torch.concat([vis[0][level], vis[1][level], dif], dim=0)

            import torch.nn.functional as F
            features = F.interpolate(features, size=[256, 256])
            vis_feature = utils.make_numpy_grid_singledim(features, padding=5, pad_value=0)
         # vis_feature = np.sqrt(vis_feature)
            file_name = os.path.join(
                self.pred_dir, f'feat_level{level}_{mode}_' + self.batch['name'][0]+'.jpg')
            plt.imsave(file_name, vis_feature, cmap='jet')

        images = torch.cat([de_norm(self.batch['A']),
                             de_norm(self.batch['B'])], dim=0)

        vis = utils.make_numpy_grid(images, pad_value=255, padding=5)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.pred_dir, 'img_' + self.batch['name'][0]+'.jpg')
        plt.imsave(file_name, vis)

    def _get_vis2(self):
        def get_norm_vis(vis, mode, level, index=37):
            if mode == 'mean':
                vis[0][level] = torch.mean(vis[0][level], dim=1, keepdim=True)
                vis[1][level] = torch.mean(vis[1][level], dim=1, keepdim=True)
            elif mode == 'several':
                vis[0][level] = vis[0][level][:, index:index+1]
                vis[1][level] = vis[1][level][:, index:index+1]
            return vis
        vis_old, vis = self.model.vis
        mode = 'several'
        levels = [0]
        index = 37
        self.model.vis = []

        for level in levels:
            vis = get_norm_vis(vis, mode, level, index=index)
            dif = torch.abs(vis[0][level] - vis[1][level])
            vis_ = vis
            vis[0][level] = norm_tensor(vis[0][level])
            vis[1][level] = norm_tensor(vis[1][level])
            # features = torch.concat([vis[0][level], vis[1][level]], dim=0)
            features = torch.concat([vis[0][level], vis[1][level], dif], dim=0)
            features = F.interpolate(features, size=[256, 256])
            vis_feature = utils.make_numpy_grid_singledim(features, padding=5, pad_value=0)
         # vis_feature = np.sqrt(vis_feature)
            file_name = os.path.join(
                self.pred_dir, f'{self.batch["name"][0]}_feat_level{level}_{mode}_after.jpg')
            plt.imsave(file_name, vis_feature, cmap='jet')

        for level in levels:
            vis_old = get_norm_vis(vis_old, mode, level, index=index)
            dif_old = torch.abs(vis_old[0][level] - vis_old[1][level])
            vis_old_ = vis_old
            vis_old[0][level] = norm_tensor(vis_old[0][level])
            vis_old[1][level] = norm_tensor(vis_old[1][level])
            # features = torch.concat([vis[0][level], vis[1][level]], dim=0)
            features_old = torch.concat([vis_old[0][level], vis_old[1][level], dif_old], dim=0)
            features_old = F.interpolate(features_old, size=[256, 256])
            vis_feature_old = utils.make_numpy_grid_singledim(features_old, padding=5, pad_value=0)
         # vis_feature = np.sqrt(vis_feature)
            file_name = os.path.join(
                self.pred_dir, f'{self.batch["name"][0]}_feat_level{level}_{mode}_before.jpg')
            plt.imsave(file_name, vis_feature_old, cmap='jet')

            # cal diff
            # diff = torch.abs(vis_ - vis_old_)
            diff = torch.concat([torch.abs(vis_old_[0][level]-vis_[0][level]),
                                 torch.abs(vis_old_[1][level]-vis_[1][level]),
                                 dif_old], dim=0)
            vis_diff = utils.make_numpy_grid_singledim(diff, padding=5, pad_value=0)
            file_name = os.path.join(
                self.pred_dir, f'{self.batch["name"][0]}_diff_{level}_{mode}.jpg')
            plt.imsave(file_name, vis_diff, cmap='jet')
class SCDEvaluator(BaseEvaluator):
    def __init__(self,
                 *args,
                 **kwargs):
        self.n_class = kwargs['n_class']
        super().__init__(*args, **kwargs)

    def configure_metric(self):
        self.running_metric_t1 = ConfuseMatrixMeter(n_class=self.n_class)
        self.running_metric_t2 = ConfuseMatrixMeter(n_class=self.n_class)
        self.log_score_name = 'F1_1'

    def configure_model(self):
        self.model = define_model(model_name=self.model_name, gpu_ids=self.gpu_ids,
                                  n_class=self.n_class)

    def _update_metric(self):
        target = self.batch['mask1'].detach()
        target2 = self.batch['mask2'].detach()
        G_pred = self.G_preds[0].detach()
        G_pred2 = self.G_preds[1].detach()
        G_pred = torch.argmax(G_pred, dim=1)
        G_pred2 = torch.argmax(G_pred2, dim=1)
        current_score_t1 = self.running_metric_t1.update_cm(pr=G_pred.cpu().numpy(),
                                                            gt=target.cpu().numpy())
        current_score_t2 = self.running_metric_t2.update_cm(pr=G_pred2.cpu().numpy(),
                                                            gt=target2.cpu().numpy())
        return (current_score_t1 + current_score_t2) / 2

    def _clear_cache(self):
        self.running_metric_t1.clear()
        self.running_metric_t2.clear()

    def _visualize_preds(self):
        vis_preds = []
        for pred in self.G_preds:
            vis_pred = torch.argmax(pred, dim=1, keepdim=True)
            vis_pred = vis_pred * 255
            vis_preds.append(vis_pred)
        return vis_preds

    def save_vis_batch_data(self):
        vis_input = utils.make_numpy_grid(utils.de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(utils.de_norm(self.batch['B']))
        vis_gt = utils.make_numpy_grid(self.batch['mask1'])
        vis_gt2 = utils.make_numpy_grid(self.batch['mask2'])
        vis_sample = [vis_input, vis_input2, vis_gt, vis_gt2]

        vis_pred = [utils.make_numpy_grid(vis) for vis in self._visualize_preds()]

        vis = np.concatenate(vis_sample + vis_pred, axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
                          str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')
        plt.imsave(file_name, vis)

    def _forward_pass(self, batch):
        self.batch = batch
        self.G_preds = self.model(batch['A'], batch['B'])

    def get_meter_scores(self):
        scores_t1 = self.running_metric_t1.get_scores()
        scores_t2 = self.running_metric_t2.get_scores()
        scores = {}
        scores['t1'] = scores_t1
        scores['t2'] = scores_t2
        return scores

    def _collect_epoch_states(self):
        scores_dict = self.get_meter_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_acc = (scores_dict['t1'][self.log_score_name] +
                          scores_dict['t2'][self.log_score_name]) / 2
        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass
        self._logger_write_dict(scores_dict)




