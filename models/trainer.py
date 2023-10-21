import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, List, Tuple, Optional, ClassVar
import os
import torch

from models import define_model
from models.losses import define_loss
from models.optimizers import get_optimizer
from models.schedulers import get_scheduler

from misc.logger_tool import Logger, Timer
from misc.metric_tool import ConfuseMatrixMeter, AverageMeter
from datasets.transforms import get_seg_augs, get_cd_augs, get_scd_augs
import utils

# Decide which device we want to run on
# torch.cuda.current_device()
VIS_FLAG = False


class BaseTrainer():
    def __init__(self,
                 logger: Logger = None,
                 dataloaders: Dict = None,
                 model_name: str = 'base_fcn_resnet18',
                 lr: float = 0.01,
                 optim_mode: str = 'sgd',
                 lr_policy: str = 'linear',
                 max_epochs=200,
                 batch_size=8,
                 gpu_ids: Union[List, Tuple] = (0,),
                 checkpoint_dir: str = '',
                 vis_dir: Optional[str] = 'vis',
                 **kwargs):
        self.checkpoint_dir = checkpoint_dir
        self.vis_dir = vis_dir
        self.dataloaders = dataloaders
        self.gpu_ids = gpu_ids
        self.device = torch.device("cuda:%s" % gpu_ids[0]
                                   if torch.cuda.is_available() and len(gpu_ids) > 0
                                   else "cpu")
        print(self.device)
        self.max_num_epochs = max_epochs
        self.batch_size = batch_size
        self.model_name = model_name
        self.lr = lr
        self.lr_policy = lr_policy
        self.optim_mode = optim_mode

        # logger file
        if logger is None:
            logger_path = os.path.join(checkpoint_dir, 'log.txt')
            logger = Logger(logger_path, **kwargs)
            self.__dict__.update(kwargs)
            logger.write_dict_str(self.__dict__)
        self.logger = logger

        # define model
        self.configure_model()

        # define some other vars to record the training states
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        # learner
        self.configure_optimizers()

        # define timer
        self.timer = Timer()

        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0

        if VIS_FLAG:
            os.makedirs(self.vis_dir, exist_ok=True)
        # check and create model dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def configure_model(self):
        # To be defined in the children class
        self.model = None

    def configure_optimizers(self):
        from models.optimizers import get_params_groups
        lr_multi = getattr(self, 'lr_multi', 1)
        params_list = get_params_groups(self.model, lr=self.lr, lr_multi=lr_multi)
        self.optimizer = get_optimizer(params_list, optim_mode=self.optim_mode,
                                       lr=self.lr, lr_policy=self.lr_policy,
                                       init_step=self.global_step, max_step=self.total_steps)
        print('get optimizer %s' % self.optim_mode)
        self.lr_scheduler = get_scheduler(self.optimizer, self.lr_policy,
                                          max_epochs=self.max_num_epochs,
                                          steps_per_epoch=self.steps_per_epoch)

    def load_pretrain(self, model_pretrained_path: str):
        """
        加载全模型的pretrain(需要从外部显式调用，不在初始化阶段运行)
        :param model_pretrained_path:
        :return:
        """
        if os.path.exists(model_pretrained_path):
            checkpoint = torch.load(os.path.join(model_pretrained_path))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.write(f'load model_pretrain_path: {model_pretrained_path} \n')
        else:
            self.logger.write(f'no model_pretrain_path: {model_pretrained_path} \n')

    def _load_checkpoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'), map_location='cpu')

            # update model states
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])

            self.model.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        # print(est)
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        # print(imps)
        return imps, est

    def _save_checkpoint(self, ckpt_name='last_ckpt.pt'):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_checkpoints(self):
        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
                          % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')


def my_test():
    logger = Logger('.1.txt')
    BaseTrainer(logger=logger)


class SEGTrainer(BaseTrainer):
    """有待修改，里面的优化器选择，等等"""

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        self.kwargs = kwargs
        self.loss_name = kwargs['loss_name']
        self.n_class = kwargs['n_class']
        self.img_size = kwargs['img_size']
        self.pretrained = kwargs['pretrained']
        self.with_dataset_aug = kwargs.get('with_dataset_aug', False)
        super().__init__(*args, **kwargs)
        self.G_loss = None
        self.configure_metric()
        self.loss_dict = dict()

    def configure_metric(self):
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class if self.n_class!=1 else 2)
        self.log_score_name = 'mf1'
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))
        self.logger.define_metric(metric_name='epoch_val_acc', summary='max')
        self.logger.define_metric(metric_name='epoch_train_acc', summary='max')

    def configure_model(self):
        self.model = define_model(model_name=self.model_name, gpu_ids=self.gpu_ids,
                                  pretrained=self.pretrained, n_class=self.n_class,
                                  head_pretrained=self.kwargs.get('head_pretrained', False),
                  frozen_backbone_weights=self.kwargs.get('frozen_backbone_weights', False))
        # define the loss functions
        self.loss_func = define_loss(loss_name=self.loss_name)
        if self.with_dataset_aug is False:
            self.augs = get_seg_augs(imgz_size=self.img_size)

    def _update_metric(self):
        target = self.batch['mask'].detach()
        if isinstance(self.G_pred, tuple):
            # G_pred = tuple([item.detach() for item in self.G_pred])
            G_pred = self.G_pred[0].detach()
        else:
            G_pred = self.G_pred.detach()
        if self.n_class == 1:
            if self.loss_name == 'bcl':
                G_pred = (G_pred > 1).long()
            else:
                G_pred = torch.sigmoid(G_pred)
                G_pred = (G_pred > 0.5).long()
        else:
            G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                      gt=target.cpu().numpy())
        return current_score

    def _clear_cache(self):
        self.running_metric.clear()

    def _update_lr_schedulers(self, state='epoch'):
        if (self.lr_policy == 'poly' and state == 'step') or \
                (self.lr_policy != 'poly' and state == 'epoch'):
            self.lr_scheduler.step()

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        # TODO:
        # from misc.torchutils import tensor_mask_colorize
        pred_vis = pred.float() / (self.n_class - 1)
        return pred_vis

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])
        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f, ' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                       imps * self.batch_size, est,
                       self.G_loss.item(), running_acc)
            for k, v in self.loss_dict.items():
                message += f'{k}: {v.item():.5f}, '
            message += '\n'
            self.logger.write(message)
        if VIS_FLAG:
            if np.mod(self.batch_id, 500) == 1:
                self.save_vis_batch_data()

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

    def _logger_write_dict(self, scores: dict):
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

    def get_meter_scores(self):
        return self.running_metric.get_scores()

    def _collect_epoch_states(self):
        scores = self.get_meter_scores()
        self.epoch_acc = scores[self.log_score_name]
        self.logger.write(f'Is_training: {self.is_training}. '
                          f'Epoch {self.epoch_id} / {self.max_num_epochs - 1},'
                          f' epoch_{self.log_score_name}= {self.epoch_acc:.5f}\n')
        # self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
        #       (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        self._logger_write_dict(scores)

    def _forward_pass(self, batch):
        self.batch = batch
        self.G_pred = self.model(batch['A'])

    def _backward_G(self):
        # gt = self.batch['mask'].long()
        # self.G_loss = self.loss_func(self.G_pred, gt)
        # self.G_loss.backward()
        gt = self.batch['mask'].long()
        self.G_loss = self.loss_func(self.G_pred, gt)
        if bool(self.loss_dict):
            self.loss_dict['loss_overall'] = 0
            loss_overall = self.G_loss
            for k, v in self.loss_dict.items():
                loss_overall = loss_overall + v  # 若改为+=的形式，self.G_loss的值也会改变；
            loss_overall.backward()
            self.loss_dict['loss_overall'] = loss_overall
            # self.loss_dict['loss_overall'].backward()
        else:
            self.G_loss.backward()

    def transfer_batch_to_device(self, batch: Dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(self.device)

    def on_after_batch_transfer(self, batch: Dict) -> Dict:
        A, m = batch['A'], batch['mask']
        if self.with_dataset_aug is False:
            batch['A'], batch['mask'] = self.augs(A, m)
        return batch

    def train_models(self, do_eval=True):
        self._load_checkpoint()
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.model.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self.transfer_batch_to_device(batch)
                batch = self.on_after_batch_transfer(batch)
                self._forward_pass(batch)
                # update G
                self.optimizer.zero_grad()
                self._backward_G()
                self.optimizer.step()
                self._update_lr_schedulers(state='step')
                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers(state='epoch')

            if not do_eval:
                continue
            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.model.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self.transfer_batch_to_device(batch)
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)
        # items = {}
        #  一个epoch中的train和val想信息，统一在这里更新到log里面
        self.items['epoch_val_acc'] = self.epoch_acc
        self.logger.log(self.items)

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)
        self.items = {}
        self.items['epoch_id'] = self.epoch_id
        self.items['epoch_train_acc'] = self.epoch_acc
        # self.logger.log(self.items, step=self.epoch_id)


class CDTrainer(SEGTrainer):
    def __init__(self,
                 *args,
                 **kwargs):
        super(CDTrainer, self).__init__(*args, **kwargs)

    def configure_model(self):
        self.model = define_model(model_name=self.model_name, gpu_ids=self.gpu_ids,
                                  pretrained=self.pretrained, n_class=self.n_class,
                                  head_pretrained=self.kwargs.get('head_pretrained', False),
                  frozen_backbone_weights=self.kwargs.get('frozen_backbone_weights', False),
                                  scale_ratios=self.kwargs.get('scale_ratios', 1))
        # define the loss functions
        self.loss_func = define_loss(loss_name=self.loss_name)
        self.augs = get_cd_augs(imgz_size=self.img_size,
                                color_jet=self.kwargs.get('color_jet', False))

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
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

    def on_after_batch_transfer(self, batch: Dict) -> Dict:
        A, B, m = batch['A'], batch['B'], batch['mask']
        if self.with_dataset_aug is False:
            batch['A'], batch['B'], batch['mask'] = self.augs(A, B, m)
        return batch

    def _forward_pass(self, batch):
        self.batch = batch
        # self.G_pred = self.model(batch['A'], batch['B'])
        out = self.model(batch['A'], batch['B'], is_train=True)
        if isinstance(out, dict):
            for k, v in out.items():
                if k == 'cd_pred':
                    self.G_pred = out['cd_pred']
                elif 'loss' in k:
                    self.loss_dict.update(k=v)
        elif isinstance(out, tuple):
            self.G_pred = out
        else:
            assert isinstance(out, torch.Tensor)
            self.G_pred = out


class CDTrainer_SR(CDTrainer):
    """CD with multitask, for example, apply cross-resolution feat consistence with
    point-level supervision
    """
    def __init__(self,
                 with_origin_scale_ret=0,
                 with_sample=None,
                 *args,
                 **kwargs):
        self.G_preds = None
        super(CDTrainer_SR, self).__init__(*args, **kwargs)
        self.with_origin_scale_ret = with_origin_scale_ret
        self.with_sample = with_sample
        self.info = dict()
        if self.with_origin_scale_ret == 1:
            self.time_hr = 'A'
            self.info.update(time_hr=self.time_hr)
        elif self.with_origin_scale_ret == 2:
            self.time_hr = 'B'
            self.info.update(time_hr=self.time_hr)
        

    def on_after_batch_transfer(self, batch: Dict) -> Dict:
        A, B, m = batch['A'], batch['B'], batch['mask']
        if self.with_dataset_aug is False:
            batch['A'], batch['B'], batch['mask'] = self.augs(A, B, m,
                                                              data_keys=('input','input','mask'),)
            if 'hr' in batch.keys():
                B = A.shape[0]
                # name='RandomGaussianBlur_3'
                self.augs._params[-2].data['batch_prob'] = torch.BoolTensor([False, ] * B)
                params = self.augs._params
                batch['hr'] = self.augs(batch['hr'], data_keys=('input',), params=params)
        return batch

    def _forward_pass(self, batch):
        self.batch = batch
        if 'hr' in batch.keys():
            self.info.update({'hr': batch[f"hr"]})
        # self.G_pred = self.model(batch['A'], batch['B'])
        if self.with_sample is not None:
            batch_size = batch['A'].size(0)
            q_sample = self.img_size ** 2 // 4
            sample_lst = torch.stack(
                [torch.from_numpy(np.random.choice(self.img_size ** 2, q_sample, replace=False)) for _
                 in range(batch_size)]).to(batch['A'].device)
            self.info['sample_lst']=sample_lst
        out = self.model(batch['A'], batch['B'], is_train=True, info=self.info)
        if isinstance(out, dict):
            for k, v in out.items():
                if k == 'cd_pred':
                    self.G_pred = out['cd_pred']
                elif 'loss' in k:
                    self.loss_dict.update(dict(loss_sr=v))
        else:
            assert isinstance(out, torch.Tensor)
            self.G_pred = out

    def _backward_G(self):
        # gt = self.batch['mask'].long()
        # self.G_loss = self.loss_func(self.G_pred, gt)
        # self.G_loss.backward()
        gt = self.batch['mask'].long()
        if self.with_sample is not None:
            sample_lst = self.info['sample_lst']
            from einops import rearrange
            gt = rearrange(gt, 'b c h w -> b (h w) c')
            gt = torch.gather(gt, 1, sample_lst.unsqueeze(2).repeat(1,1,gt.size(2))).contiguous()
            gt = rearrange(gt, ' b n c -> (b n) c')
            gt = gt.squeeze(-1)
            self.gt = gt
            self.G_pred = rearrange(self.G_pred, ' b n c -> (b n) c')

        self.G_loss = self.loss_func(self.G_pred, gt)
        if bool(self.loss_dict):
            self.loss_dict['loss_overall'] = 0
            loss_overall = self.G_loss
            for k, v in self.loss_dict.items():
                loss_overall = loss_overall + v  # 若改为+=的形式，self.G_loss的值也会改变；
            loss_overall.backward()
            self.loss_dict['loss_overall'] = loss_overall
            # self.loss_dict['loss_overall'].backward()
        else:
            self.G_loss.backward()

    def _update_metric(self):
        target = self.gt.detach()
        G_pred = self.G_pred.detach()
        if self.n_class == 1:
            G_pred = torch.sigmoid(G_pred)
            G_pred = (G_pred > 0.5).long()
        else:
            G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                      gt=target.cpu().numpy())
        return current_score


class SCDTrainer(CDTrainer):
    def __init__(self,
                 *args,
                 **kwargs):
        self.G_preds = None
        super(SCDTrainer, self).__init__(*args, **kwargs)

    def configure_metric(self):
        self.running_metric_t1 = ConfuseMatrixMeter(n_class=self.n_class)
        self.running_metric_t2 = ConfuseMatrixMeter(n_class=self.n_class)
        self.log_score_name = 'F1_1'
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))
        self.logger.define_metric(metric_name='epoch_val_acc', summary='max')
        self.logger.define_metric(metric_name='epoch_train_acc', summary='max')

    def configure_model(self):
        self.model = define_model(model_name=self.model_name, gpu_ids=self.gpu_ids,
                                  pretrained=self.pretrained, n_class=self.n_class,
                                  head_pretrained=self.kwargs.get('head_pretrained', False),
                  frozen_backbone_weights=self.kwargs.get('frozen_backbone_weights', False))
        # define the loss functions
        self.loss_func = define_loss(loss_name=self.loss_name)
        self.augs = get_scd_augs(imgz_size=self.img_size)

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

    def on_after_batch_transfer(self, batch: Dict) -> Dict:
        A, B, m1, m2 = batch['A'], batch['B'], batch['mask1'], batch['mask2']
        batch['A'], batch['B'], batch['mask1'], batch['mask2'] = self.augs(A, B, m1, m2)
        return batch

    def _forward_pass(self, batch):
        self.batch = batch
        self.G_preds = self.model(batch['A'], batch['B'])

    def _backward_G(self):
        gt = self.batch['mask1'].long()
        gt2 = self.batch['mask2'].long()
        self.G_loss = 1 / 2 * (self.loss_func(self.G_preds[0], gt) +
                               self.loss_func(self.G_preds[1], gt2)
                               )
        self.G_loss.backward()

    def get_meter_scores(self):
        scores_t1 = self.running_metric_t1.get_scores()
        scores_t2 = self.running_metric_t2.get_scores()
        return scores_t1, scores_t2

    def _collect_epoch_states(self):
        scores = self.get_meter_scores()
        self.log_acc_list = [scores[0][self.log_score_name],
                             scores[1][self.log_score_name]]
        self.epoch_acc = (scores[0][self.log_score_name] +
                          scores[1][self.log_score_name]) / 2
        self.logger.write(f'Is_training: {self.is_training}. '
                          f'Epoch {self.epoch_id} / {self.max_num_epochs - 1},'
                          f' epoch_{self.log_score_name}= {self.epoch_acc:.5f}\n')
        self.logger.write('t1: \n')
        self._logger_write_dict(scores[0])
        self.logger.write('t2: \n')
        self._logger_write_dict(scores[1])

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)
        # items = {}
        self.items['epoch_val_acc'] = self.epoch_acc
        self.items['epoch_val_acc_t1'] = self.log_acc_list[0]
        self.items['epoch_val_acc_t2'] = self.log_acc_list[1]
        self.logger.log(self.items)

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)
        self.items = {}
        self.items['epoch_id'] = self.epoch_id
        self.items['epoch_train_acc'] = self.epoch_acc
        self.items['epoch_train_acc_t1'] = self.log_acc_list[0]
        self.items['epoch_train_acc_t2'] = self.log_acc_list[1]
        # self.logger.log(self.items, step=self.epoch_id)


if __name__ == '__main__':
    my_test()