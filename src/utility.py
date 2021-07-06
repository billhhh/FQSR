import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from matlab_functions import bgr2ycbcr
import cv2

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                print('{}: {}'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def load(self, trainer):
        verbose = print
        if hasattr(self.args, 'logger'):
            verbose = self.args.logger.info

        if trainer.model.load_from is None:
            return

        try:
            if 'optimizer' in trainer.model.load_from:
                trainer.optimizer.load_state_dict(trainer.model.load_from['optimizer'])
                verbose("resume optimizer from disk")
        except BaseException:
            print("When resuming optimizer, an error happened...")

        try:
            if 'scheduler' in trainer.model.load_from:
                trainer.optimizer.scheduler.load_state_dict(trainer.model.load_from['scheduler'])
                verbose("resume scheduler from disk")
        except BaseException:
            print("When resuming scheduler, an error happened...")

    def save(self, trainer, epoch, is_best=False):
        state_dict = {
                'state_dict': trainer.model.model.state_dict(),
                'optimizer' : trainer.optimizer.state_dict(),
                'scheduler' : trainer.optimizer.scheduler.state_dict(),
                }
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best, state_dict=state_dict)
        trainer.loss.save(self.dir)

        try:
            trainer.loss.plot_loss(self.dir, epoch)
            self.plot_psnr(epoch)
        except BaseException:
            print("When plotting logs, an error happened...")

        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def calc_ssim(img1,
              img2,
              scale,
              input_order='CHW',
              test_y_channel=False,
              dataset=None):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if dataset and dataset.dataset.benchmark:
        crop_border = scale
    else:
        crop_border = scale + 6

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float64)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def make_optimizer(args, target, trainset_length=0):
    '''
        make optimizer and scheduler together
    '''
    verbose = print
    if hasattr(args, 'logger'):
        verbose = args.logger.info

    # optimizer
    #trainable = filter(lambda x: x.requires_grad, target.parameters())
    params_dict = dict(target.named_parameters())
    params = []
    for key, value in params_dict.items():
        shape = value.shape
        custom_hyper = dict()
        custom_hyper['params'] = value
        if value.requires_grad == False:
            continue

        found = False
        for i in args.custom_decay_list:
            if i in key and len(i) > 0:
              found = True
              break
        if found:
            custom_hyper['weight_decay'] = args.custom_decay

        found = False
        for i in args.custom_lr_list:
            if i in key and len(i) > 0:
              found = True
              break
        if found:
            custom_hyper['lr'] = args.custom_lr
           
        params += [custom_hyper]

        verbose("{}, decay {}, lr {}, constant {}".
                format(key, custom_hyper.get('weight_decay', "default"), custom_hyper.get('lr', "default"), custom_hyper.get('lr_constant', "No") ))
    trainable = params
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
        if hasattr(args, 'nesterov'):
            kwargs_optimizer['nesterov'] = args.nesterov
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    class CustomOptimizer(optimizer_class):
        def __init__(self, param, trainset_length, **kwargs):
            super(CustomOptimizer, self).__init__(param, **kwargs)
            self.trainset_length = trainset_length
            #self.args = args

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        #def save(self, save_dir):
        #    torch.save(self.state_dict(), self.get_dir(save_dir))

        #def load(self, load_dir, epoch=1):
        #    self.load_state_dict(torch.load(self.get_dir(load_dir)))
        #    #if epoch > 1:
        #    #    for _ in range(epoch): self.schedule()

        #def get_dir(self, dir_path):
        #    return os.path.join(dir_path, 'optimizer.pt')

        def iteration(self):
            if isinstance(self.scheduler, lrs.CosineAnnealingLR):
                self.scheduler.step()

        def schedule(self):
            if isinstance(self.scheduler, lrs.MultiStepLR):
                self.scheduler.step()

        def get_lr(self):
            lr_list = self.scheduler.get_lr()
            if isinstance(lr_list, list):
                return lr_list[0]
            else:
                return None

        def get_last_epoch(self):
            if isinstance(self.scheduler, lrs.MultiStepLR):
                return self.scheduler.last_epoch
            if isinstance(self.scheduler, lrs.CosineAnnealingLR):
                return self.scheduler.last_epoch // self.trainset_length

    
    optimizer = CustomOptimizer(trainable, trainset_length, **kwargs_optimizer)

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR
    if hasattr(args, 'lr_policy') and 'sgdr' in args.lr_policy:
        scheduler_class = lrs.CosineAnnealingLR
        kwargs_scheduler = {"T_max": trainset_length * args.epochs, "eta_min": args.eta_min } #, 'last_epoch': }
        verbose("CosineAnnealingLR with Tmax = {} * {} and eta_min: {}".format(trainset_length, args.epochs, args.eta_min))

    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


def draw_quant_hist(x, quant, name):
    return

    if not quant.enable:
        return

    if ('before' in name and quant.iteration % 20000 == 998) or ('after' in name and quant.iteration % 20000 == 999):
        dir_path = '../experiment/' + quant.args.save + '/hist/iter' + str(int(quant.iteration.item()))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        draw_hist(x, dir_path + '/index' + str(quant.index) + '_' + name)

        # save x
        torch.save(x, dir_path + '/index' + str(quant.index) + '_' + name + '.pt')


def draw_hist(x, path, clear=True):

    if clear:
        plt.clf()

    if 'before' in path:
        plt.hist(x.cpu().detach().numpy().flatten(), bins='auto')
    elif 'after' in path:
        plt.hist(x.cpu().detach().numpy().flatten(), bins=16)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.show()
    plt.savefig(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
