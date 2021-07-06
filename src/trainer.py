import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        if not self.loader_train:
            loader_train_len = 0
        else:
            loader_train_len = len(self.loader_train)
        self.optimizer = utility.make_optimizer(args, self.model, trainset_length=loader_train_len)
        self.ckp.load(self)
        self.error_last = 1e8
        self.dry_run = hasattr(self.args, 'keyword') and 'lr-test' in self.args.keyword

        # load policy for epoch updating, still has bugs, model has no policy
        # self.epoch_policies = self.model.policy.read_policy(getattr(args, 'policy', ''), section='epoch')

        # self.best_ssim = [0] * len(self.loader_test)

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.4e}'.format(epoch, lr)) #Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        # leverage policies on epoch
        # self.model.policy.deploy_on_epoch(self.model, self.epoch_policies, epoch, optimizer=self.optimizer)

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            if not self.dry_run:
                lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            if not self.dry_run:
                self.optimizer.zero_grad()
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)

                if 'quant_loss' in self.args.global_buffer:
                    loss += self.args.global_buffer['quant_loss']
                    self.args.global_buffer.pop('quant_loss')

                loss.backward()
                if self.args.gclip > 0:
                    utils.clip_grad_value_(
                        self.model.parameters(),
                        self.args.gclip
                    )
                self.optimizer.step()

            self.optimizer.iteration()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # tot_ssim = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    # ssim = utility.calc_ssim(sr, hr, scale, test_y_channel=True, dataset=d)
                    # print('ssim:', ssim)
                    # tot_ssim += ssim

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                # ave_ssim = tot_ssim / len(d)
                # if ave_ssim > self.best_ssim[idx_data]:
                #     self.best_ssim[idx_data] = ave_ssim
                # print('\nave_ssim:', ave_ssim)
                # print('best_ssim:', self.best_ssim[idx_data])
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

