import torch

import utility
import data
import model
import loss
from option import args
import os
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.calc_flops:
        args.cpu = True

    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)

            if args.calc_flops:
                from ptflops import get_model_complexity_info
                from thop import profile
                from torchstat import stat

                # macs, params = get_model_complexity_info(_model, (3, 678, 1020), as_strings=True,
                #                                          print_per_layer_stat=True, verbose=True,
                #                                          ignore_modules=['BatchNorm2d', 'LeakyReLU'])

                # input = torch.randn(1, 3, 678, 1020)
                # macs, params = profile(_model, inputs=(input,))

                # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

                if args.scale[0] == 2:
                    stat(_model, (3, 678, 1020))
                elif args.scale[0] == 4:
                    stat(_model, (3, 339, 510))

                return

            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == '__main__':
    main()
