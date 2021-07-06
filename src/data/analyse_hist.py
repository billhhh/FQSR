import torch
import matplotlib.pyplot as plt
import numpy as np


def draw_hist(x, path, clear=True):

    if clear:
        plt.clf()

    if 'before' in path:
        # plt.hist(x, bins='auto')
        # plt.hist(x, rwidth=0.4, bins='auto')
        plt.hist(x, rwidth=0.4, bins=256)
    elif 'after' in path:
        # plt.hist(x, bins=16)
        plt.hist(x, rwidth=0.4, bins=16)

    plt.xlim((-0.015, 0.016))
    plt.xticks(np.arange(-0.015, 0.016, 0.005))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.show()
    plt.savefig(path, bbox_inches='tight')


pt_path = '../../experiment/' \
          'SRResNet_x2_a4w4o8_lr=1e-3_pretrain_fp_stable20_noqloss_half_sgdr_drawfig_20210306/' \
          'hist/iter320998/'
name = 'index37_wt_before_quant'

pt_path = pt_path + name + '.pt'
x = torch.load(pt_path)
x = x.cpu().detach().numpy().flatten()

print('mean', np.mean(x), 'std', np.std(x))

draw_hist(x, name)
