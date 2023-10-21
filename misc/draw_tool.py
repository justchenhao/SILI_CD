import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
plt.rcParams['savefig.dpi'] = 400 #图片像素
plt.rcParams['figure.dpi'] = 400 #分辨率
import seaborn as sns

sns.set_style("darkgrid")
sns.set(color_codes=True)
sns.set_context("paper")


def save_acc_curve(project_name, train_name='train_acc', val_name='val_acc', out_name='acc',
                   checkpoint_root='checkpoints'):
    project_root = checkpoint_root + '/' + project_name
    train_acc = np.load(os.path.join(project_root, train_name+'.npy'), allow_pickle=True)
    val_acc = np.load(os.path.join(project_root, val_name+'.npy'), allow_pickle=True)
    n = len(val_acc)
    plt.plot(range(1, n + 1), train_acc[0:n])
    plt.plot(range(1, n + 1), val_acc[0:n])
    plt.legend([train_name, val_name])
    plt.grid(True)
    plt.title(project_name)
    plt.xlabel('epochs')
    plt.ylabel('mean F1')
    plt.savefig(os.path.join(project_root, out_name+'.png'))
    plt.close()


if __name__ == '__main__':

    project_name = r'CD_base_resnet50_LEVIR_b8_lr0.01_train_5p_val_200_linear_geokr_res50'

    project_root = 'checkpoints/' + project_name
    train_acc = np.load(os.path.join(project_root, 'train_acc.npy'))
    val_acc = np.load(os.path.join(project_root, 'val_acc.npy'))
    n = len(val_acc)
    plt.plot(range(1, n+1), train_acc[0:n])
    plt.plot(range(1, n+1), val_acc[0:n],'-')

    plt.legend(['train', 'val'])

    # title = 'training accuracy'
    plt.grid(True)
    # plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('training mean F1')

    plt.show()

    plt.savefig(os.path.join(project_root, 'acc_val_.png'))
    plt.savefig(os.path.join(project_root, 'acc_val_.pdf'))

