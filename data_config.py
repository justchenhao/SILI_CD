import os


class DataConfig(dict):
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    label_folder_name = 'label'
    img_folder_name = ['A']
    img_folder_names = ['A', 'B']
    n_class = 2
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'Path of the LEVIR-CD root'
            self.root_dir = 'D:/dataset/CD/LEVIR-CD/cut/'
        elif data_name == 'levir_try':
            self.root_dir = r'./samples/LEVIR_cut'
        elif data_name == 'SV_CD':
            self.root_dir = 'Path of the SV-CD root'
            # self.root_dir = '../data/ccd256'
        elif data_name == 'DE_CD':
            self.root_dir = 'Path of the DE-CD root'
            self.img_folder_names = ['A_low', 'B']
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


def get_pretrained_path(pretrained):
    out = None
    if pretrained is not None:
        if os.path.isfile(pretrained):
            out = pretrained
        elif pretrained == 'imagenet':
            out = pretrained
        elif pretrained == 'None' or pretrained == 'none':
            out = None
        else:
            raise NotImplementedError(pretrained)
    else:
        out = None
    return out


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)
    print(data.n_class)

