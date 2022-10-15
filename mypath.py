class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal_voc_aug':
            return r'D:\MyDocs\pythonProject\DataSets\pascal_voc_aug'
        elif database == 'pascal':
            return r'D:\MyDocs\pythonProject\DataSets\pascal\VOC2012'   # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return r'D:\MyDocs\pythonProject\DataSets\benchmark_RELEASE'  # folder that contains dataset/.
        elif database == 'coco':
            return '/path/to/coco'  # folder that contains annotations/.
        elif database == 'fixation':
            return r'D:\MyDocs\pythonProject\DataSets\pascal_voc_aug\Fixation'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        # return '/path/to/Models/'
        return './models/'
