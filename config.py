import logging
import time
import configparser

class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 2
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.6

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'
        self.dataset = 'unknown'

        self.seed = 0
  
    def init_handler(self, m):
        init_method = {
            'tsdf-camrest': self._camrest_tsdf_init,
            'tsdf-multiwoz': self._multiwoz_tsdf_init
        }
        init_method[m]()

    def _camrest_tsdf_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.para_embedding_size = 50
        self.para_hidden_size = 50
        self.split = (3, 1, 1)
        self.lr = 0.003
        self.lr_para = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-camrest.pkl'
        self.data = './data/CamRest676/CamRestFormal.json'
        self.entity = './data/CamRest676/CamRestOTGY.json'
        self.db = './data/CamRest676/CamRestDB.json'
        self.db_para = './data/CamRest676/CamRestPara.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.z_length = 8
        self.a_length = 4
        self.degree_size = 5
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100  # triggered by early stop
        self.rl_epoch_num = 1
        self.cuda = True
        self.spv_proportion = 100
        self.max_ts = 40
        self.max_para_len = 40
        self.early_stop_count = 1
        self.new_vocab = True
        self.model_path = './models/camrest'
        self.para_path = './models/camrest_para'
        self.result_path = './results/camrest.csv'
        self.para_result_path = './results/camrest_para.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.diversity_threshold = 3.4
        self.bleu_threshold = 0.2
        self.start_epoch = -1
        self.para_start_epoch = -1

    def _multiwoz_tsdf_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 6000
        self.embedding_size = 50
        self.hidden_size = 100
        self.para_embedding_size = 50
        self.para_hidden_size = 100
        self.split = (8, 1, 1)
        self.lr = 0.005
        self.lr_para = 0.005
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-multiwoz.pkl'
        self.data = './data/MultiWOZ/data_formal.json'
        self.entity = './data/MultiWOZ/value_set_processed.json'
        self.db = './data/MultiWOZ/'
        self.db_para = './data/MultiWOZ/data_para.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 64
        self.z_length = 30
        self.a_length = 20
        self.degree_size = 6
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100  # triggered by early stop
        self.rl_epoch_num = 1
        self.cuda = True
        self.spv_proportion = 100
        self.max_ts = 60
        self.max_para_len = 60
        self.early_stop_count = 5
        self.new_vocab = True
        self.model_path = './models/multiwoz'
        self.para_path = './models/multiwoz_para'
        self.result_path = './results/multiwoz.csv'
        self.para_result_path = './results/multiwoz_para.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.diversity_threshold = 3.4
        self.bleu_threshold = 0.2
        self.start_epoch = -1
        self.para_start_epoch = -1

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

