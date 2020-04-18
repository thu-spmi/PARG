import torch
import random
import numpy as np
from config import global_config as cfg
from reader import CamRest676Reader, get_glove_matrix
from reader import MultiWOZReader
from tsd_net import TSD, cuda_, nan, Paraphrase
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from reader import pad_sequences
import argparse, time
from nltk.tokenize import word_tokenize

from metric import CamRestEvaluator, MultiWOZEvaluator
import logging
from data_analysis import realization, slots_match, realization_multiwoz, slots_match_multiwoz
from nltk.translate.bleu_score import sentence_bleu
from filter_eval import filter_punct
from tsd_net import get_sparse_input_aug, get_sparse_selective_input

class Model:
    def __init__(self, dataset):
        reader_dict = {
            'camrest': CamRest676Reader,
            'multiwoz': MultiWOZReader,
        }
        model_dict = {
            'TSD':TSD
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator,
            'multiwoz': MultiWOZEvaluator,
        }
        self.reader = reader_dict[dataset]()
        self.m_para = Paraphrase(embed_size=cfg.para_embedding_size,
                                 hidden_size=cfg.para_hidden_size,
                                 vocab_size=cfg.vocab_size,
                                 layer_num=cfg.layer_num,
                                 dropout_rate=0.0,
                                 max_para_len=cfg.max_para_len,
                                 eos_token_idx=self.reader.vocab.encode('EOS_U'),
                                 vocab=self.reader.vocab,
                                 teacher_force=cfg.teacher_force,
                                 reader=self.reader,
                                 a_length=cfg.a_length)
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                                   hidden_size=cfg.hidden_size,
                                   vocab_size=cfg.vocab_size,
                                   layer_num=cfg.layer_num,
                                   dropout_rate=cfg.dropout_rate,
                                   z_length=cfg.z_length,
                                   max_ts=cfg.max_ts,
                                   beam_search=cfg.beam_search,
                                   beam_size=cfg.beam_size,
                                   eos_token_idx=self.reader.vocab.encode('EOS_M'),
                                   vocab=self.reader.vocab,
                                   teacher_force=cfg.teacher_force,
                                   degree_size=cfg.degree_size,
                                   reader=self.reader,
                                   para_hidden_size=cfg.para_hidden_size)

        self.EV = evaluator_dict[dataset]  # evaluator class
        if cfg.cuda:
            self.m = self.m.cuda()
            self.m_para = self.m_para.cuda()
        self.optim = Adam(lr=cfg.lr, params=[{'params': filter(lambda x: x.requires_grad, self.m.parameters())},
                                             {'params': filter(lambda x: x.requires_grad, self.m_para.parameters())}],
                          weight_decay=5e-5)
        # self.optim_para = Adam(lr=cfg.lr_para, params=filter(lambda x: x.requires_grad, self.m_para.parameters()),
        # weight_decay=5e-5)
        self.base_epoch = -1

    def _convert_batch_para(self, py_batch, mode, prev_a_py=None):
        u_input_np = pad_sequences(py_batch['delex_user'], cfg.max_para_len, padding='post',
                                   truncating='pre').transpose((1, 0))
        delex_para_input_np = pad_sequences(py_batch['delex_para'], cfg.max_para_len, padding='post',
                                            truncating='pre').transpose((1, 0))
        u_len = np.array(py_batch['delex_u_len'])
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        delex_para_input = cuda_(Variable(torch.from_numpy(delex_para_input_np).long()))
        if mode == 'test':
            if prev_a_py:
                for i in range(len(prev_a_py)):
                    eob = self.reader.vocab.encode('EOS_A')
                    if eob in prev_a_py[i] and prev_a_py[i].index(eob) != len(prev_a_py[i]) - 1:
                        idx = prev_a_py[i].index(eob)
                        prev_a_py[i] = prev_a_py[i][:idx + 1]
                    else:
                        prev_a_py[i] = [eob]
                    for j, word in enumerate(prev_a_py[i]):
                        if word >= cfg.vocab_size or word < 0:
                            prev_a_py[i][j] = 2 #unk
            else:
                prev_a_py = py_batch['pre_dial_act']
            prev_dial_act_input_np = pad_sequences(prev_a_py, cfg.a_length, padding='post', truncating='pre').transpose((1, 0))
            prev_dial_act_input = cuda_(Variable(torch.from_numpy(prev_dial_act_input_np).long()))
        else:
            prev_dial_act_input_np = pad_sequences(py_batch['pre_dial_act'], cfg.a_length, padding='post',
                                                   truncating='pre').transpose((1, 0))
            prev_dial_act_input = cuda_(Variable(torch.from_numpy(prev_dial_act_input_np).long()))

        return u_input, u_input_np, delex_para_input, delex_para_input_np, u_len, prev_dial_act_input

    def _get_final_input(self, py_batch, para_results, epoch):
        user = py_batch['user']
        para = py_batch['para']
        batch_size = len(py_batch['user'])
        final_user = []
        final_u_len = []
        weight = 1.0
        good = 0
        total = 0
        if epoch % 2 == 0:
            for i in range(batch_size):
                final_user.append(user[i])
                final_u_len.append(len(user[i]))
        else:
            user_delex = py_batch['delex_user']
            pre_sys = py_batch['pre_response']
            slu = []
            replace = []
            user_utter = []
            delex_user_utter = []
            para_utter = []
            if cfg.dataset == "camrest":
                slu = py_batch['realize_slu']
                replace = py_batch['replace']
            else:
                slu = py_batch['realize_slu']

            if cfg.dataset == "camrest":
                gen_para, success = realization(para_results, slu, replace)
            else:
                gen_para, success = realization_multiwoz(para_results, slu)
            for i in range(batch_size):
                user_utter.append(self.reader.vocab.sentence_decode(user[i]).split('EOS_M ')[-1])
                delex_user_utter.append(self.reader.vocab.sentence_decode(user_delex[i]).split('EOS_M ')[-1])
                para_utter.append(self.reader.vocab.sentence_decode(para[i]).split('EOS_M ')[-1])
            if cfg.dataset == "camrest":
                slots_match_success = slots_match(delex_user_utter, para_results)
            else:
                slots_match_success = slots_match_multiwoz(delex_user_utter, para_results)

            for i in range(batch_size):
                s1 = filter_punct(user_utter[i]).split(" ")
                s2 = filter_punct(para_utter[i]).split(" ")
                s = filter_punct(gen_para[i]).split(" ")
                if success[i] and slots_match_success[i] and sentence_bleu([s1, s2], s) > cfg.bleu_threshold:
                    good += 1
                    select = word_tokenize(gen_para[i]) + ['EOS_U']
                    select = self.reader.vocab.sentence_encode(select)
                    final = pre_sys[i] + select
                    final_user.append(final)
                    final_u_len.append(len(final))
                else:
                    final_user.append(user[i])
                    final_u_len.append(len(user[i]))
                total += 1
            weight = weight * good / total

        py_batch['final_user'] = final_user
        py_batch['final_u_len'] = final_u_len
        return py_batch, weight

    def _convert_batch(self, py_batch, prev_z_py=None, mode="train"):
        domain = py_batch['domain']
        if mode == "train":
            u_input_py = py_batch['final_user']
            u_len_py = py_batch['final_u_len']
        else:
            u_input_py = py_batch['user']
            u_len_py = py_batch['u_len']
        kw_ret = {}
        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size or word < 0:
                        prev_z_py[i][j] = 2 #unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['bspan'], padding='post').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))

        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        z_input = cuda_(Variable(torch.from_numpy(z_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))

        kw_ret['z_input_np'] = z_input_np

        return u_input, u_input_np, z_input, m_input, m_input_np, u_len, m_len, degree_input, kw_ret, domain

    def train(self):
        lr = cfg.lr
        lr_para = cfg.lr_para
        prev_min_loss, early_stop_count \
            = 1 << 30, cfg.early_stop_count
        train_time = 0
        for epoch in range(cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            self.m_para.self_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim
            # optim_para = self.optim_para
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    fail = 0
                    if not turn_batch['delex_user']:
                        fail = 1
                    for utter in turn_batch['delex_user']:
                        if not utter:
                            fail = 1
                    if fail:
                        break
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    # optim_para.zero_grad()

                    u_input, u_input_np, para_input, para_input_np, u_len, prev_act_input \
                        = self._convert_batch_para(turn_batch, 'train')
                    sparse_u_input_para = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                    para_dec_outs, para_idx, loss_para = self.m_para(u_input=u_input,
                                                                     para_input=para_input,
                                                                     prev_act_input=prev_act_input,
                                                                     u_input_np=u_input_np,
                                                                     u_len=u_len, mode="train",
                                                                     sparse_u_input_para=sparse_u_input_para)
                    para_results = self.reader.get_para_result(turn_batch, para_idx)
                    turn_batch, weight = self._get_final_input(turn_batch, para_results, epoch)

                    final_u_input, final_u_input_np, z_input, m_input, m_input_np, final_u_len, \
                    m_len, degree_input, kw_ret, domain \
                        = self._convert_batch(turn_batch, prev_z, "train")
                    sparse_u_input_bspan = Variable(get_sparse_input_aug(final_u_input_np), requires_grad=False)
                    z_input_np = z_input.cpu().data.numpy()
                    sparse_u_input_response = Variable(get_sparse_selective_input(z_input_np, self.reader.vocab),
                                                       requires_grad=False)
                    loss, pr_loss, m_loss, turn_states = self.m(u_input=final_u_input, z_input=z_input,
                                                                m_input=m_input, domain=domain,
                                                                degree_input=degree_input,
                                                                u_input_np=final_u_input_np,
                                                                m_input_np=m_input_np,
                                                                turn_states=turn_states,
                                                                para_dec=para_dec_outs,
                                                                para_input_np=para_input_np,
                                                                sparse_bspan=sparse_u_input_bspan,
                                                                sparse_response=sparse_u_input_response,
                                                                u_len=final_u_len, m_len=m_len, mode='train', **kw_ret)
                    total_loss = loss_para + loss
                    total_loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    torch.nn.utils.clip_grad_norm_(self.m_para.parameters(), 5.0)
                    optim.step()
                    # optim_para.step()
                    sup_loss += total_loss.cpu().item()
                    sup_cnt += 1
                    logging.debug(
                        'para_loss:{} loss:{} pr_loss:{} m_loss:{} grad:{}'.format(loss_para.item(),
                                                                                   loss.item(),
                                                                                   pr_loss.item(),
                                                                                   m_loss.item(),
                                                                                   grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time() - sw))
            valid_loss = valid_sup_loss + valid_unsup_loss

            self.save_model(epoch)
            self.save_model_para(epoch)

            if valid_loss <= prev_min_loss:
                prev_min_loss = valid_loss
                early_stop_count = cfg.early_stop_count
            else:
                prev_min_loss = valid_loss
                early_stop_count -= 1
                if not early_stop_count:
                    lr *= cfg.lr_decay
                    lr_para *= cfg.lr_decay
                    if lr < 0.08 * cfg.lr:
                        break
                    self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                                      weight_decay=5e-5)
                    self.optim_para = Adam(lr=lr_para, params=filter(lambda x: x.requires_grad, 
                                                                     self.m_para.parameters()), weight_decay=5e-5)
                    logging.info('early stop count out, learning rate %f' % lr)
                    early_stop_count = cfg.early_stop_count
                
    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        self.reader.para_result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            prev_act = None
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, para_input, para_input_np, u_len, prev_act_input \
                    = self._convert_batch_para(turn_batch, 'test', prev_act)
                sparse_u_input_para = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                para_dec_outs, para_idx, prev_act_idx = self.m_para(u_input=u_input,
                                                                 para_input=para_input,
                                                                 u_input_np=u_input_np,
                                                                 u_len=u_len, mode=mode,
                                                                 prev_act_input=prev_act_input,
                                                                 sparse_u_input_para=sparse_u_input_para)
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret, domain \
                    = self._convert_batch(turn_batch, prev_z, mode=mode)
                sparse_u_input_bspan = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                m_idx, z_idx, turn_states = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                                   m_input=m_input, domain=domain,
                                                   para_dec=para_dec_outs,
                                                   para_input_np=para_input_np,
                                                   sparse_bspan=sparse_u_input_bspan,
                                                   sparse_response=None,
                                                   degree_input=degree_input, u_input_np=u_input_np,
                                                   m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                                   dial_id=turn_batch['dial_id'], **kw_ret)
                self.reader.wrap_result(turn_batch, m_idx, z_idx, prev_z=prev_z)
                self.reader.save_result_para(turn_batch, para_idx)
                prev_z = z_idx
                prev_act = prev_act_idx
        self.reader.result_file.close()
        self.reader.para_result_file.close()
        ev = self.EV(result_path=cfg.result_path)
        res = ev.run_metrics()
        self.m.train()
        return res

    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, para_input, para_input_np, u_len, prev_act_input \
                    = self._convert_batch_para(turn_batch, 'train')
                sparse_u_input_para = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                para_dec_outs, _, loss_para = self.m_para(u_input=u_input,
                                                                 para_input=para_input,
                                                                 u_input_np=u_input_np,
                                                                 u_len=u_len, mode="train",
                                                                 prev_act_input=prev_act_input,
                                                                 sparse_u_input_para=sparse_u_input_para)
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret, domain \
                    = self._convert_batch(turn_batch, mode="val")
                sparse_u_input_bspan = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                z_input_np = z_input.cpu().data.numpy()
                sparse_u_input_response = Variable(get_sparse_selective_input(z_input_np, self.reader.vocab),
                                                   requires_grad=False)
                loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                            m_input=m_input, domain=domain,
                                                            turn_states=turn_states,
                                                            degree_input=degree_input,
                                                            para_dec=para_dec_outs,
                                                            para_input_np=para_input_np,
                                                            sparse_bspan=sparse_u_input_bspan,
                                                            sparse_response=sparse_u_input_response,
                                                            u_input_np=u_input_np, m_input_np=m_input_np,
                                                            u_len=u_len, m_len=m_len, mode='train', **kw_ret)
                total_loss = loss + loss_para
                sup_loss += total_loss.item()
                sup_cnt += 1
                logging.debug(
                    'para_loss:{} loss:{} pr_loss:{} m_loss:{}'.format(loss_para.item(),loss.item(),
                                                                       pr_loss.item(), m_loss.item()))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        print('result preview...')
        self.eval()
        return sup_loss, unsup_loss

    def reinforce_tune(self):
        lr = cfg.lr
        lr_para = cfg.lr_para
        self.optim = Adam(lr=cfg.lr, params=[{'params': filter(lambda x: x.requires_grad, self.m.parameters())},
                                             {'params': filter(lambda x: x.requires_grad, self.m_para.parameters())}],
                          weight_decay=5e-5)
        # self.optim_para = Adam(lr=cfg.lr_para, params=filter(lambda x: x.requires_grad, self.m_para.parameters()))
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        for epoch in range(self.base_epoch + cfg.rl_epoch_num + 1):
            mode = 'rl'
            if epoch <= self.base_epoch:
                continue
            epoch_loss, cnt = 0,0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim  #Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=0)
            # optim_para = self.optim_para
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    optim.zero_grad()
                    # optim_para.zero_grad()
                    u_input, u_input_np, para_input, para_input_np, u_len, prev_act_input \
                        = self._convert_batch_para(turn_batch, 'rl')
                    sparse_u_input_para = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                    para_dec_outs, _, _ = self.m_para(u_input=u_input,
                                                   para_input=para_input,
                                                   u_input_np=u_input_np,
                                                   prev_act_input=prev_act_input,
                                                   u_len=u_len, mode="rl",
                                                   sparse_u_input_para=sparse_u_input_para)
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret, domain\
                        = self._convert_batch(turn_batch, prev_z, mode="rl")
                    sparse_u_input_bspan = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)
                    loss_rl = self.m(u_input=u_input, z_input=z_input,
                                     m_input=m_input, domain=domain,
                                     degree_input=degree_input,
                                     u_input_np=u_input_np,
                                     m_input_np=m_input_np,
                                     turn_states=turn_states,
                                     para_dec=para_dec_outs,
                                     para_input_np=para_input_np,
                                     sparse_bspan=sparse_u_input_bspan,
                                     sparse_response=None,
                                     dial_id=turn_batch['dial_id'],
                                     u_len=u_len, m_len=m_len, mode=mode, **kw_ret)

                    if loss_rl is not None:
                        loss = loss_rl #+ loss_mle * 0.1
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm(self.m.parameters(), 2.0)
                        optim.step()
                        epoch_loss += loss.cpu().item()
                        cnt += 1
                        logging.debug('{} loss {}, grad:{}'.format(mode, loss.item(), grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = epoch_loss / (cnt + 1e-8)
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            valid_loss = valid_sup_loss + valid_unsup_loss

            # self.save_model(epoch)

            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
                early_stop_count = cfg.early_stop_count
            else:
                early_stop_count -= 1
                if not early_stop_count:
                    lr *= cfg.lr_decay
                    if lr < 0.1 * cfg.lr:
                        break
                    logging.info('early stop count out, learning rate %f' % lr)
                    early_stop_count = cfg.early_stop_count

    def save_model_para(self, epoch, path=None, critical=False):
        if not path:
            path = cfg.para_path + '_' + str(epoch) + '.pkl'
        if critical:
            path += '.final'
        all_state = {'lstd': self.m_para.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def save_model(self, epoch, path=None, critical=False):
        if not path:
            path = cfg.model_path + '_' + str(epoch) + '.pkl'
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, epoch, path=None):
        if not path:
            path = cfg.model_path + '_' + str(epoch) + '.pkl'
        all_state = torch.load(path, map_location='cpu')
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = epoch

    def load_model_para(self, epoch, path=None):
        if not path:
            path = cfg.para_path + '_' + str(epoch) + '.pkl'
        all_state = torch.load(path, map_location='cpu')
        self.m_para.load_state_dict(all_state['lstd'])

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.z_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.m_decoder.emb.weight.data.copy_(embedding_arr)

    def count_params(self):
        dialogue_module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        paraphrase_module_parameters = filter(lambda p: p.requires_grad, self.m_para.parameters())
        dial_param_cnt = sum([np.prod(p.size()) for p in dialogue_module_parameters])
        para_param_cnt = sum([np.prod(p.size()) for p in paraphrase_module_parameters])
        param_cnt = dial_param_cnt + para_param_cnt
        print('total trainable params: %d' % param_cnt)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-model')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.init_handler(args.model)
    cfg.dataset = args.model.split('-')[-1]

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.info(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.info('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(args.model.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        # m.load_model(cfg.start_epoch)
        # m.load_model_para(cfg.para_start_epoch)
        m.train()
    elif args.mode == 'adjust':
        m.load_model(cfg.start_epoch)
        m.load_model_para(cfg.para_start_epoch)
        m.train()
    elif args.mode == 'test':
        m.load_model(cfg.start_epoch)
        m.load_model_para(cfg.para_start_epoch)
        m.eval()
    elif args.mode == 'rl':
        m.load_model(cfg.start_epoch)
        m.load_model_para(cfg.para_start_epoch)
        m.reinforce_tune()


if __name__ == '__main__':
    main()
