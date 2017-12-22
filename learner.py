import os
import codecs
import time
import random
import theano

from nn import EmbeddingLayer
from one2nplus import *
from utils import *


def build_dic(files, dict_file):
    word_dic = {}
    print 'Building word dict from '
    print files

    for fi in files:
        print fi
        with open(fi, 'r') as fin:
            cnt = 0
            for line in fin:
                cnt += 1
                items = line.strip().split()
                if cnt == 1:
                    assert len(items) == 2, 'the first line must contain numbers of words and dimension'
                    continue

                items[0] = items[0].decode('utf8', 'ignore')
                if items[0] not in word_dic:
                    size = len(word_dic)
                    word_dic[items[0]] = size
            print 'Loaded %d lines!' % cnt

    print 'The big dictionary contains %d words' % len(word_dic)
    write_dic(word_dic, dict_file)
    return word_dic


def write_dic(word_dic, src):
    with codecs.open(src, 'w', 'utf8') as fout:
        for key, val in sorted(word_dic.items(), key=lambda (k, v): (v, k)):
            out_str = '%s\t%d\n' % (key, val)
            fout.write(out_str)


def load_dic(src):
    word_dic = {}
    print 'Loading word dict from %s' % src
    with codecs.open(src, 'r', 'utf8') as fin:
        for line in fin:
            items = line.strip().split()
            if len(items) != 2:
                print '@#@' + line + '@#@'
                continue
            word_dic[items[0]] = int(items[1])
    out_str = 'Loaded %d words!' % len(word_dic)
    print out_str
    return word_dic


def normalize_weight(weights):
    sum_ = 0.0
    for weight in weights:
        sum_ += weight

    for i in range(len(weights)):
        weights[i] = weights[i] * 1.0 / sum_

    print weights
    return weights


def random_batches(batches_train_ids, batches_train_masks):
    perm = range(len(batches_train_ids))
    random.shuffle(perm)
    batches_train_ids = [batches_train_ids[i] for i in perm]
    batches_train_masks = [batches_train_masks[i] for i in perm]
    return batches_train_ids, batches_train_masks


def create_batches(size, word_dic, embs):
    ids = range(len(word_dic))
    masks = []
    for emb_in in embs:
        tmp_mask = [0.0] * len(word_dic)
        with open(emb_in, 'r') as fin:
            for line in fin:
                items = line.strip().split()
                if len(items) == 2:
                    continue

                word = items[0].decode('utf8', 'ignore')
                if word in word_dic:
                    word_id = word_dic[word]
                    if word_id < len(word_dic):
                        tmp_mask[word_id] = 1
        masks.append(tmp_mask)

    batches_train_ids = []
    batches_train_masks = []
    batches_dev_ids = []
    batches_dev_masks = []
    word_cnt = len(word_dic)
    slices_num = word_cnt / size
    for slice in range(slices_num):
        start = slice * size
        end = (slice + 1) * size
        batch_id = ids[start:end]
        batch_mask = masks[:][start:end]

        if slice < slices_num - 100:
            batches_train_ids.append(batch_id)
            batches_train_masks.append(batch_mask)
        else:
            batches_dev_ids.append(batch_id)
            batches_dev_masks.append(batch_mask)

    perm = range(len(batches_train_ids))
    random.shuffle(perm)
    batches_train_ids = [batches_train_ids[i] for i in perm]
    batches_train_masks = [batches_train_masks[i] for i in perm]
    return batches_train_ids, batches_train_masks, batches_dev_ids, batches_dev_masks


class Learner(object):
    def __init__(self, args):
        self.args = args
        self.embs = args.embs[0::2]
        self.weights = normalize_weight([int(k) for k in args.embs[1::2]])
        assert len(self.embs) == len(self.weights), 'number of embs must be equal to number of weights'

        self.dic = {}
        if os.path.exists(args.dict):
            self.dic = load_dic(args.dict)
        else:
            self.dic = build_dic(self.embs, args.dict)

        embs_set = [EmbeddingLayer(self.dic, n_d=128)]
        for embs_in in self.embs:
            print embs_in
            embs_set.append(EmbeddingLayer(self.dic, pre_embs=load_embedding_iterator(embs_in)))

        self.mapper = One2NPlusModel(args, embs_set, self.weights)

    def run(self):
        mapper = self.mapper
        args = self.args
        embs = self.embs
        dic = self.dic

        print args
        size = args.batch
        train_ids, train_masks, dev_ids, dev_masks = create_batches(size, dic, embs)

        print 'building model...'
        mapper.build_model()

        print 'Building graph...'
        train_model, predict_model, meta_embs_output, params = mapper.build_graph()
        print 'Done'

        unchanged = 0
        dropout_rate = np.float64(args.dropout_rate).astype(theano.config.floatX)

        start_time = time.time()
        eval_period = args.eval_period

        say(str(["%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in params]) + "\n")
        mapper.dropout.set_value(dropout_rate)
        print 'begin to train...'
        for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 20: return
            train_loss = 0.0
            train_ids, train_masks = random_batches(train_ids, train_masks)
            N = len(train_ids)
            print 'N %d' % N
            for i in xrange(N):
                if (i + 1) % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()
                va, grad_norm = train_model(train_ids[i], train_masks[i])

                train_loss += va
                if (i == N - 1) or (eval_period > 0 and (i + 1) % eval_period == 0):
                    mapper.dropout.set_value(0)
                    say("\n")
                    say("Epoch %.3f\tloss=%.4f\t|g|=%s  [%.2fm]\n" % (
                        epoch + (i + 1) / (N + 0.0),
                        train_loss / (i + 1),
                        float(grad_norm),
                        (time.time() - start_time) / 60.0
                    ))
                    say(str(["%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in params]) + "\n")

                    M = len(dev_ids)
                    print 'M %d' % M
                    valid_loss = 0
                    for j in xrange(M):
                        valid_loss += predict_model(dev_ids[j], dev_masks[j])
                    valid_loss /= M
                    say("Valid loss is %.4f\n" % valid_loss)

                    mapper.dropout.set_value(dropout_rate)
                    start_time = time.time()

            meta_embs = meta_embs_output(dev_ids[0])
            print meta_embs.shape
            print 'Done'
