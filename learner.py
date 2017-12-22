import os
import codecs


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


class Learner(object):
    def __init__(self, args):
        self.embs = args.embs[0::2]
        self.weights = [int(k) for k in args.embs[1::2]]
        assert len(self.embs) == len(self.weights), 'number of embs must be equal to number of weights'

        self.dic = {}
        if os.path.exists(args.dict):
            self.dic = load_dic(args.dict)
        else:
            self.dic = build_dic(self.embs, args.dict)

    def run(self):
        return

