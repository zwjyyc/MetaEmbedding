import numpy as np
import theano
import theano.tensor as T

from utils import say
from .initialization import default_srng, default_rng, USE_XAVIER_INIT
from .initialization import set_default_rng_seed, random_init, create_shared
from .initialization import ReLU, sigmoid, tanh, softmax, linear, get_activation_by_name


class Dropout(object):
    def __init__(self, dropout_prob, srng=None, v2=False):
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(
                n=1,
                p=1 - self.dropout_prob,
                size=x.shape,
                dtype=theano.config.floatX
            )
        return x * mask / d


def apply_dropout(x, dropout_prob, v2=False):
    return Dropout(dropout_prob, v2=v2).forward(x)


class Layer(object):
    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            has_bias=True, scale = 1):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias
        self.scale = scale
        self.create_parameters()

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(n_in, n_out, activation)

    def initialize_params(self, n_in, n_out, activation):
        scale = self.scale
        if USE_XAVIER_INIT:
            if activation == ReLU:
                scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            elif activation == softmax:
                scale = np.float64(0.001 * scale).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            W_vals = random_init((n_in,n_out), rng_type="normal") * scale
        else:
            W_vals = random_init((n_in,n_out))
            if activation == softmax:
                W_vals *= (0.001 * self.scale)
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            else:
                b_vals = random_init((n_out,))
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(
                    T.dot(x, self.W) + self.b
                )
        else:
            return self.activation(
                    T.dot(x, self.W)
                )

    @property
    def params(self):
        if self.has_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())


class EmbeddingLayer(object):
    def __init__(self, vocab, n_d=300, oov="<unk>", pre_embs=None, fix_init_embs=False):
        # print vocab
        if pre_embs is not None:
            t_word = ''
            t_vector = []
            for word, vector in pre_embs:
                if n_d != len(vector):
                    say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(vector), len(vector)
                        ))
                n_d = len(vector)
                t_word = word
                t_vector = vector
                break
            
            embs = random_init((len(vocab) + 1, n_d)) * 0.01
            cnt = 0
            t_word = t_word.decode('utf8')
            if t_word in vocab:
                embs[vocab[t_word]] = t_vector
                cnt = 1

            for word, vector in pre_embs:
                uword = word.decode('utf8')
                if uword in vocab:
                    embs[vocab[uword]] = vector
                    cnt += 1

            say("{} pre-trained embeddings loaded.\n".format(cnt))
            embs[len(vocab)] = random_init((n_d,)) * 0.0 # for oov embs
            emb_vals = embs  # np.vstack(embs).astype(theano.config.floatX)
        else:
            emb_vals = random_init((len(vocab) + 1, n_d))

        self.init_end = len(vocab) if fix_init_embs else -1
        self.embs = create_shared(emb_vals)
        self.n_d = n_d

    def forward(self, x):
        return self.embs[x]

    @property
    def params(self):
        return [self.embs]

    @params.setter
    def params(self, param_list):
        self.embs.set_value(param_list[0].get_value())

