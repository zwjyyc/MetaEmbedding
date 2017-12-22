import theano
import theano.tensor as T

from nn import apply_dropout, Layer, linear
from nn import create_optimization_updates
from utils import *

class One2NPlusModel(object):
    def __init__(self, args, embs, weights):
        self.args = args
        self.embs = embs
        self.weights = weights

    def build_model(self):
        args = self.args
        weights = self.weights

        meta_emb = self.meta_emb = self.embs[0]
        golden_embs = self.embs[1:]

        n_m_d = meta_emb.n_d
        dropout = self.dropout = theano.shared(np.float64(args.dropout_rate).astype(theano.config.floatX))

        batch_ids = self.batch_ids = T.ivector('batch_d_char')
        batch_masks = self.batch_masks = T.fmatrix('batch_d_char_mask')

        layers = self.layers = [meta_emb]

        slices_embs = meta_emb.forward(batch_ids.ravel())
        slices_embs = slices_embs.reshape((batch_ids.shape[0], n_m_d))
        prev_output = apply_dropout(slices_embs, dropout, v2=True)

        self.all_loss = 0.0
        for i in range(len(weights)):
            mask, weight, golden_emb = batch_masks[i], weights[i], golden_embs[i]
            n_o_d = golden_emb.n_d
            layer = Layer(n_m_d, n_o_d, linear)
            layers.append(layer)
            mapped_output = layer.forward(prev_output)

            slices_embs = golden_emb.forward(batch_ids.ravel())
            slices_embs = slices_embs.reshape((batch_ids.shape[0], n_o_d))
            self.all_loss += weight * T.sum(T.sum((mapped_output - slices_embs) * (mapped_output - slices_embs), axis=1) * mask) / (1e-8 + T.sum(mask))

        for i, l in enumerate(layers[1:]):
            say("layer {}: n_in={}\tn_out={}\n".format(
                i, l.n_in, l.n_out
            ))

        self.l2_sqr = None
        self.params = []

        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p ** 2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p ** 2)

        self.all_loss += self.l2_sqr
        n_params = sum(len(x.get_value(borrow=True).ravel()) for x in self.params)
        say("total # parameters: {}\n".format(n_params))

    def build_graph(self):
        args = self.args
        cost = self.all_loss
        meta_emb = self.meta_emb
        updates, lr, gnorm = create_optimization_updates(
            cost=cost,
            params=self.params,
            lr=args.learning_rate,
            method=args.learning
        )[:3]

        train_model = theano.function(
            inputs=[self.batch_ids, self.batch_masks],
            outputs=[cost, gnorm],
            updates=updates,
            allow_input_downcast=True
        )

        predict_model = theano.function(
            inputs=[self.batch_ids, self.batch_masks],
            outputs=cost,
            allow_input_downcast=True
        )

        embs_output = theano.function(
            inputs=[self.batch_ids],
            outputs=meta_emb.embs,
            allow_input_downcast=True
        )

        return train_model, predict_model, embs_output, self.params