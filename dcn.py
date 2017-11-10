from __future__ import print_function
import sys
import os
# code to automatically download dataset
# curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# sys.path = [os.path.join(curr_path, "../autoencoder")] + sys.path
import mxnet as mx
import numpy as np
import aelib.data
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import aelib.model
from aelib.autoencoder import AutoEncoderModel
from aelib.solver import Solver, Monitor
import logging
import time

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((int(D),int(D)), dtype=np.int64)
  for i in range(Y_pred.size):
    w[int(Y_pred[i]), int(Y[i])] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def batch_km(data, center, count):
    """
    Function to perform a KMeans update on a batch of data, center is the centroid
    from last iteration.

    """
    N = data.shape[0]
    K = center.shape[0]

    # update assignment
    idx = np.zeros(N, dtype=np.int)
    idx_onehot = np.zeros([N,K], dtype=np.int)
    for i in range(N):
        dist = np.inf
        ind = 0
        for j in range(K):
            temp_dist = np.linalg.norm(data[i] - center[j])
            if temp_dist < dist:
                dist = temp_dist
                ind = j
        idx[i] = ind
        idx_onehot[i][ind]=1

    # update centriod
    center_new = center
    for i in range(N):
        c = idx[i]
        count[c] += 1
        eta = 1 / count[c]
        center_new[c] = (1 - eta) * center_new[c] + eta * data[i]

    return idx_onehot, center_new, count

class DCNModel(aelib.model.MXModel):

    class DCNLoss(mx.operator.NumpyOp):
        def __init__(self, num_centers, alpha):
            super(DCNModel.DCNLoss, self).__init__(need_top_grad=False)
            self.num_centers = num_centers
            self.alpha = alpha

        def forward(self, in_data, out_data):
            z = in_data[0]
            mu = in_data[1]
            q = out_data[0]
            self.mask = 1.0/(1.0+cdist(z, mu)**2/self.alpha)
            q[:] = self.mask**((self.alpha+1.0)/2.0)
            q[:] = (q.T/q.sum(axis=1)).T

        def backward(self, out_grad, in_data, out_data, in_grad):
            q = out_data[0]
            z = in_data[0]
            mu = in_data[1]
            p = in_data[2]
            dz = in_grad[0]
            dmu = in_grad[1]
            self.mask *= (self.alpha+1.0)/self.alpha*(p-q)
            dz[:] = (z.T*self.mask.sum(axis=1)).T - self.mask.dot(mu)
            dmu[:] = (mu.T*self.mask.sum(axis=0)).T - self.mask.T.dot(z)

        def infer_shape(self, in_shape):
            assert len(in_shape) == 3
            assert len(in_shape[0]) == 2
            input_shape = in_shape[0]
            label_shape = (input_shape[0], self.num_centers)
            mu_shape = (self.num_centers, input_shape[1])
            out_shape = (input_shape[0], self.num_centers)
            return [input_shape, mu_shape, label_shape], [out_shape]

        def list_arguments(self):
            return ['data', 'mu', 'label']

    def setup(self, X, num_centers, alpha, save_to='dec_model'):
        sep = X.shape[0]*9/10
        X_train = X[:sep]
        X_val = X[sep:]
        ae_model = AutoEncoderModel(self.xpu, [X.shape[1],500,500,2000,10], pt_dropout=0.2)
        if not os.path.exists(save_to+'_pt.arg'):
            ae_model.layerwise_pretrain(X_train, 256, 500000, 'sgd', l_rate=0.1, decay=0.0,
                                        lr_scheduler=mx.misc.FactorScheduler(20000,0.1))

            ae_model.finetune(X_train, 256, 100000, 'sgd', l_rate=0.1, decay=0.0,
                              lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
            ae_model.save(save_to+'_pt.arg')
            logging.log(logging.INFO, "Autoencoder Training error: %f"%ae_model.eval(X_train))
            logging.log(logging.INFO, "Autoencoder Validation error: %f"%ae_model.eval(X_val))

        else:
            ae_model.load(save_to+'_pt.arg')
        self.ae_model = ae_model

        self.dec_op = DCNModel.DCNLoss(num_centers, alpha)
        label = mx.sym.Variable('label')
        self.feature = self.ae_model.encoder
        self.loss = self.dec_op(data=self.ae_model.encoder, label=label, name='dec') #We need to change this!
        self.args.update({k:v for k,v in self.ae_model.args.items() if k in self.ae_model.encoder.list_arguments()})
        self.args['dec_mu'] = mx.nd.empty((num_centers, self.ae_model.dims[-1]), ctx=self.xpu)
        self.args_grad.update({k: mx.nd.empty(v.shape, ctx=self.xpu) for k,v in self.args.items()})
        self.args_mult.update({k: k.endswith('bias') and 2.0 or 1.0 for k in self.args})
        self.num_centers = num_centers

    def cluster(self, X, y=None, update_interval=None):
        N = X.shape[0]
        if not update_interval:
            update_interval = N
        batch_size = 256
        test_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=self.xpu) for k, v in self.args.items()}
        z = list(aelib.model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values())[0]
        kmeans = KMeans(self.num_centers, n_init=20)
        kmeans.fit(z)
        args['dec_mu'][:] = kmeans.cluster_centers_
        solver = Solver('sgd', momentum=0.9, wd=0.0, learning_rate=0.01)
        def l2_norm(label, pred): #Here we should add the autoencoder loss, but I don't know how
            M = args['dec_mu'].asnumpy()
            return np.mean(np.square(pred-np.dot(label,M)))/2.0
        solver.set_metric(mx.metric.CustomMetric(l2_norm))

        label_buff = np.zeros((X.shape[0], self.num_centers))
        train_iter = mx.io.NDArrayIter({'data': X}, {'label': label_buff}, batch_size=batch_size,
                                       shuffle=False, last_batch_handle='roll_over')
        self.y_pred = np.zeros((X.shape[0]))
        # added by jwj to implement DCN
        self.count = 10 * np.ones(self.num_centers, dtype=np.int)
        def refresh(i):
            if i%update_interval == 0:
                z = list(aelib.model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values())[0]
                center = args['dec_mu'].asnumpy()
                idx_onehot, center_new, count = batch_km(z, center, self.count)
                train_iter.data_list[1][:] = idx_onehot #s_ij
                args['dec_mu'][:] = center_new #m_ij
                self.count[:] = count #c_i^k

                # p = np.zeros((z.shape[0], self.num_centers))
                # self.dec_op.forward([z, args['dec_mu'].asnumpy()], [p]) #Can we also calculate the backward?
                # y_pred = p.argmax(axis=1)
                # print(np.std(np.bincount(y_pred)), np.bincount(y_pred))
                # print(np.std(np.bincount(y.astype(np.int))), np.bincount(y.astype(np.int)))
                # if y is not None:
                #     print(cluster_acc(y_pred, y)[0])
                # weight = 1.0/p.sum(axis=0)
                # weight *= self.num_centers/weight.sum()
                # p = (p**2)*weight
                # # We can update s, m, and store them in data_list.
                # train_iter.data_list[1][:] = (p.T/p.sum(axis=1)).T #This is where we can update "label"
                # print(np.sum(y_pred != self.y_pred), 0.001*y_pred.shape[0])
                # if np.sum(y_pred != self.y_pred) < 0.001*y_pred.shape[0]:
                #     self.y_pred = y_pred
                #     return True
                # self.y_pred = y_pred

        solver.set_iter_start_callback(refresh)
        solver.set_monitor(Monitor(50))

        solver.solve(self.xpu, self.loss, args, self.args_grad, None,
                     train_iter, 0, 20, {}, False)
        self.end_args = args
        if y is not None:
            return cluster_acc(self.y_pred, y)[0]
        else:
            return -1

def mnist_exp(xpu):
    X, Y = aelib.data.get_mnist()
    dec_model = DCNModel(xpu, X, 10, 1.0)
    acc = []
    for i in [10*(2**j) for j in range(1)]: #we first test 1 interval
        acc.append(dec_model.cluster(X, Y, i))
        logging.log(logging.INFO, 'Clustering Acc: %f at update interval: %d'%(acc[-1], i))
    logging.info(str(acc))
    logging.info('Best Clustering ACC: %f at update_interval: %d'%(np.max(acc), 10*(2**np.argmax(acc))))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mnist_exp(mx.cpu(0))
