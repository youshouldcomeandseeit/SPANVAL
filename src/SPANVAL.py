

import torch
import numpy as np
import time
import copy
from sklearn.metrics import roc_auc_score, r2_score
from matplotlib import pyplot as plt
from functorch import make_functional_with_buffers, vmap, grad
from torch.func import grad, vmap
from scipy.stats import percentileofscore
from os import listdir, mkdir
from shutil import rmtree
class SPANVAL():

    def __init__(self, x_train, y_train, x_valid, y_valid, predictor, estimator, problem, include_marginal=True):
        ''''''
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.predictor = predictor
        self.estimator = estimator
        self.problem = problem

        self.include_marginal = include_marginal

        if self.problem == 'classification':
            self.num_classes = len(np.unique(self.y_train.detach().cpu().numpy()))

    def fit(self, model, x, y, crit, batch_size, lr, iters, use_cuda=True, verbose=False):
        '''Fits a model with a given number of iterations

        NOTE: There is a critical distinction between training with iterations rather than epochs; with epochs, more obs. results in
              more iterations/gradient-updates, whereas with iterations the amount of data is indenpendant of the number of iterations.
              This is critical for our DVRL implementation b/c the RL algorithm may be confounded by performance differences due to number
              of weight updates, e.g., hyper parameter of `inner_iter`. By training with iterations (rather than epochs), we ensure independance
              of `inner_iter` with sample importance.
        '''

        if torch.cuda.is_available() & use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        model = model.to(device).train()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(iters):

            batch_idx = torch.randint(0, x.size(0), size=(batch_size,))

            x_batch = x[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)

            optim.zero_grad()
            yhat_batch = model(x_batch)
            loss = crit(yhat_batch, y_batch)
            loss.backward()
            optim.step()
            if verbose: print(f'epoch: {i} | loss: {loss.item()}', end='\r')

        return model.cpu().eval()

    def _get_perf(self, model, x, y, metric, device='cpu'):
        '''return performance of `model` on `dataset` using `metric`'''
        yhat, y = self._predict(model, x, y, device=device)

        if metric == 'mse':
            crit = torch.nn.MSELoss()
            return crit(yhat, y).item()
        elif metric == 'r2':
            return r2_score(y.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel(),
                            multioutput='uniform_average')
        elif metric == 'bce':
            crit = torch.nn.BCELoss()
            return crit(yhat, y).item()
        elif metric == 'acc':
            return ((1. * ((yhat.argmax(dim=1)).view(-1) == y.view(-1))).mean()).item()
        elif metric == 'auroc':
            y = y.view(-1).detach().cpu().numpy()
            yhat = torch.softmax(yhat, dim=-1).detach().cpu().numpy()

            if yhat.shape[1] == 2:
                # binary classification
                return roc_auc_score(y, yhat[:, 1])
            elif yhat.shape[1] > 2:
                # multiclass
                return roc_auc_score(y, yhat, multi_class='ovr')
            else:
                # yhat size == 1; only makes sense for regression - and auroc is for classification
                # assumes we're not using BCE
                raise Exception('is this a regression problem? choose different perf. metric.')
        else:
            raise Exception('not implemented')

    def _predict(self, model, x, y, device='cpu', batch_size=256):
        '''return y,yhat for given model, dataset'''
        model = model.eval().to(device)
        _yhat = []
        _y = []
        with torch.no_grad():
            for batch_idx in torch.split(torch.arange(x.size(0)), batch_size):
                x_batch = x[batch_idx, :].to(device)
                y_batch = y[batch_idx, :].to(device)

                _yhat.append(model(x_batch))
                _y.append(y_batch)
        return torch.cat(_yhat, dim=0), torch.cat(_y, dim=0)
    def pretrain_(self, crit, target_crit, batch_size=256, lr=1e-3, epochs=100, use_cuda=True, verbose=True, report_metric=None):
        '''
        in-place model pre-training on source dataset
        '''
        if torch.cuda.is_available() & use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        if verbose: print('using device:', device)

        self.predictor.train().to(device)
        optim = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        for i in range(epochs):
            losses = []
            for idx_batch in torch.split(torch.randperm(self.x_train.size(0)), batch_size):
                x = self.x_train[idx_batch, :]
                y = self.y_train[idx_batch, :]
                x,y = myto(x,y,device)
                optim.zero_grad()
                yhat = self.predictor(x)
                loss = crit(yhat, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())
                if report_metric is not None:
                    metric = report_metric(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
                else:
                    metric = -666
            if verbose: print(f'epoch: {i} | loss: {np.mean(losses):.4f} | metric: {metric:0.4f}', end='\r')
    def _get_endog_and_marginal(self, x, y, val_model):
        '''predict the `val_model` yhat and concat with y; input into estimator'''

        if self.problem == 'classification':
            y_onehot = torch.nn.functional.one_hot(y.type(dtype=torch.long), num_classes=self.num_classes).type(
                torch.float).view(x.size(0), -1)
        else:
            y_onehot = y

        if self.include_marginal:
            with torch.no_grad():
                yhat_logits = val_model(x)
                yhat_valid = torch.softmax(yhat_logits, dim=-1)

            # marginal calculation as done in DVRL code line 419-421 (https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py)
            if self.problem == 'classification':
                marginal = torch.abs(y_onehot - yhat_valid)
            else:
                marginal = torch.abs(y_onehot - yhat_valid) / y_onehot

        # alternative option: don't include marginal; for sensitivity analysis
        else:
            marginal = torch.zeros_like(y_onehot)

        return torch.cat((y_onehot, marginal), dim=1)


    def _get_grad(self, loss, params):
        ''''''
        dl = torch.autograd.grad(loss, params, create_graph=True)
        dl= torch.cat([x.view(-1) for x in dl])
        return dl
    def eval_estimator(self, estimator, val_model):
        estimator.eval()
        with torch.no_grad():
            x = self.x_train.to('cuda')
            y = self.y_train.to('cuda')
            inp = self._get_endog_and_marginal(x=x, y=y, val_model=val_model)
            p = estimator(x=x, y=inp)

        return p.detach().cpu().numpy()


    def run(self, crit,target_crit, iter=500, num_epochs=100, batch=1000, num_restarts=1,
            estim_lr=1e-2, pred_lr=1e-2, moving_average_window=20, fix_baseline=False, use_cuda=True, target_batch_size=512, source_batch_size=512,
            save_dir='./meta_pg_results/', noise_labels=None):
        '''
        train the estimator model

        args:
            perf_metric                     metric to optimize, current options: ["mse", "bce", "acc", "auroc"]
            crit_pred                       predictor loss criteria, NOTE: must not have an reduction, e.g., out.size(0) == x.size(0); sample loss must be multiplied by bernoulli sampled value ([0,1])
            outer_iter                      number of estimator epochs to train
            inner_iter                      number of iterations to train predictor model at each estimator step
            outer_batch                     outer loop batch size; Bs
            inner_batch                     inner loop batch size; Bp
            estim_lr                        estimator learning rate
            pred_lr                         predictor learning rate
            moving_average_window           moving average window of the baseline
            entropy_beta                    starting entropy weight
            entropy_decay                   exponential decay rate of entropy weight
            fix_baseline                    whether to use a fixed baseline or moving average window
            use_cuda                        whether to use GPU if available

        output
            data values                     probability of inclusion [0,1]; shape: (nsamples,)
        '''

        if torch.cuda.is_available() & use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        nn = 0


        grad_params = [n for n,_ in self.predictor.named_parameters()]
        # place holder for vals

        # for marginal calc
        val_model = self.fit(model=copy.deepcopy(self.predictor), x=self.x_valid, y=self.y_valid, crit=crit,
                             iters=100, use_cuda=use_cuda, lr=pred_lr, batch_size=256)
        self.val_model = val_model
        print(f"val_model train finished")
        estimator = self.estimator.to(device).train()
        val_model = val_model.to(device).eval()

        model = self.predictor



        for i in range(num_epochs):
            model.reset_parameters()
            opt = torch.optim.Adam(model.parameters(), lr=pred_lr)

            losses = []
            reward_list = []
            kk = 0
            batches = torch.split(torch.randperm(self.x_valid.size(0)), target_batch_size)
            for idx_target in batches:
                estimator.reset_parameters()
                est_optim = torch.optim.Adam(estimator.parameters(), lr=estim_lr, weight_decay=0)

                x_target = self.x_valid[idx_target, :]
                y_target = self.y_valid[idx_target, :]
                self.predictor.train()
                x_target, y_target = myto(x_target, y_target, device)

                # step 1: compute target/validation gradient
                yhat_target = self.predictor(x_target)
                val_loss = crit(yhat_target, y_target)
                grad_target = self._get_grad(val_loss, [p for n, p in self.predictor.named_parameters() if n in grad_params])


                if hasattr(model, "turn_batchnorm_off_"): model.turn_batchnorm_off_()


                fmodel, params, buffers = make_functional_with_buffers(model)
                ft_compute_sample_grad = get_per_sample_grad_func(target_crit, fmodel)
                for _ in range(iter):
                    with torch.no_grad():
                        # outer_idxs = torch.randint(0, self.x_train.size(0), size=(outer_batch,))
                        outer_idxs = torch.randperm(self.x_train.size(0))[:batch]

                        x = self.x_train[outer_idxs]
                        y = self.y_train[outer_idxs]
                        x = x.to(device)
                        y = y.to(device)  # .view(y.size(0),-1)
                        inp = self._get_endog_and_marginal(x=x, y=y, val_model=val_model)

                    p = estimator(x=x, y=inp).view(-1, 1)

                    dist = torch.distributions.Bernoulli(probs=p)
                    s = dist.sample()

                    #  select only s=1 train data ; speed up inner loop training
                    s_idx = s.nonzero(as_tuple=True)[0]

                    xs = x[s_idx].detach()
                    ys = y[s_idx].detach()

                    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, xs, ys)


                    batch_grads = torch.cat([_g.reshape(-1) for _g, (n, p) in
                                             zip(ft_per_sample_grads, model.named_parameters()) if
                                             n in grad_params])

                    sim = torch.nn.CosineSimilarity(dim=1)(grad_target.unsqueeze(0), batch_grads).detach()
                    reward_list.append(sim.cpu().item())
                    exploration_weight = 1e3
                    exploration_threshold = 0.9
                    explore_loss = exploration_weight * (
                            max(p.mean() - exploration_threshold, 0) + max((1 - exploration_threshold) - p.mean(),
                                                                           0))

                    # NOTE: this is equivalent to torch.sum(s*torch.log(p) + (1-s)*torch.log(1-p))
                    log_prob = dist.log_prob(s).sum()

                    # update estimator params
                    est_optim.zero_grad()
                    loss = -sim * log_prob + explore_loss
                    loss.backward()
                    est_optim.step()

                    with torch.no_grad():

                        if noise_labels is not None:
                            iii = outer_idxs.detach().cpu().numpy()
                            ppp = 1 - p.detach().cpu().numpy()
                            auc = roc_auc_score(noise_labels[iii], ppp)
                            n_corr_sel = int(noise_labels[iii][s_idx.detach().cpu().numpy()].sum())
                        else:
                            auc = -1
                            n_corr_sel = -1

                        print(
                            f'iter: {_} || reward: {sim.cpu().item():.4f} || avg_reward: {sum(reward_list) / len(reward_list):.4f}|| mean prob: {p.mean().cpu().item():.2f} || max prob: {p.max().cpu().item():.2f}|| min prob: {p.min().cpu().item():.2f} || noise auc: {auc:.2f}|| crpt/tot: {n_corr_sel}/{s_idx.size(0)} [{n_corr_sel / s_idx.size(0):.2f}]',
                            end='\r')

                data_vals = self.eval_estimator(estimator, val_model)
                np.save(f'.//results//data_value_iter={nn}', data_vals)
                print(f"save_success {nn}")
                nn += 1
                opt.zero_grad()
                val_loss.backward()
                opt.step()
                losses.append(val_loss.item())



    def agg(self, path, reduction='mean'):
        '''aggregate all data values in path directory'''
        files = [x for x in listdir(path) if 'data_value_' in x]

        if reduction == 'none':
            return np.stack([np.load(f'{path}/{f}').ravel() for f in files], axis=1)

        elif reduction == 'quantile':
            x = None
            for f in files:
                xx = np.load(f'{path}/{f}')
                r = percentileofscore(xx, xx, kind='rank')
                if x is None:
                    x = r
                else:
                    x += r

            return x / len(files)

        elif reduction == 'mean':
            # NOTE: more memory efficient for large datasets
            x = np.load(f'{path}/{files[0]}')
            for f in files[1:]:
                x += np.load(f'{path}/{f}')
            return x / len(files)

    def clean(self, path):
        '''removes path'''
        rmtree(path)

def myto(x_valid, y_valid, device):
    ''''''
    if torch.is_tensor(x_valid):
        x_valid, y_valid = x_valid.to(device), y_valid.to(device)
    else:
        y_valid = y_valid.to(device)
        x_valid = [el.to(device) for el in x_valid]

    return x_valid, y_valid


def get_per_sample_grad_func(crit, fmodel):
    def compute_loss_stateless_model(params, buffers, sample, target):


        predictions = fmodel(params, buffers, sample)

        loss = crit(predictions, target)
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)

    return ft_compute_grad

