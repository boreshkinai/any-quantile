# %matplotlib inline
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# import torch
# from torch import gluon

import datetime as dt

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

from gluonts.mx import NBEATSEnsembleEstimator
from gluonts.mx import DeepAREstimator
from gluonts.mx import WaveNetEstimator

from gluonts.mx import LSTNetEstimator
from gluonts.mx import TransformerEstimator

from gluonts.mx import TemporalFusionTransformerEstimator
from gluonts.torch.model.deep_npts import DeepNPTSEstimator
from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

from gluonts.mx import DeepTPPEstimator
from gluonts.mx import DeepVARHierarchicalEstimator

from gluonts.torch.model.deep_npts import DeepNPTSEstimator

mode = 'mlp'  # 'wave'#'mlp' 'nbeats' 'wave' 'deepar'
modee = ['mlp', 'nbeats', 'wave', 'deepar']
modee = ['mlp', 'wave', 'deepar', 'trans', 'lstm', 'tft', 'deeptpp']  # ,'trans',
modee = ['mlp', 'wave', 'deepar', 'trans', 'tft']
# modee=['deeptpp', 'lstm']#te nie dzialaja ('deeptpp', 'lstm')
# modee=['tft', 'deeptpp']#'[ 'nbeats']
# modee=['mlp',  'wave', 'deepar','trans','tft', 'deeptpp']
# modee=['deepar']
modee = ['deepar', 'mlp', 'wave', 'trans', 'tft']
# modee=['check']


N = 35  # 10  # number of time series
T = 365  # 100  # number of timesteps
prediction_length = 48  # 24
freq = "1H"
# custom_dataset = np.random.normal(size=(N, T))
# start = pd.Period("01-01-2019", freq=freq)  # can be different for each time series
start = pd.Period("01-01-2018", freq=freq)  # can be different for each time series

pr_all = np.zeros((48 * 364, 35))

ens_max = 5  # 5

# pr_q90 = np.zeros((48*365,35))
# pr_q_all=[]
MANDATORY_TESTING_QUANTS = [0.5, 0.001, 0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999]
NUM_OF_TESTING_QUANTS = 100  # apart from MANDATORY_TESTING_QUANTS

SEED_FOR_TESTING = 17;
stateOfRNG = None
now = dt.datetime.now()
hour = int(now.strftime('%H'))
if hour != 23:  # we do not want to sample always the same validation series
    dayOfYear = int(now.strftime('%j'))
    SEED_FOR_TESTING += dayOfYear  # when sampling for validation, all workers need to be synchronized

quants_series = []
for i in range(0, 35):
    # np.random.seed(SEED_FOR_TESTING) #pandas reportedly use np.random
    np.random.seed(SEED_FOR_TESTING + i)  # pandas reportedly use np.random
    stateOfRNG = np.random.get_state()
    np.random.set_state(stateOfRNG)
    # quants=MANDATORY_TESTING_QUANTS+list(np.random.uniform(size=NUM_OF_TESTING_QUANTS)) #so we generate random quants, but all workers are synchronized (generate the same quants)
    quants = MANDATORY_TESTING_QUANTS + list(np.random.uniform(
        size=NUM_OF_TESTING_QUANTS))  # so we generate random quants, but all workers are synchronized (generate the same quants)

    stateOfRNG = np.random.get_state()
    np.random.seed(None)  # randomize state again for all other endeavours
    # quants = ['%.3f' % elem for elem in quants]
    # quants_series.append(quants)
    quants_series.append(np.around(quants, decimals=3))
# aa= np.array([np.array(x) for x in quants_series])
# aa=aa.astype(np.float16)
# np.savetxt('progQ/Quants_'+str(modee)+'bd.txt', aa)
np.savetxt('progQ/Quants_' + 'bd.txt', np.array([np.array(x) for x in quants_series]))

# custom_dataset1=custom_dataset
# df1 = pd.read_csv('data/MHLV_small_data.csv')
df1 = pd.read_csv('data/MHLV.csv')  # bigdata
myda1 = df1.to_numpy()
myda = myda1[:, 1:]
custom_dataset_full = myda.T

from gluonts.dataset.common import ListDataset

import datetime as dt
import time as time

nowe = dt.datetime.now()
print(nowe.strftime("%Y-%m-%d %H:%M:%S"))
tstart = time.time()

custom_dataset = custom_dataset_full[:, :-364 * 24]

train_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset[:, :-24]],
    freq=freq,
)
for mode in modee:
    # pr_q_all = []
    pr_q_all = np.zeros((len(quants_series[0]), 48 * 364, 35))
    # pr_q_all_ens = []

    # pr_q_all_ens=np.zeros((111,48*365,35))
    pr_q_all_ens = np.zeros((len(quants_series[0]), 48 * 364, 35))
    tst = np.zeros((int(len(pr_q_all_ens[1])), 35))
    np.savetxt('progQ/Quants_' + mode + 'bd.txt', np.array([np.array(x) for x in quants_series]))

    for ens_num in range(0, ens_max):
        tstart = time.time()

        if (mode == 'mlp'):
            #     estimator = SimpleFeedForwardEstimator(
            #         num_hidden_dimensions=[10],#[10],
            #         prediction_length=48,#24#dataset.metadata.prediction_length,
            #         context_length=100,#100,
            #         trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
            #     )

            estimator = SimpleFeedForwardEstimator(num_hidden_dimensions=[10], prediction_length=48,
                                                   trainer=Trainer())

        elif (mode == 'nbeats'):
            # estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=24, meta_bagging_size= 10, trainer=Trainer(
            # add_default_callbacks=True, callbacks=None, clip_gradient=10.0, ctx=None, epochs=100, hybridize=True,
            # init='xavier', learning_rate=0.001, num_batches_per_epoch=50, weight_decay=1e-08), num_stacks= 30 )

            # estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=24,  trainer=Trainer(epochs=5))

            # estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=24, meta_bagging_size = 3,meta_context_length = [prediction_length * mlp for mlp in [1,7] ],meta_loss_function = ['sMAPE'], trainer=Trainer())
            # estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=24,
            #                                    meta_context_length=[prediction_length * mlp for mlp in [2, 7]],
            #                                    meta_loss_function=['sMAPE','MAPE'], trainer=Trainer())
            # estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=24,
            #                                    meta_context_length=[prediction_length * mlp for mlp in [ 7]],
            #                                    meta_loss_function=['sMAPE','MAPE'], trainer=Trainer())

            estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=48, meta_bagging_size=1,
                                                meta_context_length=[prediction_length * mlp for mlp in [7]],
                                                meta_loss_function=['sMAPE'], trainer=Trainer())

            # estimator = NBEATSEnsembleEstimator(freq=freq, prediction_length=24,
            #                                    meta_context_length=[prediction_length * mlp for mlp in [7]],
            #                                     trainer=Trainer())
        elif (mode == 'wave'):
            # estimator = WaveNetEstimator(
            # freq=freq, prediction_length=24, trainer= Trainer(
            # add_default_callbacks=True, callbacks=None, clip_gradient=10.0, ctx=None, epochs=200, hybridize=False,
            # init='xavier', learning_rate=0.01, num_batches_per_epoch=50, weight_decay=1e-08), embedding_dimension= 5, num_bins= 1024, hybridize_prediction_net= False, n_residue = 24, n_skip = 32,
            # n_stacks= 1, temperature= 1.0, act_type = 'elu', num_parallel_samples= 200,  batch_size= 32, negative_data = False)

            estimator = WaveNetEstimator(
                freq=freq, prediction_length=48, trainer=Trainer())
        elif (mode == 'deepar'):
            # estimator =DeepAREstimator(prediction_length=48, freq=freq, trainer=Trainer())#epochs=1
            estimator = DeepAREstimator(prediction_length=48, context_length=168, freq=freq, trainer=Trainer())
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, num_layers=3, context_length=168,trainer=Trainer())  #4,5675 , context_length=168,batch_size=10,lags_seq=168,time_features=168,
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, num_layers=3,batch_size=5,trainer=Trainer())  #5,91 , context_length=168,batch_size=10,lags_seq=168,time_features=168,
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, num_layers=5, batch_size=2,trainer=Trainer())  #5,06 , context_length=168,batch_size=10,lags_seq=168,time_features=168,
            # estimator = DeepAREstimator(prediction_length=48, freq=freq,context_length=168, num_layers=6, batch_size=10,trainer=Trainer())  #5,25 , context_length=168,batch_size=10,lags_seq=168,time_features=168,
            # estimator =DeepAREstimator(prediction_length=48, freq=freq, trainer=Trainer(learning_rate=0.0001))#4.0779
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, trainer=Trainer(learning_rate=0.00001))  #4,04
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, trainer=Trainer(learning_rate=0.001))  # 9,21
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, trainer=Trainer(learning_rate=0.0003))  #3.9603
            # estimator =DeepAREstimator(prediction_length=48, freq=freq, num_layers=1, trainer=Trainer())#4.14

            # estimator = DeepAREstimator(prediction_length=48, freq=freq,context_length = prediction_length *7, trainer=Trainer())
            # estimator = DeepAREstimator(prediction_length=48, freq=freq, context_length=prediction_length * 3.5,trainer=Trainer())


        elif (mode == 'lstm'):

            # gluonts.model.lstnet.LSTNetEstimator(
            #     prediction_length: int, context_length: int, num_series: int, skip_size: int, ar_window: int, channels: int, lead_time: int = 0, kernel_size: int = 6, trainer: gluonts.mx.trainer._base.Trainer = gluonts.mx.trainer._base.Trainer(
            #     add_default_callbacks=True, callbacks=None, clip_gradient=10.0, ctx=None, epochs=100, hybridize=True,
            #     init="xavier", learning_rate=0.001, num_batches_per_epoch=50, weight_decay=1e-08), dropout_rate:
            # typing.Optional[float] = 0.2, output_activation: typing.Optional[
            #     str] = None, rnn_cell_type: str = 'gru', rnn_num_cells: int = 100, rnn_num_layers: int = 3, skip_rnn_cell_type: str = 'gru', skip_rnn_num_layers: int = 1, skip_rnn_num_cells: int = 10, scaling: bool = True, train_sampler:
            # typing.Optional[gluonts.transform.sampler.InstanceSampler] = None, validation_sampler: typing.Optional[
            #     gluonts.transform.sampler.InstanceSampler] = None, batch_size: int = 32, dtype: typing.Type = <

            estimator = LSTNetEstimator(prediction_length=48, context_length=48, num_series=55, skip_size=1,
                                        ar_window=1, channels=1, trainer=Trainer())

        elif (mode == 'trans'):

            # class gluonts.model.transformer.TransformerEstimator(freq: str, prediction_length: int
            #
            # , context_length: Optional[
            #     int] = None, trainer: gluonts.mx.trainer._base.Trainer = gluonts.mx.trainer._base.Trainer(
            #     add_default_callbacks=True, callbacks=None, clip_gradient=10.0, ctx=None, epochs=100, hybridize=True,
            #     init='xavier', learning_rate=0.001, num_batches_per_epoch=50,
            #     weight_decay=1e-08), dropout_rate: float = 0.1, cardinality: Optional[List[
            #     int]] = None, embedding_dimension: int = 20, distr_output: gluonts.mx.distribution.distribution_output.DistributionOutput = gluonts.mx.distribution.student_t.StudentTOutput(), model_dim: int = 32, inner_ff_dim_scale: int = 4, pre_seq: str = 'dn', post_seq: str = 'drn', act_type: str = 'softrelu', num_heads: int = 8, scaling: bool = True, lags_seq:
            # Optional[List[int]] = None, time_features: Optional[List[Callable[[
            #                                                                       pandas.core.indexes.period.PeriodIndex], numpy.ndarray]]] = None, use_feat_dynamic_real: bool = False, use_feat_static_cat: bool = False, num_parallel_samples: int = 100, train_sampler:
            # Optional[gluonts.transform.sampler.InstanceSampler] = None, validation_sampler: Optional[
            #     gluonts.transform.sampler.InstanceSampler] = None, batch_size: int = 32)[sou
            estimator = TransformerEstimator(prediction_length=48, freq=freq, trainer=Trainer())

        elif (mode == 'tft'):
            estimator = TemporalFusionTransformerEstimator(prediction_length=48, freq=freq,
                                                           trainer=Trainer())  # works
        elif (mode == 'deeptpp'):
            estimator = DeepTPPEstimator(prediction_interval_length=48, context_interval_length=24, num_marks=1,
                                         freq=freq,
                                         trainer=Trainer(
                                             hybridize=False))  # needs additiona parameters prediction_interval_length context_interval_length num_marks

        elif (mode == 'check'):
            estimator = TemporalFusionTransformerEstimator(prediction_length=48, freq=freq,
                                                           trainer=Trainer(epochs=5))  # works
            # estimator = DeepNPTSEstimator(prediction_length=48, freq=freq,context_length=48)  # torch,works
            # estimator =  MQF2MultiHorizonEstimator(prediction_length=48, freq=freq)  # torch,works

            # estimator = DeepTPPEstimator(prediction_interval_length=48,context_interval_length=24,num_marks=1, freq=freq, trainer=Trainer(epochs=5,hybridize=False))  # needs additiona parameters prediction_interval_length context_interval_length num_marks
            # estimator = DeepVARHierarchicalEstimator(s=train_ds,prediction_length=48, freq=freq, trainer=Trainer(epochs=5))  # needs additiona parameters s, target_dim

        predictor = estimator.train(train_ds)

        for day in range(364, 0, -1):  # for day in range(364,-1,-1):
            custom_dataset = custom_dataset_full[:, :-day * 24]
            tst[(364 - day) * 48:(364 - day + 1) * 48, :] = custom_dataset_full[:,
                                                            (len(custom_dataset_full[0]) - day * 24) - 24:(
                                                                        len(custom_dataset_full[0]) - day * 24 + 24)].T

            if day == 0:
                custom_dataset = custom_dataset_full

            # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
            train_ds = ListDataset(
                [{"target": x, "start": start} for x in custom_dataset[:, :-24]],
                freq=freq,
            )
            # test dataset: use the whole dataset, add "target" and "start" fields
            test_ds = ListDataset(
                [{"target": x, "start": start} for x in custom_dataset], freq=freq
            )

            predictions = predictor.predict(train_ds)
            predictions_num = list(predictions)

            pr = np.zeros((48, 35))
            # pr_q = np.zeros((48, 35))
            for x in range(0, 35):
                # pr[:,x]=forecasts[x].mean
                pr[:, x] = predictions_num[x].mean
                # pr_q[:,x] =predictions_num[x].quantile(0.2)

                # aa=predictions_num[x].quantile(quants_series[x])#check

            # pr_all[(364-day)*48:(364-day+1)*48,:]=pr
            pr_all[(364 - day) * 48:(364 - day + 1) * 48, :] = pr_all[(364 - day) * 48:(364 - day + 1) * 48,
                                                               :] + pr  # ensemble

            for q in range(0, len(quants_series[x])):
                pr_q = np.zeros((48, 35))
                for x in range(0, 35):
                    pr_q[:, x] = predictions_num[x].quantile(quants_series[x][q])
                    # pr_q[:, x] = round(predictions_num[x].quantile(quants_series[x][q]),1)
                    pr_q_all[q, (364 - day) * 48:(364 - day + 1) * 48,
                    :] = pr_q  # pr_q_all[q][(364-day)*48:(364-day+1)*48,:]#+pr_q



                if day == 1:
                    pr_q_all_ens[q] = pr_q_all_ens[q] + pr_q_all[q]


                if (day == 1 and ens_num == ens_max - 1):  # if day==0:
                    if q == 0:
                        pr_all = pr_all / ens_max
                        ape=abs(pr_all-tst)/tst*100

                    pr_q_all_ens[q] = pr_q_all_ens[q] / ens_max

                    # np.savetxt('progQ/Prog_Quant_med_GluonTS_big_data_' + mode + '.txt', pr_all)
                    ##pr_q_all[(364 - day) * 48:(364 - day + 1) * 48, :] = pr_q
                    ##np.savetxt('progQ/Prog_Quant'+str(q)+'_GluonTS_big_data_'+mode+'.txt', np.around(pr_q_all[q], decimals=2))
                    # np.savetxt('progQ/Prog_Quant'+str(q)+'_GluonTS_big_data_'+mode+'.txt', np.around(pr_q_all_ens[q], decimals=2))


        elapsed = time.time() - tstart
        print(elapsed)

        print(mode)
        nowe = dt.datetime.now()
        print(nowe.strftime("%Y-%m-%d %H:%M:%S"))

    forec_q = pr_q_all_ens.transpose(2, 1, 0)
    forec_q = forec_q.reshape(forec_q.shape[0], forec_q.shape[1] * forec_q.shape[2]).T


    tst = np.repeat(tst, repeats=len(quants_series[0]), axis=0)


    pbLosses = []
    normPbLosses = []

    for country in range(0, 35):
        #pbLosses.append([])  # the last one for all quantiles
        #normPbLosses.append([])


        qq = np.tile(quants_series[country], (len(pr_q_all_ens[1]), 1))
        qq=qq.reshape(qq.shape[0]*qq.shape[1],1)


        seriesByWorker_df = pd.DataFrame(0, index=np.arange(364 * 48 * len(quants_series[0])),
                                         columns=['quants', 'actuals', 'aggForec'])

        seriesByWorker_df['quants'] = qq  # [:, country]
        seriesByWorker_df['actuals'] = tst[:, country]
        seriesByWorker_df['aggForec'] = forec_q[:, country]  # pr_q_all_ens[q,val, country]
        seriesByWorker_df.to_pickle("H:/Q/pickle/v2/" + mode + '_' + str(country) + ".pickle")


        diff = seriesByWorker_df['actuals'] - seriesByWorker_df['aggForec']
        quant = seriesByWorker_df['quants']
        pbLoss = np.maximum(diff * quant, diff * (quant - 1))
        # assert np.all(pbLoss>0)#
        #pbLosses[-1].append(np.nanmean(pbLoss))
        #normPbLosses[-1].append(np.nanmean(pbLoss) / np.nanmean(seriesByWorker_df['actuals']))
        pbLosses.append(np.nanmean(pbLoss))
        normPbLosses.append(np.nanmean(pbLoss) / np.nanmean(seriesByWorker_df['actuals']))

    print('MAPE:', f'{ np.mean(ape):.4}')
    #print('Approx CRPS:', f'{2 * np.mean(pbLosses[-1]):.4}')
    #print('Avg normalized quantile loss [%]:', f'{100 * np.nanmean(normPbLosses[-1]):.4}')
    print('Approx CRPS:', f'{2 * np.mean(pbLosses):.4}')
    print('Avg normalized quantile loss [%]:', f'{100 * np.nanmean(normPbLosses):.4}')
