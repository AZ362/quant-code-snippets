import numpy as np
import optuna
from StrategySphere.common import utils_common
from StrategySphere.ml.utils import utils_ml
# import ast
from StrategySphere.ml.learners import abr, block_rnn, tft, tide, nhits, etr, mlpr, bnbc, knc, svc, tcn, tm, sgdr, ardr

optuna.logging.set_verbosity(optuna.logging.WARNING)

class BaseLearner():
	def __init__(self, dir_current, symbol, time_horizon, obj_dataset, dict_config):
		self.dir_current = dir_current
		self.symbol = symbol
		self.time_horizon = time_horizon
		self.obj_dataset = obj_dataset
		self.dict_config = dict_config
		
		# self.optimisation_metric = utils.get_str('train', 'optimisation_metric')
		self.optimisation_metric = dict_config['train']['optimisation_metric']
		utils_common.validate_str(self.optimisation_metric, utils_ml.get_allowed_optimisation_metric(), 'optimisation_metric')

		# self.optuna_trials_dict = utils.get_optuna_trials_per_model()
		self.optuna_trials_dict = dict_config['train']['optuna_trials_per_model']
		# self.optuna_trials_dict = ast.literal_eval(optuna_trials_str)

		# print(obj_dataset.X_backtest)
		# print(obj_dataset.X_backtest.index[0])

		# ### get 1m data that will be used for backtesting and forwardtesting
		# self.index_open, self.index_high, self.index_low, self.index_datetime, self.np_1m = utils_ml.get_min_data_for_bt(exchange_name, self.symbol, df_backtest.index[0])
	
	# def get

	def train(self, model_name):
		if model_name == 'abr':
			self.obj_learner = abr.ABRegressor(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'etr':
			self.obj_learner = etr.ETRegressor(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'sgdr':
			self.obj_learner = sgdr.SGRegressor(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'ardr':
			self.obj_learner = ardr.ARRegression(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'mlpr':
			self.obj_learner = mlpr.MLRegressor(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'bnbc':
			self.obj_learner = bnbc.BNBClassifier(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'knc':
			self.obj_learner = knc.KNClassifier(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'svc':
			self.obj_learner = svc.SVClassifier(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'brnn':
			self.obj_learner = block_rnn.BlockRNN(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'tft':
			self.obj_learner = tft.TemporalFusionTransformer(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'tide':
			self.obj_learner = tide.TimeseriesDenseEncoder(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'nhits':
			self.obj_learner = nhits.NHiTS(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'tcn':
			self.obj_learner = tcn.TemporalConvolutionalNetwork(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		elif model_name == 'tm':
			self.obj_learner = tm.TSTransformerModel(self.dir_current, self.symbol, self.time_horizon, self.obj_dataset, self.optimisation_metric, self.dict_config)
		
		print('training...')
		self.obj_learner.train(self.optuna_trials_dict[model_name])

