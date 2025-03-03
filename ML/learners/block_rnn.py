import logging
logging.basicConfig(level=logging.WARNING)
pl_logger = logging.getLogger('pytorch_lightning')
pl_logger.setLevel(logging.WARNING)
rank_zero_logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
rank_zero_logger.setLevel(logging.WARNING)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from darts.models import BlockRNNModel
import os
import optuna
from optuna.trial import TrialState
import torch
from StrategySphere.ml.utils import utils_ml
import copy

class BlockRNN():
	def __init__(self, dir_current, symbol, time_horizon, obj_dataset, optimisation_metric, dict_config):
		self.model_name = 'ts_brnn'
		self.dir_current = dir_current
		self.time_horizon = time_horizon
		self.best_metric = optimisation_metric
		self.symbol = symbol
		self.dict_config = dict_config
		self.df_training_info = None
		self.best_model = None
		self.best_model_metric = None
		self.df_training_info = None

		# get directories for this model
		self.dir_models, self.dir_results, self.path_db_training_results, self.dir_reports, self.dir_metadata = utils_ml.get_learner_dirs(dir_current, symbol, time_horizon, self.model_name)

		# get torch device, i.e., cpu or gpu
		self.pl_trainer_kwargs = utils_ml.get_torch_device(dict_config)

		if dict_config['train']['find_best_features']:
			# get best ts features based on permutation method, and update the obj_dataset accordingly
			best_features = utils_ml.get_ts_features(self.model_name, obj_dataset)
			self.obj_dataset = copy.deepcopy(obj_dataset)
			self.obj_dataset = utils_ml.update_dataset_with_best_features(self.obj_dataset, best_features)

			# save the best ts features for this model
			path_metadata = os.path.join(self.dir_metadata, f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_metadata_{self.dict_config['train']['started_at']}.json")
			utils_ml.save_training_metadata(self.model_name, self.obj_dataset, path_metadata)
			# utils_ml.save_input_features_as_csv(best_features, os.path.join(dir_input_features, f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_{self.model_name}_input_features.csv"))
		else:
			self.obj_dataset = obj_dataset
			path_metadata = os.path.join(self.dir_metadata, f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_metadata_{self.dict_config['train']['started_at']}.json")
			utils_ml.save_training_metadata(self.model_name, self.obj_dataset, path_metadata)
			# utils_ml.save_input_features_as_csv(obj_dataset.best_input_features, os.path.join(dir_input_features, f"{self.symbol}_{self.obj_dataset.exchange_name}_{self.time_horizon}_{self.obj_dataset.trained_until}_{self.model_name}_input_features.csv"))


	def train(self, optuna_trials):
		self.direction=utils_ml.get_optimisation_direction(self.best_metric)
		
		# run optuna to find the best model
		study = optuna.create_study(direction=self.direction)
		study.optimize(self.objective, n_trials=optuna_trials)

		# save all trained models hyperparameters, and their corresponding backtest and forwardtest stats in a db
		utils_ml.upload_training_results(self.df_training_info, self.dir_models, self.dir_metadata, self.symbol, self.time_horizon)
		utils_ml.save_df_as_db(self.path_db_training_results, self.model_name, self.df_training_info)

	def get_hyperparameters(self, trial):
		#features= trial.suggest_categorical('features',['group1'], ['group2'], ['group3'], ['group4'], ['group5'])
		model = trial.suggest_categorical('model', ['RNN', 'LSTM', 'GRU'])
		hidden_dim = trial.suggest_int('hidden_dim', 10, 500)
		n_rnn_layers = trial.suggest_int('n_rnn_layers', 1, 5)
		hidden_fc_sizes = trial.suggest_categorical('hidden_fc_sizes', [[50], [100], [50, 50], [100, 100]])
		dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

		# Training configuration parameters
		batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
		n_epochs = trial.suggest_int('n_epochs', 20, 100)
		# n_epochs = 3
		nr_epochs_val_period = trial.suggest_int('nr_epochs_val_period', 1, 10)
		input_chunk_length = trial.suggest_int('input_chunk_length', 2, 30)

		# Optimizer and learning rate scheduler
		optimizer_cls = trial.suggest_categorical('optimizer_cls', [torch.optim.Adam, torch.optim.SGD])

		optimizer_kwargs = {}
		if optimizer_cls == torch.optim.SGD:
			optimizer_kwargs['lr'] = trial.suggest_loguniform('sgd_lr', 1e-5, 1e-1)
			optimizer_kwargs['momentum'] = trial.suggest_uniform('sgd_momentum', 0.0, 1.0)
		elif optimizer_cls == torch.optim.Adam:
			optimizer_kwargs['lr'] = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
			optimizer_kwargs['betas'] = (trial.suggest_uniform('adam_beta1', 0.85, 0.99),
										trial.suggest_uniform('adam_beta2', 0.95, 0.999))
			optimizer_kwargs['eps'] = trial.suggest_loguniform('adam_eps', 1e-8, 1e-6)

		list_parameters_names = ['input_chunk_length', 'model', 'hidden_dim', 'n_rnn_layers',	'hidden_fc_sizes', 'dropout', 'optimizer_cls', 'batch_size', 'n_epochs', 'nr_epochs_val_period', 'optimizer_kwargs']
		list_parameters_values = [input_chunk_length, model, hidden_dim, n_rnn_layers, hidden_fc_sizes, dropout, optimizer_cls, batch_size, n_epochs, nr_epochs_val_period, optimizer_kwargs]
		param_dict = {list_parameters_names[i]: list_parameters_values[i] for i in range(len(list_parameters_names))}

		for previous_trial in trial.study.trials:
			if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
				raise optuna.exceptions.TrialPruned()

		return param_dict

	def objective(self, trial):
		# get model's hyperparameter
		param_dict_tmp = self.get_hyperparameters(trial)
		_, output_chunk_length = utils_ml.get_ts_in_out_length()
		input_chunk_length = param_dict_tmp['input_chunk_length']
		self.dict_config['input_chunk_length'] = input_chunk_length
		model = BlockRNNModel(#input_chunk_length=input_chunk_length,
        						output_chunk_length=output_chunk_length,
								random_state=42,
								pl_trainer_kwargs=self.pl_trainer_kwargs,
		  						**param_dict_tmp
								)
		
		# print(input_chunk_length)

		try:
			# print(model)
			model.fit(series=self.obj_dataset.darts_y_train_target, past_covariates=self.obj_dataset.darts_X_train)
			# print("input_chunk_length", input_chunk_length)
			
			trial_model_name = f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_{self.model_name}_{trial.number}_{input_chunk_length}_{self.dict_config['train']['started_at']}"
			param_dict = {'model_name': trial_model_name, 'learner': self.model_name, 'time_horizon': self.time_horizon, 'trial_number': trial.number}
			# update params dict with current trial's model hyperparameters
			param_dict.update(param_dict_tmp)

			# perform backtest, forwardtest, and get stats
			df_ledger_bt, df_ledger_ft, predictions_bt, balance_bt, pnl_percent_bt = utils_ml.get_btft_stats(self.model_name, model, self.obj_dataset, param_dict, self.dict_config)

			# current trial number
			utils_ml.save_trial_result(self.model_name, model, self.obj_dataset, trial_model_name, df_ledger_bt, df_ledger_ft, self.dir_models, self.dir_results, self.dir_reports, self.dict_config)

			# append current trial's params in the dataframe
			self.df_training_info = utils_ml.append_training_info_df(param_dict, self.df_training_info)
			# print(self.df_training_info)
			
			# get current trial's metric
			metric = utils_ml.get_current_trial_metric(self.best_metric, self.obj_dataset, predictions_bt, df_ledger_bt, balance_bt, pnl_percent_bt, self.obj_dataset.time_horizon_minutes, self.dict_config)
			return metric
		except Exception as e:
			# print(f"An error occurred: {e}")
			return -100
