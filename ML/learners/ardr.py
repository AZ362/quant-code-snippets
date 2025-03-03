import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.linear_model import ARDRegression
import os
from optuna.trial import TrialState
from StrategySphere.ml.utils import utils_ml
import copy

class ARRegression():
	def __init__(self, dir_current, symbol, time_horizon, obj_dataset, optimisation_metric, dict_config):
		self.model_name = 'ml_ardr'
		self.dir_current = dir_current
		self.symbol = symbol
		self.time_horizon = time_horizon
		self.obj_dataset = obj_dataset
		self.best_metric = optimisation_metric
		self.dict_config = dict_config
		self.dir_models, self.dir_results, self.path_db_training_results, self.dir_reports, self.dir_metadata = utils_ml.get_learner_dirs(dir_current, symbol, time_horizon, self.model_name)
		self.df_training_info = None
		self.best_model = None
		self.best_model_metric = None
	
		# save the best ts features for this model
		if dict_config['train']['find_best_features']:
			# best_features = utils_ml.get_ts_features(self.model_name, obj_dataset)
			best_features = utils_ml.get_best_input_features(obj_dataset.X_train, obj_dataset.y_train_target)
			self.obj_dataset = copy.deepcopy(obj_dataset)
			self.obj_dataset = utils_ml.update_ml_dataset_with_best_features(self.obj_dataset, best_features)

			# save the best ts features for this model
			path_metadata = os.path.join(self.dir_metadata, f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_metadata_{self.dict_config['train']['started_at']}.json")
			utils_ml.save_training_metadata(self.model_name, self.obj_dataset, path_metadata)
		else:
			path_metadata = os.path.join(self.dir_metadata, f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_metadata_{self.dict_config['train']['started_at']}.json")
			utils_ml.save_training_metadata(self.model_name, self.obj_dataset, path_metadata)
		
		# utils_ml.save_input_features_as_csv(obj_dataset.best_input_features, os.path.join(dir_input_features, f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_{self.model_name}_input_features.csv"))

	def train(self, optuna_trials):
		self.direction=utils_ml.get_optimisation_direction(self.best_metric)
		
		# run optuna to find best model
		study = optuna.create_study(direction=self.direction)
		study.optimize(self.objective, n_trials=optuna_trials)

		# save all trained models hyperparameters, and their corresponding backtest and forwardtest stats in a db
		utils_ml.upload_training_results(self.df_training_info, self.dir_models, self.dir_metadata, self.symbol, self.time_horizon)
		utils_ml.save_df_as_db(self.path_db_training_results, self.model_name, self.df_training_info)

	def get_hyperparameters(self, trial):
		param_dict = {
			'alpha_1': trial.suggest_loguniform('alpha_1', 1e-10, 1.0),
			'alpha_2': trial.suggest_loguniform('alpha_2', 1e-10, 1.0),
			'lambda_1': trial.suggest_loguniform('lambda_1', 1e-10, 1.0),
			'lambda_2': trial.suggest_loguniform('lambda_2', 1e-10, 1.0),
			'compute_score': trial.suggest_categorical('compute_score', [True, False]),
			'copy_X': trial.suggest_categorical('copy_X', [True, False]),
			'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
			'n_iter': trial.suggest_int('n_iter', 100, 1000),
			'threshold_lambda': trial.suggest_float('threshold_lambda', 1e-5, 1.0),
			'tol': trial.suggest_float('tol', 1e-5, 1e-2)
		}


		for previous_trial in trial.study.trials:
			if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
				raise optuna.exceptions.TrialPruned()

		return param_dict

	def objective(self, trial):
		# param_dict = {'trial_number': trial.number}

		# get hyperparameters for this trial
		param_dict_tmp = self.get_hyperparameters(trial)

		# create model with current trial's hyperparameters
		model = ARDRegression(**param_dict_tmp)

		try:
			# train model
			model.fit(self.obj_dataset.X_train, self.obj_dataset.y_train_target)
			
			trial_model_name = f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_{self.model_name}_{trial.number}_1_{self.dict_config['train']['started_at']}"
			param_dict = {'model_name': trial_model_name, 'learner': self.model_name, 'time_horizon': self.time_horizon, 'trial_number': trial.number}
			
			# update params dict with current trial's model hyperparameters
			param_dict.update(param_dict_tmp)

			# perform backtest, forwardtest, and get stats
			df_ledger_bt, df_ledger_ft, predictions_bt, balance_bt, pnl_percent_bt = utils_ml.get_btft_stats(self.model_name, model, self.obj_dataset, param_dict, self.dict_config)

			# current trial number
			# trial_model_name = f"{self.symbol}_{self.time_horizon}_{self.obj_dataset.exchange_name}_{self.obj_dataset.trained_until}_{self.model_name}_{trial.number}"
			utils_ml.save_trial_result(self.model_name, model, self.obj_dataset, trial_model_name, df_ledger_bt, df_ledger_ft, self.dir_models, self.dir_results, self.dir_reports, self.dict_config)

			# append current trial's params in the dataframe
			self.df_training_info = utils_ml.append_training_info_df(param_dict, self.df_training_info)
			
			# get current trial's metric
			metric = utils_ml.get_current_trial_metric(self.best_metric, self.obj_dataset, predictions_bt, df_ledger_bt, balance_bt, pnl_percent_bt, self.obj_dataset.time_horizon_minutes, self.dict_config)
			return metric
		except Exception as e:
			# print(f"An error occurred: {e}")
			return -100
