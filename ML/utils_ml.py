import os
import warnings
import pandas as pd
import numpy as np
import torch
from StrategySphere.ml.backtest import backtest
import optuna
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import ast
import csv
from scipy import stats
import quantstats as qs
import sqlite3
from darts import TimeSeries
from darts.models import NHiTSModel, TFTModel, TiDEModel, BlockRNNModel, TransformerModel, TCNModel
import joblib
import math
import glob
import smbclient
import json
import datetime
import psycopg2
from psycopg2 import OperationalError
from psycopg2.extras import execute_values
from StrategySphere.common import utils_common
import random
import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_allowed_models():
	return ['abr', 'etr', 'mlpr', 'bnbc', 'knc', 'svc', 'tft', 'tide', 'nhits', 'brnn', 'tcn', 'tm', 'sgdr', 'ardr']

def get_allowed_optimisation_metric():
	return ['model_accuracy', 'pnl_accuracy', 'pnl', 'balance', 'r2', 'sharpe', 'sortino', 'drawdown', 'win_loss', 'recovery_factor']

def np_create_rolling_window(df, rw_size):
	# Create a sliding window view of the DataFrame values
	window_view = np.lib.stride_tricks.sliding_window_view(df.values, (rw_size, df.shape[1]))
	# Reshape the sliding window view to a 2D array
	new_arr = window_view.reshape(-1, df.shape[1] * rw_size)
	# Create a new DataFrame with the sliding window values
	new_df = pd.DataFrame(new_arr)
	new_df.columns = range(df.shape[1] * rw_size)

	# Copy the DateTime index from the original DataFrame to the new DataFrame
	new_df.index = df.index[rw_size - 1:]	

	return new_df

def validate_train_test_split(train_split_percent, backtest_split_percent, forwardtest_split_percent):
	if sum([train_split_percent, backtest_split_percent, forwardtest_split_percent]) != 100:
		raise ValueError("Train, backtest and forwardtest split percentages must sum to 100")
	if backtest_split_percent < 1:
		raise ValueError("Backtest split percentage must be at least 1%")
	if forwardtest_split_percent < 1:
		raise ValueError("Forwardtest split percentage must be at least 1%")
	
def get_train_test_split(df, train_split_percent, backtest_split_percent, forwardtest_split_percent):
	idx1 = int(len(df) * (train_split_percent / 100.0))
	idx2 = idx1 + int(len(df) * (backtest_split_percent / 100.0))

	# Split the DataFrame
	df1 = df.iloc[:idx1]
	df2 = df.iloc[idx1:idx2]
	df3 = df.iloc[idx2-3:]
	df4 = df.iloc[idx1:]

	return df1, df2, df3, df4	

# def create_override_dir(path):
# 	if not os.path.exists(path):
# 		os.makedirs(path)
# 	else:
# 		shutil.rmtree(path)
# 		os.makedirs(path)
# 	return path

# def create_if_not_exists(path):
# 	if not os.path.exists(path):
# 		os.makedirs(path)  

# def delete_dir(path):
# 	if os.path.exists(path) and os.path.isdir(path):
# 		shutil.rmtree(path)

# def delete_file(path):
# 	if os.path.exists(path):
# 		os.remove(path)

def create_models_results_dirs(dir_current):
	# utils_common.create_if_not_exists(os.path.join(dir_current, 'models'))
	# utils_common.create_if_not_exists(os.path.join(dir_current, 'results'))
	# utils_common.create_if_not_exists(os.path.join(dir_current, 'dbs'))
	# utils_common.create_if_not_exists(os.path.join(dir_current, 'reports'))
	# utils_common.create_if_not_exists(os.path.join(dir_current, 'metadata'))
	utils_common.create_if_not_exists(os.path.join(dir_current, 'models'))
	utils_common.create_if_not_exists(os.path.join(dir_current, 'results'))
	utils_common.create_if_not_exists(os.path.join(dir_current, 'dbs'))
	utils_common.create_if_not_exists(os.path.join(dir_current, 'reports'))
	utils_common.create_if_not_exists(os.path.join(dir_current, 'metadata'))

# def get_cuda_device():
# 	if torch.cuda.is_available():
	
# Function to apply the specified fill operations
def fill_zeros(df, column_name):

	df[column_name] = df[column_name].where(df[column_name] != 0.0, pd.NA)
	df[column_name].ffill(inplace=True)
	if pd.isna(df[column_name].iloc[0]):
		df[column_name] = df[column_name].bfill()

	# # Check if the first value is 0 and backward fill if true
	# if df[column_name].iloc[0] == 0:
	# 	df[column_name] = df[column_name].bfill()
	# # Forward fill to fill remaining zeros
	# df[column_name] = df[column_name].replace(0, np.nan).ffill()
	return df

def get_backtest_params(dict_config, time_horizon_minutes=10):
	# config = get_config_file()

	# starting_balance = get_float('backtest', 'starting_balance')
	starting_balance = dict_config['backtest']['starting_balance']
	utils_common.validate_float(starting_balance, 30, 10000000)

	# take_profit = get_float('backtest', 'take_profit')
	take_profit = dict_config['backtest']['take_profit']
	utils_common.validate_float(take_profit, 0.05, 100)

	# stop_loss = get_float('backtest', 'stop_loss')
	stop_loss = dict_config['backtest']['stop_loss']
	utils_common.validate_float(stop_loss, 0.05, 100)

	# transaction_fee = get_float('backtest', 'transaction_fee')
	transaction_fee = dict_config['backtest']['transaction_fee']
	utils_common.validate_float(transaction_fee, 0.0, 0.1)

	# leverage = get_float('backtest', 'leverage')
	leverage = dict_config['backtest']['leverage']
	utils_common.validate_float(leverage, 0.1, 125)

	# slippage = get_float('backtest', 'slippage')
	slippage = dict_config['backtest']['slippage']
	utils_common.validate_float(slippage, 0.0, 1)

	# buy_after_minutes = get_int('backtest', 'buy_after_minutes')
	buy_after_minutes = dict_config['backtest']['buy_after_minutes']
	utils_common.validate_int(buy_after_minutes, 0, int(time_horizon_minutes/2))

	return starting_balance, take_profit, stop_loss, buy_after_minutes, transaction_fee, leverage, slippage

def get_optimisation_direction(best_metric):

	if best_metric == 'model_accuracy' or best_metric == 'pnl_accuracy' or best_metric == 'pnl' or best_metric == 'balance' or best_metric == 'r2' or best_metric == 'sharpe' or best_metric == 'sortino' or best_metric == 'win_loss' or best_metric == 'recovery_factor' or best_metric == "drawdown":
		direction="maximize"
	# elif best_metric == "drawdown":
	# 	direction="minimize"

	return direction

def perform_backtest(obj_dataset, df_predictions, dict_config):
	starting_balance, take_profit, stop_loss, buy_after_minutes, transaction_fee, leverage, slippage = get_backtest_params(dict_config, time_horizon_minutes=obj_dataset.time_horizon_minutes)
	obj_backtest = backtest.Backtest(obj_dataset.obj_minute_data, df_predictions, starting_balance, take_profit, stop_loss, buy_after_minutes, transaction_fee, leverage, slippage)
	return obj_backtest.run()

def get_current_trial_metric(best_metric, obj_dataset, predictions, df_ledger, balance, pnl_percent, time_horizon_minutes, dict_config):
	if best_metric == "model_accuracy":
		actual_signs = np.sign(obj_dataset.y_backtest_direction)

		if "input_chunk_length" in dict_config:
			actual_signs = actual_signs.iloc[dict_config["input_chunk_length"]-1:].to_list()

		predicted_signs = np.sign(predictions)
		# print(len(predicted_signs), predicted_signs)
		return sum(predicted_signs == actual_signs) / len(predicted_signs)
	elif best_metric == "pnl":
		return pnl_percent
	elif best_metric == "balance":
		return balance

	# print(df_ledger)
	if 'datetime' not in df_ledger.columns:
		df_ledger['datetime'] = df_ledger.index
	df_type = 'bt'
	dict_stats = get_stats_from_ledger(df_ledger, time_horizon_minutes, df_type, dict_config)

	if best_metric == "pnl_accuracy":
		metric = dict_stats[f'{df_type}_num_profits'] / (dict_stats[f'{df_type}_num_profits']+dict_stats[f'{df_type}_num_losses'])
	elif best_metric == "r2":
		metric = dict_stats[f'{df_type}_r2']
	elif best_metric == "sharpe":
		metric = dict_stats[f'{df_type}_sharpe']
	elif best_metric == "sortino":
		metric = dict_stats[f'{df_type}_sortino']
	elif best_metric == "drawdown":
		metric = dict_stats[f'{df_type}_max_drawdown']
	elif best_metric == "win_loss":
		metric = dict_stats[f'{df_type}_win_loss_ratio']
	elif best_metric == "recovery_factor":
		metric = dict_stats[f'{df_type}_recovery_factor']
	# elif best_metric == "error":
	# 	metric = mean_squared_error(obj_dataset.np_y_test, predictions, squared=True)
		
	return metric

def get_best_model_metric(metric, best_model_metric, model, direction):
	if best_model_metric == None:
		return metric, model
	
	if direction == "minimize":
		if metric < best_model_metric:
			return metric, model
	elif direction == "maximize":	
		if metric > best_model_metric:
			return metric, model
	
	return None, None

def get_ts_in_out_length():
	input_chunk_length = 2
	output_chunk_length = 1

	return input_chunk_length, output_chunk_length
	
def get_ts_predictions(model, darts_series, darts_past_covariates, time_horizon_minutes, input_chunk_length=1):
	if input_chunk_length == 1:
		N, _ = get_ts_in_out_length()  
		N = N-1
	else:
		N = input_chunk_length - 1
	
	# print(input_chunk_length, N)

	predictions = []
	for i in range(len(darts_series) - N):
		start_time = darts_series.start_time() + pd.Timedelta(minutes=i * time_horizon_minutes)
		end_time = start_time + pd.Timedelta(minutes= N * time_horizon_minutes)
		# print(start_time, end_time)
		series_ = darts_series.slice(start_time, end_time)
		past_covariates_ = darts_past_covariates.slice(start_time, end_time)



		prediction = model.predict(n=1, series=series_, past_covariates=past_covariates_, verbose=False, show_warnings=False)
		predicted_value = prediction.univariate_values()[0]  
		timestamp_of_prediction = prediction.time_index[-1]
		# print(timestamp_of_prediction, predicted_value)
		if math.isnan(predicted_value):
			predictions.append(-1)
		else:	
			predictions.append(predicted_value)
		if i == 0:
			print(timestamp_of_prediction, predicted_value)
	print(N)
	print(len(predictions))
	print(predictions)
	# predictions = []
	# for i in range(len(darts_series)):
	# 	prediction = model.predict(n=1, series=darts_series[i], past_covariates=darts_past_covariates[i], verbose=False, show_warnings=False)
	# 	predicted_value = prediction.univariate_values()[0]  # Assuming a univariate TimeSeries
	# 	# timestamp_of_prediction = prediction.time_index[-1]
	# 	# print(timestamp_of_prediction, predicted_value)
	# 	predictions.append(predicted_value)
	
	return predictions

def get_best_input_features(X_train, y_train, threshold=50000, n_trials=5, cv_folds=3, n_jobs=1):
	print("Finding best features...")
	# import time
	# start = time.time()
	# Check if the dataset size exceeds the threshold
	if len(X_train) > threshold:
		# Use only the latest `threshold` length of the data
		X_train_sampled = X_train.iloc[-threshold:]
		y_train_sampled = y_train.iloc[-threshold:]
	else:
		# Use the full dataset
		X_train_sampled, y_train_sampled = X_train, y_train

	# Create and run the study with limited parallel execution
	study = optuna.create_study(direction='minimize')
	study.optimize(objective_input_features(X_train_sampled, y_train_sampled, cv_folds, n_jobs), n_trials=n_trials, n_jobs=n_jobs)  # Adjust the number of trials and enable controlled parallelism

	# Best trial
	trial = study.best_trial
	# Re-train model with best parameters to get feature importances
	best_params = trial.params
	cumulative_importance_threshold = best_params.pop('cumulative_importance_threshold', None)
	
	best_model = ExtraTreesRegressor(
		n_estimators=best_params['n_estimators'],
		max_depth=best_params['max_depth'],
		min_samples_split=best_params['min_samples_split'],
		random_state=42,
		n_jobs=n_jobs
	)
	best_model.fit(X_train, y_train)
	feature_importances = best_model.feature_importances_

	# Calculate cumulative importance and select best features based on the best threshold found
	indices = np.argsort(feature_importances)[::-1]
	cumulative_importance = np.cumsum(feature_importances[indices])
	num_features_needed = np.where(cumulative_importance >= cumulative_importance_threshold)[0][0] + 1
	
	# Get the names of the top important features
	top_features = X_train.columns[indices[:num_features_needed]]
	if 'close' not in top_features:
		top_features = np.append(top_features, 'close')

	# print(f"Time taken: {time.time() - start} seconds")
	# print(X_train.columns)
	# print(top_features.tolist())
	return top_features.tolist()
	

def objective_input_features(X_train, y_train, cv_folds, n_jobs):
	def objective(trial):
		# Suggest hyperparameters
		n_estimators = trial.suggest_int('n_estimators', 50, 100)
		max_depth = trial.suggest_int('max_depth', 10, 50)
		min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
		
		# Initialize and fit the ExtraTrees model to calculate feature importances
		model_for_importance = ExtraTreesRegressor(
			n_estimators=n_estimators,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			random_state=42,
			n_jobs=n_jobs
		)
		model_for_importance.fit(X_train, y_train)
		
		# Calculate and sort feature importances
		feature_importances = np.array(model_for_importance.feature_importances_)
		indices = np.argsort(feature_importances)[::-1]  # Sort in descending order
		
		# Determine the cumulative importance threshold dynamically
		cumulative_importance_threshold = trial.suggest_uniform('cumulative_importance_threshold', 0.70, 0.95)
		cumulative_importance = np.cumsum(feature_importances[indices])
		
		min_features_to_select = 5  # Adjust this based on your needs

		# Identify the number of features needed to reach the cumulative importance threshold
		num_features_needed = np.where(cumulative_importance >= cumulative_importance_threshold)[0]

		# If no features meet the threshold or fewer than min_features_to_select do, adjust the number
		if len(num_features_needed) == 0 or len(num_features_needed) < min_features_to_select:
			num_features_needed = min_features_to_select
		else:
			num_features_needed = num_features_needed[0] + 1  # Adjust for zero-indexing
		
		# Select the top important features based on the threshold
		top_features_indices = indices[:num_features_needed]
		X_train_refined = X_train.to_numpy()[:, top_features_indices]
		
		# Perform cross-validation with the refined feature set
		scores = cross_val_score(model_for_importance, X_train_refined, y_train, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=n_jobs)
		mse = -scores.mean()
		return mse
		
	return objective


def get_best_input_features_(X_train, y_train, threshold=5000):
	print("Finding best features...")
	# # X_train, X_test, y_train, y_test = train_test_split(df_train, df_target['target'], test_size=0.9, random_state=42, shuffle=False)
	# print(df_train)
	# print(df_target)
	# X_train, y_train = df_train, df_target['target']
	import time
	start = time.time()
	# Check if the dataset size exceeds the threshold
	if len(X_train) > threshold:
		# Use only the latest `threshold` length of the data
		print(f"Reducing dataset len from {len(X_train)} to {threshold}")
		X_train = X_train.iloc[-threshold:]
		y_train = y_train.iloc[-threshold:]
		print(len(X_train))
		print(len(y_train))

	# Create and run the study
	study = optuna.create_study(direction='minimize')
	# study.optimize(objective_input_features(X_train, y_train), n_trials=random.randint(5, 10))  # Adjust the number of trials as needed
	study.optimize(objective_input_features(X_train, y_train), n_trials=5, n_jobs=-1)  # Adjust the number of trials as needed

	# Best trial
	trial = study.best_trial
	# Re-train model with best parameters to get feature importances
	best_params = trial.params
	# Remove the 'cumulative_importance_threshold' parameter since it's not part of RandomForestRegressor
	cumulative_importance_threshold = best_params.pop('cumulative_importance_threshold', None)
	
	best_model = RandomForestRegressor(
		n_estimators=best_params['n_estimators'],
		max_depth=best_params['max_depth'],
		min_samples_split=best_params['min_samples_split'],
		random_state=42
	)
	best_model.fit(X_train, y_train)
	feature_importances = best_model.feature_importances_

	# Calculate cumulative importance and select best features based on the best threshold found
	indices = np.argsort(feature_importances)[::-1]
	cumulative_importance = np.cumsum(feature_importances[indices])
	num_features_needed = np.where(cumulative_importance >= cumulative_importance_threshold)[0][0] + 1
	
	# Get the names of the top important features
	top_features = X_train.columns[indices[:num_features_needed]]
	if 'close' not in top_features:
		top_features = np.append(top_features, 'close')

	print(f"Time taken: {time.time() - start} seconds")
	return top_features.tolist()
	
def objective_input_features_(X_train, y_train):
	def objective(trial):
		# Suggest hyperparameters
		n_estimators = trial.suggest_int('n_estimators', 50, 300)
		max_depth = trial.suggest_int('max_depth', 10, 100)
		min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
		
		# Initialize and fit the RandomForest model to calculate feature importances
		model_for_importance = RandomForestRegressor(
			n_estimators=n_estimators,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			random_state=42
		)
		model_for_importance.fit(X_train, y_train)
		
		# Calculate and sort feature importances
		feature_importances = np.array(model_for_importance.feature_importances_)
		indices = np.argsort(feature_importances)[::-1]  # Sort in descending order
		
		# Determine the cumulative importance threshold dynamically
		cumulative_importance_threshold = trial.suggest_uniform('cumulative_importance_threshold', 0.70, 0.95)
		cumulative_importance = np.cumsum(feature_importances[indices])
		
		min_features_to_select = 5  # Adjust this based on your needs

		# Identify the number of features needed to reach the cumulative importance threshold
		num_features_needed = np.where(cumulative_importance >= cumulative_importance_threshold)[0] #[0] + 1

		# If no features meet the threshold or fewer than min_features_to_select do, adjust the number
		if len(num_features_needed) == 0 or len(num_features_needed) < min_features_to_select:
			num_features_needed = min_features_to_select
		else:
			num_features_needed = num_features_needed[0] + 1  # Adjust for zero-indexing
		
		# Select the top important features based on the threshold
		top_features_indices = indices[:num_features_needed]
		X_train_refined = X_train.to_numpy()[:, top_features_indices]
		
		# Perform cross-validation with the refined feature set
		scores = cross_val_score(model_for_importance, X_train_refined, y_train, cv=3, scoring='neg_mean_squared_error')
		mse = -scores.mean()
		return mse
		
	return objective

# def get_min_data_for_bt(df_1m, start_datetime):
# 	# # path_symbol = get_symbol_path_2021(exchange_name, symbol, '1m')
# 	# path_symbol = get_symbol_path(exchange_name, symbol, '1m')
# 	# df_1m = pd.read_csv(path_symbol)
# 	df_1m['datetime'] = pd.to_datetime(df_1m['datetime'], utc=True)	
# 	df_1m = df_1m[df_1m['datetime'] >= start_datetime]


# 	df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])

# 	index_open = df_1m.columns.get_loc("open")
# 	index_high = df_1m.columns.get_loc("high")
# 	index_low = df_1m.columns.get_loc("low")
# 	index_datetime = df_1m.columns.get_loc("datetime")	
# 	df_1m.drop_duplicates(subset = "datetime", keep = 'first', inplace = True)
# 	np_1m = df_1m.to_numpy()
# 	np_1m[:, index_datetime] = np_1m[:, index_datetime].astype(np.datetime64)

# 	del df_1m

# 	return index_open, index_high, index_low, index_datetime, np_1m	

def get_optuna_trials_per_model():
	# config = get_config_file()
	optuna_trials_str = config['train']['optuna_trials_per_model']
	optuna_trials_per_model = ast.literal_eval(optuna_trials_str)
	return optuna_trials_per_model

def save_input_features_as_csv(column_names, path):
	with open(path, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(column_names)

def save_training_metadata(model_name, obj_dataset, path_metadata):

	input_features = obj_dataset.X_train.columns.tolist() 
	training_end_date = str(obj_dataset.X_train.index.max())
	backtest_end_date = str(obj_dataset.X_backtest.index.max())
	forward_end_date = str(obj_dataset.X_forwardtest.index.max())
	training_metadata = {
			'input_features': input_features,
			'training_end_date': training_end_date,
			'backtest_end_date': backtest_end_date,
			'forwardtest_end_date': forward_end_date
		}
	
	if os.path.exists(path_metadata):
		with open(path_metadata, 'r') as file:
			try:
				data = json.load(file)
			except json.JSONDecodeError:  # Handles empty file case
				data = {}
	else:
		data = {}

	data[model_name] = training_metadata
	
	with open(path_metadata, 'w') as file:
		json.dump(data, file, indent=4)

	# with open(path, 'w', newline='') as file:
	# 	writer = csv.writer(file)
	# 	writer.writerow(column_names)

def get_stats_from_ledger(df_ledger, time_horizon_minutes, df_type, dict_config):
	# starting_balance, _, _, _, transaction_fee, _, _, = get_backtest_params(time_horizon_minutes)
	starting_balance, _, _, _, transaction_fee, _, _, = get_backtest_params(dict_config, time_horizon_minutes=time_horizon_minutes)
	df_ledger['datetime'] = df_ledger['datetime']	
	df_ledger.set_index('datetime', inplace=True)
	df_ledger = df_ledger[df_ledger['sell_price'] != 0]
	df_ledger['pnl'] = df_ledger['pnl'] - transaction_fee
	
	try:
		if len(df_ledger) > 10 :
			# print(df_ledger)
			df_ledger['is_successful'] = df_ledger['pnl'] > 0
			long_trades = df_ledger[df_ledger['predicted_direction'] == 'long']
			short_trades = df_ledger[df_ledger['predicted_direction'] == 'short']
			if long_trades.empty:
				long_accuracy = 0
			else:
				long_accuracy = round(long_trades['is_successful'].mean()*100, 2)

			if short_trades.empty:
				short_accuracy = 0
			else:
				short_accuracy = round(short_trades['is_successful'].mean()*100, 2)

			

			# print(long_accuracy, short_accuracy)


			ending_balance = round(float(df_ledger['balance'].iloc[-1]), 2)
			ending_pnl = round(float(df_ledger['pnl_sum'].iloc[-1]), 2)
			res = stats.linregress(range(len(df_ledger.pnl_sum)), df_ledger.pnl_sum.to_numpy())
			r2 = res.rvalue**2
			if ending_balance < starting_balance:
				r2 = -r2
			r2 = round(r2, 2)
			df_ledger = df_ledger['pnl']/100
			df_ledger.index = df_ledger.index.tz_localize(None)
			wins = df_ledger[df_ledger >= 0]
			loss = df_ledger[df_ledger < 0]
			sharpe = round(qs.stats.sharpe(df_ledger, periods=365), 2)
			sortino = round(qs.stats.sortino(df_ledger, periods=365), 2)
			max_drawdown = round(qs.stats.max_drawdown(df_ledger) * 100, 2)
			num_trades = df_ledger.count()
			num_profits = wins.count()
			num_losses = loss.count()
			total_profit_percent = round(wins.sum() * 100, 2)
			total_loss_percent = round(loss.sum() * 100, 2)
			win_loss_ratio = round(qs.stats.win_loss_ratio(df_ledger), 2)
			profit_ratio = round(qs.stats.profit_ratio(df_ledger), 2)
			recovery_factor = round(qs.stats.recovery_factor(df_ledger), 2)

			long_trades = len(long_trades)
			long_trades = round((long_trades/num_trades) * 100, 2)
			short_trades = len(short_trades)
			short_trades = round((short_trades/num_trades) * 100, 2)

		else:
			ending_balance, ending_pnl, r2, sharpe, sortino, max_drawdown, num_trades, num_profits, num_losses, total_profit_percent, total_loss_percent, win_loss_ratio, profit_ratio, recovery_factor, long_accuracy, short_accuracy, long_trades, short_trades = -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100
	except Exception as e:
		ending_balance, ending_pnl, r2, sharpe, sortino, max_drawdown, num_trades, num_profits, num_losses, total_profit_percent, total_loss_percent, win_loss_ratio, profit_ratio, recovery_factor, long_accuracy, short_accuracy, long_trades, short_trades = -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100











	dict_stats  = {
				f'{df_type}_ending_balance': ending_balance,
				f'{df_type}_pnl_percent': ending_pnl,
				f'{df_type}_r2': r2,
				f'{df_type}_sharpe': sharpe,
				f'{df_type}_sortino': sortino,
				f'{df_type}_max_drawdown': max_drawdown,
				f'{df_type}_num_trades': num_trades,
				f'{df_type}_num_profits': num_profits,
				f'{df_type}_num_losses': num_losses,
				f'{df_type}_total_profit_pct': total_profit_percent,
				f'{df_type}_total_loss_pct': total_loss_percent,
				f'{df_type}_win_loss_ratio': win_loss_ratio,
				f'{df_type}_profit_ratio': profit_ratio,
				f'{df_type}_recovery_factor': recovery_factor,
				f'{df_type}_pnl_long_accuracy': long_accuracy,
				f'{df_type}_pnl_short_accuracy': short_accuracy,
				f'{df_type}_long_trades_pct': long_trades,
				f'{df_type}_short_trades_pct': short_trades,
			}



	# remove nan if any
	# # dict_stats = {k: v if v is not None else -100 for k, v in dict_stats.items()}
	# print(dict_stats)
	convert_large_numbers = lambda d: {k: -100 if v is None or (isinstance(v, (float, np.float_)) and (np.isnan(v) or np.isinf(v))) else v for k, v in d.items()}
	dict_stats = convert_large_numbers(dict_stats)
	# print(dict_stats)

	return dict_stats

def get_ts_df_dropped(df, N):
	# N, _ = get_ts_in_out_length()
	N = N-1
	# print(df)

	range_to_drop = list(range(N))
	# print("range_to_drop", range_to_drop)
	df = df.drop(range_to_drop)

	return df

	


def get_ml_backtest(model, obj_dataset, bt_type, dict_config):

	# backtest
	if bt_type == 'bt':
		bt_predictions = model.predict(obj_dataset.X_backtest)
		obj_dataset.df_backtest['predicted_direction'] = np.sign(bt_predictions)
		fill_zeros(obj_dataset.df_backtest, 'predicted_direction')
		df_ledger, balance, pnl_percent = perform_backtest(obj_dataset, obj_dataset.df_backtest, dict_config)
		# print(df_ledger)
		return df_ledger, balance, pnl_percent, bt_predictions

	# forwardtest
	elif bt_type == 'ft':
		ft_predictions = model.predict(obj_dataset.X_forwardtest)
		obj_dataset.df_forwardtest['predicted_direction'] = np.sign(ft_predictions)
		fill_zeros(obj_dataset.df_forwardtest, 'predicted_direction')
		df_ledger, balance, pnl_percent = perform_backtest(obj_dataset, obj_dataset.df_forwardtest, dict_config)
		# print(df_ledger)
		return df_ledger, balance, pnl_percent, ft_predictions
	
	else:
		print('Invalid backtest type')
		exit(0)
	
def save_df_as_db(path_db, table_name, df):
	# print(df)
	# if df.empty:
	# 	return
	if df is not None and len(df) > 0:
		df = df.fillna("-100")
		for col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='ignore')
		conn = sqlite3.connect(path_db)
		c = conn.cursor()
		df.to_sql(table_name, conn, if_exists='replace', index=False)
		conn.commit()
		conn.close()

def save_html_report(obj_dataset, model, path_report, learner_type, dict_config):
	if learner_type == 'ml':
		predictions = model.predict(obj_dataset.X_bt_ft)
		df_tmp = obj_dataset.df_bt_ft.copy()
		df_tmp['predicted_direction'] = np.sign(predictions)
	elif learner_type == 'ts':
		predictions = get_ts_predictions(model, obj_dataset.darts_y_bt_ft_target, obj_dataset.darts_X_bt_ft, obj_dataset.time_horizon_minutes, input_chunk_length=dict_config['input_chunk_length'])
		df_tmp = get_ts_df_dropped(obj_dataset.df_bt_ft.copy(), dict_config['input_chunk_length'])
		df_tmp['predicted_direction'] = np.sign(predictions)
		# fill_zeros(df_tmp, 'predicted_direction')
	else:
		print('Invalid learner type')
		exit(0)

	fill_zeros(df_tmp, 'predicted_direction')
	df_ledger, _, _ = perform_backtest(obj_dataset, df_tmp, dict_config)
	# _, _, _, _, transaction_fee, _, _, = get_backtest_params(obj_dataset.time_horizon_minutes)
	_, _, _, _, transaction_fee, _, _, = get_backtest_params(dict_config, time_horizon_minutes=obj_dataset.time_horizon_minutes)
	df_ledger['datetime'] = pd.to_datetime(df_ledger['datetime'])
	df_ledger.set_index('datetime', inplace=True)
	if len(df_ledger[df_ledger['sell_price'] != 0]) > 2:
		df_ledger = df_ledger[df_ledger['sell_price'] != 0]
		df_ledger['pnl'] = df_ledger['pnl'] - transaction_fee
	df_ledger = df_ledger['pnl']/100
	df_ledger.index = df_ledger.index.tz_localize(None)
	# qs.reports.html(df_ledger, output=True, compounded=False, download_filename=path_report)
	if not df_ledger.empty:
		try:
			qs.reports.html(df_ledger, output=True, compounded=False, download_filename=path_report)
		except Exception:
			pass

def get_ts_backtest(model, obj_dataset, bt_type, dict_config):
	# backtest
	if bt_type == 'bt':
		bt_predictions = get_ts_predictions(model, obj_dataset.darts_y_backtest_target, obj_dataset.darts_X_backtest, obj_dataset.time_horizon_minutes, input_chunk_length=dict_config['input_chunk_length'])
		
		df_tmp = get_ts_df_dropped(obj_dataset.df_backtest.copy(), dict_config['input_chunk_length'])
		df_tmp['predicted_direction'] = np.sign(bt_predictions)
		fill_zeros(df_tmp, 'predicted_direction')
		df_ledger, balance, pnl_percent = perform_backtest(obj_dataset, df_tmp, dict_config)
		# print(df_ledger)

		# obj_dataset.df_backtest['predicted_direction'] = np.sign(bt_predictions)
		# fill_zeros(obj_dataset.df_backtest, 'predicted_direction')
		# df_ledger, balance, pnl_percent = perform_backtest(obj_dataset, obj_dataset.df_backtest)
		return df_ledger, balance, pnl_percent, bt_predictions

	# forwardtest
	elif bt_type == 'ft':
		ft_predictions = get_ts_predictions(model, obj_dataset.darts_y_forwardtest_target, obj_dataset.darts_X_forwardtest, obj_dataset.time_horizon_minutes, input_chunk_length=dict_config['input_chunk_length'])
		df_tmp = get_ts_df_dropped(obj_dataset.df_forwardtest.copy(), dict_config['input_chunk_length'])
		df_tmp['predicted_direction'] = np.sign(ft_predictions)
		fill_zeros(df_tmp, 'predicted_direction')
		df_ledger, balance, pnl_percent = perform_backtest(obj_dataset, df_tmp, dict_config)
		# print(df_ledger)
		return df_ledger, balance, pnl_percent, ft_predictions
		# obj_dataset.df_forwardtest['predicted_direction'] = np.sign(ft_predictions)
		# fill_zeros(obj_dataset.df_forwardtest, 'predicted_direction')
		# return perform_backtest(obj_dataset, obj_dataset.df_forwardtest)

	else:
		print('Invalid backtest type')
		exit(0)

def flatten_dict(d, parent_key='', sep='_'):
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, dict):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)

def get_training_results_db_name():
	return 'training_results.db'

def get_learner_dirs(dir_current, symbol, time_horizon, model_name):
	dir_models = os.path.join(dir_current, symbol, time_horizon, 'models', model_name)
	dir_results = os.path.join(dir_current, symbol, time_horizon, 'results', model_name)
	path_db_training_results = os.path.join(dir_current, symbol, time_horizon, 'dbs', f'{get_training_results_db_name()}')
	dir_reports = os.path.join(dir_current, symbol, time_horizon, 'reports', model_name)
	# dir_input_features = os.path.join(dir_current, symbol, time_horizon, 'input_features')
	dir_metadata = os.path.join(dir_current, symbol, time_horizon, 'metadata')

	utils_common.create_override_dir(dir_models)
	utils_common.create_override_dir(dir_results)
	utils_common.create_override_dir(dir_reports)
	utils_common.create_if_not_exists(dir_metadata)

	# drop if table exists
	conn = sqlite3.connect(path_db_training_results)
	cur = conn.cursor()
	cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{model_name}'")
	if cur.fetchone():
		cur.execute(f"DROP TABLE {model_name}")
	conn.commit()
	conn.close()

	return dir_models, dir_results, path_db_training_results, dir_reports, dir_metadata

def get_halving_dates():
	return [
		pd.to_datetime('2012-11-28 00:00:00+00:00'),
		pd.to_datetime('2016-07-09 00:00:00+00:00'),
		pd.to_datetime('2020-05-11 00:00:00+00:00'),
		pd.to_datetime('2024-04-21 00:00:00+00:00'),
		pd.to_datetime('2028-03-21 00:00:00+00:00'),
		# Add future halving dates if needed
		]

def until_next_halving(date):
	for halving_date in get_halving_dates():
	# 	if date < halving_date:
	# 		return (halving_date - date).days
	# return 0  # No future halvings, return 0
		if date < halving_date:
			delta = halving_date - date
			days = delta.days
			weeks = days / 7
			months = days / 30  # Approximation, as not all months have 30 days
			return days, weeks, months
	return 0, 0, 0  # No future halvings

def get_ts_model(model_name):
	# NHiTSModel, TFTModel, TiDEModel, BlockRNNModel
	input_chunk_length, output_chunk_length = get_ts_in_out_length()
	n_epochs = 10
	pl_trainer_kwargs = {"accelerator": "cpu"}

	if model_name == 'ts_nhits':
		return NHiTSModel(input_chunk_length=input_chunk_length,
							output_chunk_length=output_chunk_length,
							random_state=42,
							pl_trainer_kwargs=pl_trainer_kwargs,
							n_epochs=n_epochs
							), input_chunk_length
	elif model_name == 'ts_tft':
		return TFTModel(input_chunk_length=input_chunk_length,
								output_chunk_length=output_chunk_length,
								random_state=42,
								pl_trainer_kwargs=pl_trainer_kwargs,
								add_relative_index=True,
								n_epochs=n_epochs
								), input_chunk_length
	elif model_name == 'ts_brnn':
		return BlockRNNModel(input_chunk_length=input_chunk_length,
								output_chunk_length=output_chunk_length,
								random_state=42,
								pl_trainer_kwargs=pl_trainer_kwargs,
								n_epochs=n_epochs
								), input_chunk_length
	elif model_name == 'ts_tide':
		return TiDEModel(input_chunk_length=input_chunk_length,
								output_chunk_length=output_chunk_length,
								random_state=42,
								pl_trainer_kwargs=pl_trainer_kwargs,
								n_epochs=n_epochs
								), input_chunk_length
	elif model_name == 'ts_tcn':
		# print(input_chunk_length)
		input_chunk_length = 6
		return TCNModel(input_chunk_length=input_chunk_length,
								output_chunk_length=output_chunk_length,
								# kernel_size=input_chunk_length-1,
								random_state=42,
								pl_trainer_kwargs=pl_trainer_kwargs,
								n_epochs=n_epochs
								), input_chunk_length
	elif model_name == 'ts_tm':
		return TransformerModel(input_chunk_length=input_chunk_length,
								output_chunk_length=output_chunk_length,
								random_state=42,
								pl_trainer_kwargs=pl_trainer_kwargs,
								n_epochs=n_epochs
								), input_chunk_length
	else:
		print('Invalid model name')
		exit(0)


def get_ts_features(model_name, obj_dataset, n_permutations=1):
	print(f"find best features for {model_name}...")
	model, N = get_ts_model(model_name)
	model.fit(series=obj_dataset.darts_y_train_target, past_covariates=obj_dataset.darts_X_train, verbose=False)

	validation_target = obj_dataset.darts_y_backtest_target
	validation_past_covariates = obj_dataset.darts_X_backtest
	time_horizon_minutes = obj_dataset.time_horizon_minutes

	original_pred = get_ts_predictions(model, validation_target, validation_past_covariates, time_horizon_minutes, N)
	list_validation_target = validation_target.values()
	list_validation_target = [item[0] for item in list_validation_target]
	# N, _ = get_ts_in_out_length()  
	list_validation_target = list_validation_target[N-1:]
	original_score = mean_squared_error(list_validation_target, original_pred)
	
	importances = {}
	for column in validation_past_covariates.columns:
		scores = []
		for _ in range(n_permutations):
			# Permute the column values
			permuted = validation_past_covariates.copy()
			permuted_df = permuted.pd_dataframe()
			permuted_df[column] = np.random.permutation(permuted_df[column].values)
			permuted = TimeSeries.from_dataframe(permuted_df)

			permuted_pred = get_ts_predictions(model, validation_target, permuted, time_horizon_minutes, N)
			permuted_score = mean_squared_error(list_validation_target, permuted_pred)
			scores.append(permuted_score)
		
		# Calculate the average increase in error
		importances[column] = np.mean(scores) - original_score
	# print(importances)
	top_features = get_best_ts_features(importances)
	# print(top_features)
	return top_features

def get_best_ts_features(importance_dict, fallback_percentage=0.2):
	# Separate features with positive importance scores
	positive_features = [feature for feature, importance in importance_dict.items() if importance > 0]
	
	# If there are positive features, return them
	if positive_features:
		return positive_features
	else:
		# If no positive features, select a certain percentage of top features based on their absolute importance scores
		num_features = len(importance_dict)
		num_to_select = max(int(num_features * fallback_percentage), 1)  # Ensure at least one feature is selected
		# Sort features by absolute importance scores in descending order
		sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
		# Select the top features based on the fallback criterion
		top_features = [feature for feature, _ in sorted_features[:num_to_select]]
		return top_features

import configparser
import datetime
import glob
import math
import os
import random
import openai

def group_selected_features_via_chatgpt(best_features):
	#openai.api_key = 'sk-proj-nEeoauNVvLUy9UEQYgRUT3BlbkFJjrXV0aWAmaMazu8xC5P9'
	client = openai.OpenAI()
	# Construct the message to send to ChatGPT
	prompt = (
		"Based on your domain knowledge and expert judgment, please collect these features "
		"into groups to use in training our machine learning models. Take into consideration "
		f"the need to cover different types of indicators in each group(each group should contain "
		"combinations of, price, volatility,trend, and momentum)"
		"Please follow bellow adjustments carefully:"
		"_ Required number of groups is 5"
		"_ Required number of features per group is higher than 15"
		"_ features into group should not have same type of indicators, for example don't merge, sma with ema, Bollinger Band with Keltner channels "
		"_ Use close , open , high in all groups."
		"_ Use all  day_  indicators in all groups"
		"_ you can use same features in different group if needed to ensure getting required number of features (in current case 10) for each group "
		"_ Features derived from same indicator must be in same group"
		f". Here are the features: {', '.join(best_features)}\n\n"
	)

	response = client.chat.completions.create(
		model="gpt-4",
		messages=[
			{"role": "system", "content": "You are an expert in trading strategy configuration validation."},
			{"role": "user", "content": prompt}
		],
		max_tokens=1000
	)

	# print(response)
	# Parse the response
	groups = response.choices[0].message.content.strip()

	return groups
def update_dataset_with_best_features(obj_dataset, best_features):
    # Update the dataset with the best features
    obj_dataset.X_train = obj_dataset.X_train[best_features]
    obj_dataset.darts_X_train = obj_dataset.darts_X_train[best_features]
    obj_dataset.darts_X_backtest = obj_dataset.darts_X_backtest[best_features]
    obj_dataset.darts_X_forwardtest = obj_dataset.darts_X_forwardtest[best_features]
    obj_dataset.darts_X_bt_ft = obj_dataset.darts_X_bt_ft[best_features]

    # Calculate the correlation matrix
    correlation_matrix = obj_dataset.X_train.corr()

    # Unstack the correlation matrix and filter out self-correlations
    corr_pairs = correlation_matrix.where(
        pd.np.triu(pd.np.ones(correlation_matrix.shape), k=1).astype(bool)
    ).stack()

    # Sort the correlation pairs by absolute value
    sorted_corr_pairs = corr_pairs.abs().sort_values(ascending=False)
    # Filter pairs with correlation higher than 0.5
    high_corr_pairs = sorted_corr_pairs[sorted_corr_pairs > 0.5]

    pd.set_option('display.max_rows', len(high_corr_pairs))
    print(f'It selected {len(best_features)} features:')
    print(f'Selected features are: {best_features}')
    # Print the sorted correlation pairs with values higher than 0.5
    print("Correlation matrix of the selected features in X_train (sorted from highest to lowest) with correlation > 0.5:")
    print(high_corr_pairs)

    groups = group_selected_features_via_chatgpt(best_features)
    print('selected groups of features  using chatgbt: ')
    print(groups)
    return obj_dataset
def update_ml_dataset_with_best_features(obj_dataset, best_features):
    # Update the dataset with the best features
    obj_dataset.X_train = obj_dataset.X_train[best_features]
    obj_dataset.X_backtest = obj_dataset.X_backtest[best_features]
    obj_dataset.X_forwardtest = obj_dataset.X_forwardtest[best_features]
    obj_dataset.X_bt_ft = obj_dataset.X_bt_ft[best_features]

    # Calculate the correlation matrix
    correlation_matrix = obj_dataset.X_train.corr()

    # Unstack the correlation matrix and filter out self-correlations
    corr_pairs = correlation_matrix.where(
		pd.np.triu(pd.np.ones(correlation_matrix.shape), k=1).astype(bool)
	).stack()
    # Sort the correlation pairs by absolute value
    sorted_corr_pairs = corr_pairs.abs().sort_values(ascending=False)

    # Filter pairs with correlation higher than 0.5
    high_corr_pairs = sorted_corr_pairs[sorted_corr_pairs > 0.5]

    pd.set_option('display.max_rows', len(high_corr_pairs))
    print(f'It selected {len(best_features)} features:')
    print(f'Selected features are: {best_features}')
    # Print the sorted correlation pairs with values higher than 0.5
    print("Correlation matrix of the selected features in X_train (sorted from highest to lowest) with correlation > 0.5:")
    print(high_corr_pairs)

    groups = group_selected_features_via_chatgpt(best_features)
    print('selected groups of features  using chatgbt: ')
    print(groups)
    return obj_dataset


def get_torch_device(dict_config):
	# use_gpu = get_bool('train', 'use_gpu')
	use_gpu = dict_config['train']['use_gpu']
	if use_gpu:
		if torch.cuda.is_available():
			pl_trainer_kwargs = {"accelerator": "gpu"}
		else:
			print("GPU is not available, training will use CPU")
			pl_trainer_kwargs = {"accelerator": "cpu"}
	else:
		pl_trainer_kwargs = {"accelerator": "cpu"}

	return pl_trainer_kwargs

def save_trial_result(model_name, model, obj_dataset, trial_model_name, df_ledger_bt, df_ledger_ft, dir_models, dir_results, dir_reports, dict_config):

	if 'ts' in model_name:
		model_type = 'ts'
	elif 'ml' in model_name:
		model_type = 'ml'
	else:
		print('Invalid model type')
		exit(0)

	if model_type == 'ts':
		model.save(os.path.join(dir_models, trial_model_name + '.pt'))
	if model_type == 'ml':
		joblib.dump(model, os.path.join(dir_models, trial_model_name + '.pkl') )

	# save backtest and forwardtest ledgers of the current model, and their combined report
	# backtest
	df_ledger_bt.to_csv(os.path.join(dir_results, trial_model_name + '_backtest_ledger.csv'))

	# forwardtest
	df_ledger_ft.to_csv(os.path.join(dir_results, trial_model_name + '_forwardtest_ledger.csv'))

	# save html report
	path_report = os.path.join(dir_reports, f'{trial_model_name}.html')
	save_html_report(obj_dataset, model, path_report, model_type, dict_config)

def append_training_info_df(param_dict, df_training_info):
	# flatten and convert params datatype
	param_dict = flatten_dict(param_dict)
	param_dict = {key: str(value) for key, value in param_dict.items()}

	# append this trial's params in the dataframe
	df_tmp = pd.DataFrame([param_dict])
	if df_training_info is None:
		df_training_info = df_tmp
	else:
		df_training_info = df_training_info.append(df_tmp, ignore_index=True)

	return df_training_info

def get_model_accuracy(bt_type, obj_dataset, predictions, param_dict):
	if bt_type == 'bt':
		actual_signs = np.sign(obj_dataset.y_backtest_direction)
	elif bt_type == 'ft':
		actual_signs = np.sign(obj_dataset.y_forwardtest_direction)
	predicted_signs = np.sign(predictions)
	try:
		actual_signs = actual_signs.iloc[param_dict['input_chunk_length']-1:]
	except:
		pass
	# print(actual_signs)
	# print(predicted_signs)
	model_accuracy = round(sum(predicted_signs == actual_signs) / len(predicted_signs), 2)
	dict_tmp = {f'{bt_type}_model_accuracy_pct': model_accuracy}
	param_dict.update(dict_tmp)

	ones_mask = actual_signs == 1
	ones_accuracy = np.mean(predicted_signs[ones_mask] == 1)
	neg_ones_mask = actual_signs == -1
	neg_ones_accuracy = np.mean(predicted_signs[neg_ones_mask] == -1)
	dict_tmp = {f'{bt_type}_model_long_accuracy_pct': round(ones_accuracy, 2), f'{bt_type}_model_short_accuracy_pct': round(neg_ones_accuracy, 2)}
	param_dict.update(dict_tmp)


def get_btft_stats(model_name, model, obj_dataset, param_dict, dict_config):
	bt_type = 'bt'
	if 'ts' in model_name:
		df_ledger_bt, balance_bt, pnl_percent_bt, predictions_bt = get_ts_backtest(model, obj_dataset, bt_type, dict_config)
	elif 'ml' in model_name:
		df_ledger_bt, balance_bt, pnl_percent_bt, predictions_bt = get_ml_backtest(model, obj_dataset, bt_type, dict_config)

	dict_stats = get_stats_from_ledger(df_ledger_bt, obj_dataset.time_horizon_minutes, bt_type, dict_config)
	param_dict.update(dict_stats)

	###### model accuracy
	get_model_accuracy(bt_type, obj_dataset, predictions_bt, param_dict)

	# perform forwardtest and get stats
	bt_type = 'ft'
	if 'ts' in model_name:
		df_ledger_ft, _, _, predictions_ft = get_ts_backtest(model, obj_dataset, bt_type, dict_config)
	elif 'ml' in model_name:
		df_ledger_ft, _, _, predictions_ft = get_ml_backtest(model, obj_dataset, bt_type, dict_config)
		
	dict_stats = get_stats_from_ledger(df_ledger_ft, obj_dataset.time_horizon_minutes, bt_type, dict_config)
	param_dict.update(dict_stats)

	###### model accuracy
	get_model_accuracy(bt_type, obj_dataset, predictions_ft, param_dict)

	return df_ledger_bt, df_ledger_ft, predictions_bt, balance_bt, pnl_percent_bt

def read_model_metrics(dir_current, symbol, time_horizon, model_name):

	db_path = os.path.join(dir_current, symbol, time_horizon, 'dbs', f'{get_training_results_db_name()}')
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	tables = cursor.fetchall()

	found = False
	for table in tables:
		if model_name in table[0]:  # Check if the name is in the first (and only) element of the tuple
			found = True
			table_name = table[0]  # Get the name of the table
			break
	
	if not found:
		print(f"Model {model_name} not found in the {get_training_results_db_name()}.")
		return None

	df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	def model_score(row, df_type='bt'):
		profit_weight = 0.3
		drawdown_weight = 0.2  # Lower drawdown is better, hence it will be subtracted
		sharpe_weight = 0.2
		sortino_weight = 0.2
		recovery_factor_weight = 0.1

		score = (
			profit_weight * row[f'{df_type}_pnl_percent'] -
			drawdown_weight * abs(row[f'{df_type}_max_drawdown']) +
			sharpe_weight * row[f'{df_type}_sharpe'] +
			sortino_weight * row[f'{df_type}_sortino'] +
			recovery_factor_weight * row[f'{df_type}_recovery_factor']
		)
		# print(row['trial_number'], score)

		return score

	df['bt_score'] = df.apply(model_score, df_type='bt', axis=1)
	df['ft_score'] = df.apply(model_score, df_type='ft', axis=1)
	df['avg_score'] = (df['bt_score'] + df['ft_score']) / 2

	conn.close()
	return df.loc[df['avg_score'].idxmax()]

# def get_selected_models(dir_current, symbol, time_horizon, model_name, dict_config):

# 	db_path = os.path.join(dir_current, symbol, time_horizon, 'dbs', f'{get_training_results_db_name()}')
# 	conn = sqlite3.connect(db_path)
# 	cursor = conn.cursor()
# 	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# 	tables = cursor.fetchall()

# 	found = False
# 	for table in tables:
# 		if model_name in table[0]:  # Check if the name is in the first (and only) element of the tuple
# 			found = True
# 			table_name = table[0]  # Get the name of the table
# 			break
	
# 	if not found:
# 		# print(f"Model {model_name} not found in the {get_training_results_db_name()}.")
# 		return None, None

# 	df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# 	print(df)
# 	cols_metric = []
# 	for col in df.columns:
# 		if dict_config['models_selection']['metric'] in col:
# 			cols_metric.append(col)
# 			df[col] = pd.to_numeric(df[col], errors='coerce')
# 		# df[col] = pd.to_numeric(df[col], errors='coerce')

# 	if len(cols_metric) == 0:
# 		# print(f"Metric {dict_config['models_selection']['metric']} not found in the table {table_name} of {get_training_results_db_name()}")
# 		return None, None
# 	else:
# 		if dict_config['models_selection']['duration'] == 'backtest':
# 			col_name = next((x for x in cols_metric if 'bt_' in x), None)
# 			filtered_rows = df[df[col_name] > float(dict_config['models_selection']['value'])].dropna(how='all')
# 		elif dict_config['models_selection']['duration'] == 'forwardtest':
# 			col_name = next((x for x in cols_metric if 'ft_' in x), None)
# 			filtered_rows = df[df[col_name] > float(dict_config['models_selection']['value'])].dropna(how='all')
# 		elif dict_config['models_selection']['duration'] == 'average':
# 			row_averages = df[cols_metric].mean(axis=1)
# 			filtered_rows = df[row_averages > float(dict_config['models_selection']['value'])]
# 		else:
# 			print(f"Invalid duration {dict_config['models_selection']['duration']}")
		
# 	if len(filtered_rows) == 0:
# 		# print(f"No models found with {dict_config['models_selection']['metric']} of {dict_config['models_selection']['value']}")
# 		return None, None
# 	else:
# 		dir_models = os.path.join(dir_current, symbol, time_horizon, 'models', table_name)
# 		# selected_models = []
# 		# for trial_num in filtered_rows['trial_number']:
# 		# 	print(trial_num)
# 		# 	print(filtered_rows[trial_num])
# 		# 	pattern = os.path.join(dir_models, f"*_{trial_num}.*")
# 		# 	matches = glob.glob(pattern)
# 		# 	selected_models.extend(matches)

# 		# print(filtered_rows)
# 		ft_metrics_names = ['ft_pnl_percent', 'ft_r2', 'ft_sharpe', 'ft_sortino', 'ft_max_drawdown', 'ft_win_loss_ratio', 'ft_recovery_factor', 'ft_model_long_accuracy(%)', 'ft_model_short_accuracy(%)', 'ft_long_trades(%)', 'ft_short_trades(%)']
# 		selected_models = []
# 		db_records = []
# 		for index, row in filtered_rows.iterrows():
# 			## find and save model path
# 			# pattern = os.path.join(dir_models, f"*_{int(row['trial_number'])}.*")
# 			pattern = os.path.join(dir_models, f"*{row['model_name']}.*")
# 			matches = glob.glob(pattern)
# 			selected_models.extend(matches)

# 			## find and save model metrics a long with its name
# 			if isinstance(matches, list):
# 				model_name = os.path.basename(matches[0]).split('.')[0]
# 			else:
# 				model_name = os.path.basename(matches).split('.')[0]
# 			dict_ft_metrics = {'model_name': model_name, 'time_horizon': time_horizon}  # Example new key-value pair
# 			dict_tmp = row[ft_metrics_names].to_dict()
# 			dict_tmp = {k.replace('ft_', '').replace('(%)', ''): v for k, v in dict_tmp.items()}
# 			dict_ft_metrics.update(dict_tmp)
# 			db_records.append(dict_ft_metrics)

# 		## find model metadata
# 		dir_metadata = os.path.join(dir_current, symbol, time_horizon, 'metadata')
# 		matches = glob.glob(os.path.join(dir_metadata, f"{model_name.rsplit('_', 4)[0]}*"))
# 		selected_models.extend(matches)
	
# 	conn.close()

# 	return selected_models, db_records


# conditional metric
def get_selected_models(time_horizon, dict_config):
	metric = dict_config['models_selection']['metric']
	# print(metric)
	list_models_names = []
	connection = utils_common.get_db_connection_obj()
	cursor = connection.cursor()

	try:
		# Check if the models_stats table exists
		cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_results');")
		table_exists = cursor.fetchone()[0]
		metric = metric.replace(' and ', ' AND ').replace(' or ', ' OR ')
		if table_exists:  # Only proceed if the table exists
			columns = [
					'model_name', 'time_horizon', 'ft_pnl_percent', 'ft_r2', 'ft_sharpe', 'ft_sortino',
					'ft_max_drawdown', 'ft_win_loss_ratio', 'ft_recovery_factor',
					'ft_model_long_accuracy_pct', 'ft_model_short_accuracy_pct',
					'ft_long_trades_pct', 'ft_short_trades_pct'
				]
			# Convert the list of columns to a comma-separated string
			columns_str = ', '.join(columns)
			
			# Build the query with dynamic conditions
			query = f"""
				SELECT {columns_str}
				FROM training_results
				WHERE {metric} AND time_horizon='{time_horizon}';
			"""
			# print(query)
			# Execute the query
			cursor.execute(query)
			
			rows = cursor.fetchall()
			# print(rows)
			new_columns = [col if not col.startswith('ft_') else col[3:] for col in columns]
			list_db_records = [dict(zip(new_columns, row)) for row in rows]

			list_models_names = [model['model_name'] for model in list_db_records]

		else:
			print("The table 'models_stats' does not exist in the database.")
	except Exception as e:
		print("An error occurred:", e)
	finally:
		cursor.close()
		connection.close()
	
	if len(list_models_names) > 0:
		metadata_name = get_metadata_from_model_name(list_models_names[0])
		list_models_names.append(f"{metadata_name}")
		return list_models_names, list_db_records
	else: 
		return None, None


def get_selected_models_(dir_current, symbol, time_horizon, model_name, dict_config):
	db_path = os.path.join(dir_current, symbol, time_horizon, 'dbs', f'{get_training_results_db_name()}')
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	tables = cursor.fetchall()

	found = False
	for table in tables:
		if model_name in table[0]:  # Check if the name is in the first (and only) element of the tuple
			found = True
			table_name = table[0]  # Get the name of the table
			break
	
	if not found:
		# print(f"Model {model_name} not found in the {get_training_results_db_name()}.")
		return None, None

	df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE {dict_config['models_selection']['metric']}", conn)

	print(df)
		
	if len(df) == 0:
		# print(f"No models found with {dict_config['models_selection']['metric']} of {dict_config['models_selection']['value']}")
		return None, None
	else:
		dir_models = os.path.join(dir_current, symbol, time_horizon, 'models', table_name)
		# selected_models = []
		# for trial_num in filtered_rows['trial_number']:
		# 	print(trial_num)
		# 	print(filtered_rows[trial_num])
		# 	pattern = os.path.join(dir_models, f"*_{trial_num}.*")
		# 	matches = glob.glob(pattern)
		# 	selected_models.extend(matches)

		# print(filtered_rows)
		ft_metrics_names = ['ft_pnl_percent', 'ft_r2', 'ft_sharpe', 'ft_sortino', 'ft_max_drawdown', 'ft_win_loss_ratio', 'ft_recovery_factor', 'ft_model_long_accuracy_pct', 'ft_model_short_accuracy_pct', 'ft_long_trades_pct', 'ft_short_trades_pct']
		selected_models = []
		db_records = []
		for index, row in df.iterrows():
			## find and save model path
			# pattern = os.path.join(dir_models, f"*_{int(row['trial_number'])}.*")
			pattern = os.path.join(dir_models, f"*{row['model_name']}.*")
			matches = glob.glob(pattern)
			selected_models.extend(matches)

			## find and save model metrics a long with its name
			if isinstance(matches, list):
				model_name = os.path.basename(matches[0]).split('.')[0]
			else:
				model_name = os.path.basename(matches).split('.')[0]
			dict_ft_metrics = {'model_name': model_name, 'time_horizon': time_horizon}  # Example new key-value pair
			dict_tmp = row[ft_metrics_names].to_dict()
			dict_tmp = {k.replace('ft_', '').replace('(%)', ''): v for k, v in dict_tmp.items()}
			dict_ft_metrics.update(dict_tmp)
			db_records.append(dict_ft_metrics)

		## find model metadata
		dir_metadata = os.path.join(dir_current, symbol, time_horizon, 'metadata')
		matches = glob.glob(os.path.join(dir_metadata, f"{model_name.rsplit('_', 4)[0]}*"))
		selected_models.extend(matches)
	
	conn.close()

	return selected_models, db_records
	
# def get_samba_credentials():
# 	server_ip = "161.97.76.136"
# 	username = "SA_PSQL_Engine"
# 	password = "Lmmoyjwt1"

# 	return server_ip, username, password

# def get_db_credentials():
# 	host = "161.97.76.136"
# 	database = "DataHub"
# 	username = "SA_PSQL_Engine"
# 	password = "Lmmoyjwt1"
# 	port = "5432"
# 	return host, database, username, password, port


def upload_best_models_to_samba(list_selected_models, symbol, time_horizon):

	server_ip, username, password = utils_common.get_samba_credentials()
	path_training_models = f'{server_ip}/sambashare/training_models/{symbol}/{time_horizon}/' 
	path_best_models = f'{server_ip}/sambashare/best_models/{symbol}/{time_horizon}/' 

	smbclient.register_session(server=server_ip, username=username, password=password, port=445)
	files = smbclient.listdir(path_training_models)

	if not smbclient.path.exists(path_best_models):
		smbclient.makedirs(path_best_models, exist_ok=True)

	for selected_model in list_selected_models:
		for file in files:
			base_name, extension = os.path.splitext(file)
			if extension == '.ckpt':
				base_name, _ = os.path.splitext(base_name)
				extension = '.pt.ckpt'
			if selected_model == base_name:
				try:
					# Copy the file from source to destination
					with smbclient.open_file(f"{path_training_models}{file}", mode='rb') as src_file:
						with smbclient.open_file(f"{path_best_models}{file}", mode='wb') as dest_file:
							shutil.copyfileobj(src_file, dest_file)
					# print("File copied successfully!")
				except Exception as e:
					print(f"An error occurred: {e}")

	smbclient.reset_connection_cache()


def upload_models_to_samba(list_selected_models, symbol, time_horizon):
	for selected_model in list_selected_models:
		server_ip, username, password = utils_common.get_samba_credentials()
		path_server = f'/sambashare/best_models/{symbol}/{time_horizon}/'  # Path should be relative to the shared_folder

		smbclient.register_session(server=server_ip, username=username, password=password, port=445, auth_protocol='ntlm')
		if not smbclient.path.exists(server_ip + path_server):
			# smbclient.mkdir(server_ip + path_server)
			smbclient.makedirs(server_ip + path_server, exist_ok=True)
			
		with open(selected_model, 'rb') as source_file:
			file_contents = source_file.read()

		server_file_name = os.path.basename(selected_model)
		with smbclient.open_file(server_ip + path_server + server_file_name, mode='wb') as dest_file:
			dest_file.write(file_contents)

def udpate_db_model_stats(list_db_records):
	connection = utils_common.get_db_connection_obj()
	cursor = connection.cursor()
	check_and_create_table(cursor)
	# print(list_db_records) # model_long_accuracy(%)

	
	insert_query = """
	INSERT INTO models_stats (model_name, time_horizon, pnl_percent, r2, sharpe, sortino, max_drawdown, win_loss_ratio, recovery_factor, model_long_accuracy_pct, model_short_accuracy_pct, long_trades_pct, short_trades_pct)
	VALUES (%(model_name)s, %(time_horizon)s, %(pnl_percent)s, %(r2)s, %(sharpe)s, %(sortino)s, %(max_drawdown)s, %(win_loss_ratio)s, %(recovery_factor)s, %(model_long_accuracy_pct)s, %(model_short_accuracy_pct)s, %(long_trades_pct)s, %(short_trades_pct)s)
	ON CONFLICT (model_name) DO UPDATE SET
	time_horizon = EXCLUDED.time_horizon,
	pnl_percent = EXCLUDED.pnl_percent,
	r2 = EXCLUDED.r2,
	sharpe = EXCLUDED.sharpe,
	sortino = EXCLUDED.sortino,
	max_drawdown = EXCLUDED.max_drawdown,
	win_loss_ratio = EXCLUDED.win_loss_ratio,
	recovery_factor = EXCLUDED.recovery_factor,
	model_long_accuracy_pct = EXCLUDED.model_long_accuracy_pct,
	model_short_accuracy_pct = EXCLUDED.model_short_accuracy_pct,
	long_trades_pct = EXCLUDED.long_trades_pct,
	short_trades_pct = EXCLUDED.short_trades_pct;
	"""

	psycopg2.extras.execute_batch(cursor, insert_query, list_db_records)
	connection.commit()

	cursor.close()
	connection.close()
	
# def get_db_connection():
	# host, database, username, password, port = get_db_credentials()

# 	try:
# 		connection = psycopg2.connect(
# 			host=host,
# 			database=database,
# 			user=username,
# 			password=password,
# 			port=port
# 		)
# 	except OperationalError as e:
# 		print(f"The error '{e}' occurred")

# 	return connection

def check_and_create_table(cursor):
	# cursor.execute("DROP TABLE IF EXISTS models_stats;")
	# print("Table 'models_stats' dropped.")
	cursor.execute("""
		SELECT EXISTS(
			SELECT * FROM information_schema.tables 
			WHERE table_name=%s
		);
	""", ('models_stats',))
	
	# 'model_name', 'time_horizon', 'pnl_percent', 'r2', 'sharpe', 'sortino',
	# 				'max_drawdown', 'win_loss_ratio', 'recovery_factor',
	# 				'model_long_accuracy_pct', 'model_short_accuracy_pct',
	# 				'long_trades_pct', 'short_trades_pct'

	if not cursor.fetchone()[0]:
		cursor.execute("""
			CREATE TABLE models_stats (
				id SERIAL PRIMARY KEY,
				model_name VARCHAR(255) UNIQUE,
				time_horizon VARCHAR(255),
				pnl_percent FLOAT,
				r2 FLOAT,
				sharpe FLOAT,
				sortino FLOAT,
				max_drawdown FLOAT,
				win_loss_ratio FLOAT,
				recovery_factor FLOAT,
				model_long_accuracy_pct FLOAT,
				model_short_accuracy_pct FLOAT,
				long_trades_pct FLOAT,
				short_trades_pct FLOAT
			);
		""")
		print("Table 'models_stats' created.")
	# else:
	# 	print("Table 'models_stats' already exists.")

def get_selected_models_list(metric, time_horizon):
	list_models_names = []
	# print(metric)

	# Establish database connection
	connection = utils_common.get_db_connection_obj()
	cursor = connection.cursor()

	try:
		# Check if the models_stats table exists
		cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'models_stats');")
		table_exists = cursor.fetchone()[0]
		metric = metric.replace('and', ' AND ').replace('or', ' OR ')
		if table_exists:  # Only proceed if the table exists
			# Build the query with dynamic conditions
			query = f"""
				SELECT model_name
				FROM models_stats
				WHERE {metric} AND time_horizon='{time_horizon}';
			"""
			cursor.execute(query)
			model_names = cursor.fetchall()
			list_models_names = [item[0] for item in model_names]
			print("Selected Models:", list_models_names)
		else:
			print("The table 'models_stats' does not exist in the database.")
	except Exception as e:
		print("An error occurred:", e)
	finally:
		cursor.close()
		connection.close()

	return list_models_names
		
# def get_selected_models_list(metric, value, time_horizon):
# 	# print(metric, value, time_horizon)

# 	list_models_names = []

# 	connection = get_db_connection()
# 	cursor = connection.cursor()

# 	# # Check if the table exists
# 	# cursor.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = %s);", ('models_stats',))
# 	cursor.execute("""
# 		SELECT EXISTS(
# 			SELECT * FROM information_schema.tables 
# 			WHERE table_name=%s
# 		);
# 	""", ('models_stats',))
# 	table_exists = cursor.fetchone()[0]

# 	if table_exists:
# 		if metric in ['pnl_percent', 'r2', 'sharpe', 'sortino', 'max_drawdown', 'win_loss_ratio', 'recovery_factor', 'long_accuracy', 'short_accuracy']:  
# 			query = """
# 					SELECT model_name
# 					FROM models_stats
# 					WHERE time_horizon = %s AND {} >= %s;
# 					""".format(metric)
# 			cursor.execute(query, (time_horizon, value))
# 			model_names = cursor.fetchall()
# 			list_models_names = [item[0] for item in model_names]
# 		else:
# 			print(f"Invalid metric {metric} passed")
# 	else:
# 		# Handle the case where the table does not exist
# 		print(f"The table model_stats does not exist in the database.")


# 	cursor.close()
# 	connection.close()

# 	return list_models_names

def get_conditional_expression(expression):
	conditions = []
	# Split the expression by 'and' or 'or'
	parts = expression.split(' and ') if ' and ' in expression else expression.split(' or ')
	for part in parts:
		# Split each part by comparison operators
		metric, condition = part.split(' >= ') if ' >= ' in part else \
							part.split(' <= ') if ' <= ' in part else \
							part.split(' > ') if ' > ' in part else \
							part.split(' < ') if ' < ' in part else \
							part.split(' == ') if ' == ' in part else \
							part.split(' != ') if ' != ' in part else [None, None]
		# Extract metric, condition, and value
		conditions.append({"metric": metric.strip(), "condition": condition.strip(), "value": float(part.split(condition.strip())[1])})
	return conditions

def download_models_if_missing(dir_models, symbol, time_horizon, list_models):
	server_ip, username, password = utils_common.get_samba_credentials()
	path_server = f'{server_ip}/sambashare/best_models/{symbol}/{time_horizon}/'  
	smbclient.register_session(server=server_ip, username=username, password=password, port=445)

	if not os.path.exists(dir_models):
		os.makedirs(dir_models)

	files = smbclient.listdir(path_server)
	for file in files:
		base_name, extension = os.path.splitext(file)
		if extension == '.ckpt':
			base_name, _ = os.path.splitext(base_name)
			extension = '.pt.ckpt'

		if base_name in list_models and extension in ['.pkl', '.pt', '.pt.ckpt']:
			local_path = os.path.join(dir_models, file)
			remote_path = os.path.join(path_server.strip('smb://'), file)  

			if not os.path.exists(local_path):
				# print(f"Downloading {file}...")
				with smbclient.open_file(remote_path, mode='rb') as remote_file, open(local_path, 'wb') as local_file:
					local_file.write(remote_file.read())
				print(f"Downloaded {file} successfully.")
			# else:
			# 	print(f"{file} is already present locally.")

	# metadata_names = [item.rsplit('_', 4)[0] + '_metadata.json' for item in list_models]	
	metadata_names = [get_metadata_from_model_name(item) + '.json' for item in list_models]	
	metadata_names = list(set(metadata_names))
	# metadata_name = f"{list_models[0].rsplit('_', 4)[0]}_metadata.json"
	for metadata_name in metadata_names:
		if metadata_name in files:
			remote_path = os.path.join(path_server.strip('smb://'), metadata_name) 
			local_path = os.path.join(dir_models, metadata_name)
			with smbclient.open_file(remote_path, mode='rb') as remote_file, open(local_path, 'wb') as local_file:
				local_file.write(remote_file.read())


	list_model_paths = []
	allowed_extensions = ('.pkl', '.pt')
	for file in os.listdir(dir_models):
		for model_name in list_models:
			if file.startswith(model_name) and file.endswith(allowed_extensions):
				list_model_paths.append(os.path.join(dir_models, file))

	return list_model_paths

def update_signals(strategy_name, df_signals):
	# print(metric, value, time_horizon)
	# df_signals['datetime'] = pd.to_datetime(df_signals.index)
	df_signals.rename(columns={'predicted_direction': 'signal'}, inplace=True)
	df_signals = df_signals.reset_index()
	df_signals.rename(columns={'index': 'datetime'}, inplace=True)
	df_signals['datetime'] = pd.to_datetime(df_signals['datetime'])
	df_signals['datetime'] = df_signals['datetime'].apply(lambda dt: dt.replace(second=0, microsecond=0))

	# data_to_insert = df_signals.to_dict(orient='records')
	# # print(data_to_insert)
	# print(df_signals)

	# list_models_names = []

	connection = utils_common.get_db_connection_obj()
	cursor = connection.cursor()

	# # Check if the table exists
	# cursor.execute("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = %s);", ('models_stats',))
	cursor.execute("""
		SELECT EXISTS(
			SELECT * FROM information_schema.tables 
			WHERE table_name=%s
		);
	""", (f'signals_{strategy_name}',))
	table_exists = cursor.fetchone()[0]
	# print(table_exists)

	if not table_exists:
		columns = ', '.join([f'{col} REAL' for col in df_signals.columns if col != 'datetime'])
		create_table_query = f"""
		CREATE TABLE IF NOT EXISTS signals_{strategy_name} (
			id SERIAL PRIMARY KEY,
			datetime TIMESTAMP WITHOUT TIME ZONE UNIQUE,
			{columns}
		);
		"""
		# print(create_table_query)
		cursor.execute(create_table_query)
		# connection.commit()
		latest_record = df_signals.iloc[-1:]
		latest_record['datetime'] = pd.to_datetime(latest_record['datetime']).dt.tz_convert(None)
		columns = ', '.join(latest_record.columns)
		placeholders = ', '.join(['%s'] * len(latest_record.columns))
		insert_query = f"""
		INSERT INTO signals_{strategy_name} ({columns})
		VALUES ({placeholders})
		ON CONFLICT (datetime) DO NOTHING;
		"""
		# datetime_value = latest_record['datetime'].values[0].astype('M8[ms]').astype(datetime.datetime)
		data_tuple = tuple(latest_record.values[0])
		cursor.execute(insert_query, data_tuple)
		connection.commit()
		print(f"Table signals_{strategy_name} created.")
		# data_to_insert = list(latest_record[['datetime', 'predicted_direction']].itertuples(index=False, name=None))

	else:
		cursor.execute(f'SELECT datetime FROM signals_{strategy_name}')
		# existing_datetimes = set([record[0] for record in cursor.fetchall()])
		# df_signals['datetime'] = pd.to_datetime(df_signals['datetime']).dt.tz_convert(None)
		# new_records = df_signals[~df_signals['datetime'].isin(existing_datetimes)]
		# data_to_insert = list(new_records[['datetime', 'predicted_direction']].itertuples(index=False, name=None))
		# print(df_signals)
		max_datetime_in_db = cursor.fetchone()[-1]
		df_signals['datetime'] = pd.to_datetime(df_signals['datetime']).dt.tz_convert(None)
		newer_records = df_signals[df_signals['datetime'] > max_datetime_in_db]
		columns = ', '.join(newer_records.columns)
		placeholders = ', '.join(['%s'] * len(newer_records.columns))
		insert_query = f"""
		INSERT INTO signals_{strategy_name} ({columns})
		VALUES ({placeholders})
		ON CONFLICT (datetime) DO NOTHING;
		"""
		# Convert DataFrame to list of tuples
		data_tuples = list(newer_records.itertuples(index=False, name=None))
		cursor.executemany(insert_query, data_tuples)
		connection.commit()
		# data_to_insert = list(newer_records[['datetime', 'predicted_direction']].itertuples(index=False, name=None))

	# if data_to_insert:
	# 	# # Modify this query to match your table's schema
	# 	insert_query = f'INSERT INTO signals_{strategy_name} (datetime, signal) VALUES %s ON CONFLICT (datetime) DO NOTHING'
	# 	execute_values(cursor, insert_query, data_to_insert, template=None, page_size=100)
	# 	connection.commit()
	# 	# print(f"{len(data_to_insert)} new records inserted into 'signals_{strategy_name}'.")
	# # else:
	# # 	print("No new records to insert.")



	# with connection.cursor() as cursor:
	# 	for record in data_to_insert:
	# 		datetime, predicted_direction = record
	# 		print(datetime, predicted_direction, record)
	# 		# Convert datetime to the correct format if necessary
	# 		# datetime = datetime.isoformat()  # Uncomment if manual conversion is needed
	# 		try:
	# 			cursor.execute(
	# 				f"""
	# 				INSERT INTO signals_{strategy_name} (datetime, signal)
	# 				VALUES (%s, %s)
	# 				ON CONFLICT (datetime) DO NOTHING
	# 				""",
	# 				(datetime, predicted_direction)
	# 			)
	# 		except Exception as e:
	# 			print(f"Error: {e}")
	# 			connection.rollback()  # Rollback the transaction on error
	# 		else:
	# 			connection.commit()  # Commit the transaction
		
		
		


	# cursor.close()
	connection.close()

	return []

def create_training_results_table(cursor):
		cursor.execute("""
		SELECT EXISTS(
			SELECT * FROM information_schema.tables 
			WHERE table_name=%s
		);
		""", ('training_results',))
		
		if not cursor.fetchone()[0]:
			cursor.execute("""
				CREATE TABLE training_results (
					id SERIAL PRIMARY KEY,
					model_name VARCHAR(255) UNIQUE,
					learner VARCHAR(255),
					time_horizon VARCHAR(255),
				  	trial_number INTEGER,
					bt_ending_balance FLOAT,
					bt_pnl_percent FLOAT,
					bt_r2 FLOAT,
					bt_sharpe FLOAT,
					bt_sortino FLOAT,
					bt_max_drawdown FLOAT,
					bt_num_trades INTEGER,
					bt_num_profits INTEGER,
					bt_num_losses INTEGER,
					bt_total_profit_pct FLOAT,
					bt_total_loss_pct FLOAT,
					bt_win_loss_ratio FLOAT,
					bt_profit_ratio FLOAT,
					bt_recovery_factor FLOAT,
					bt_pnl_long_accuracy FLOAT,
					bt_pnl_short_accuracy FLOAT,
					bt_long_trades_pct FLOAT,
					bt_short_trades_pct FLOAT,
					bt_model_accuracy_pct FLOAT,
					bt_model_long_accuracy_pct FLOAT,
					bt_model_short_accuracy_pct FLOAT,
					ft_ending_balance FLOAT,
					ft_pnl_percent FLOAT,
					ft_r2 FLOAT,
					ft_sharpe FLOAT,
					ft_sortino FLOAT,
					ft_max_drawdown FLOAT,
					ft_num_trades INTEGER,
					ft_num_profits INTEGER,
					ft_num_losses INTEGER,
					ft_total_profit_pct FLOAT,
					ft_total_loss_pct FLOAT,
					ft_win_loss_ratio FLOAT,
					ft_profit_ratio FLOAT,
					ft_recovery_factor FLOAT,
					ft_pnl_long_accuracy FLOAT,
					ft_pnl_short_accuracy FLOAT,
					ft_long_trades_pct FLOAT,
					ft_short_trades_pct FLOAT,
					ft_model_accuracy_pct FLOAT,
					ft_model_long_accuracy_pct FLOAT,
					ft_model_short_accuracy_pct FLOAT
				);
			""")
			print("Table 'training_results' created.")	


def upload_training_results(df_training_info, dir_models, dir_metadata, symbol, time_horizon):
	# from StrategySphere.common import utils_common
	# from psycopg2.extras import execute_values
	# import pandas as pd
	# import glob
	# import smbclient
	
	if len(df_training_info) > 0:
		df_training_info['bt_ending_balance'] = pd.to_numeric(df_training_info['bt_ending_balance'], errors='coerce')
		df_filtered = df_training_info[df_training_info['bt_ending_balance'] > 1000]

		if len(df_filtered) > 0:
			
			########################################################################################
			#### upload models to samba
			########################################################################################
			list_selected_models = []
			for index, row in df_filtered.iterrows():
				## find and save model path
				pattern = os.path.join(dir_models, f"*{row['model_name']}.*")
				matches = glob.glob(pattern)
				list_selected_models.extend(matches)
				model_name = row['model_name']

			# ### get metadata file
			# parts = model_name.split('_')
			# # Extract the relevant parts
			# base_parts = parts[:4]  # ['btc1', '12h', 'binance', '27Jan24']
			# timestamp_parts = parts[-2:]  # ['24150', '1229']
			# # Construct the new filename with 'metadata'
			# metadata_filename = '_'.join(base_parts) + '_metadata_' + '_'.join(timestamp_parts)
			metadata_filename = get_metadata_from_model_name(model_name)
			# print(metadata_filename)
			matches = glob.glob(os.path.join(dir_metadata, f"{metadata_filename}*"))
			list_selected_models.extend(matches)

			for selected_model in list_selected_models:
				server_ip, username, password = utils_common.get_samba_credentials()
				path_server = f'/sambashare/training_models/{symbol}/{time_horizon}/'  # Path should be relative to the shared_folder

				smbclient.register_session(server=server_ip, username=username, password=password, port=445, auth_protocol='ntlm')
				if not smbclient.path.exists(server_ip + path_server):
					# smbclient.mkdir(server_ip + path_server)
					smbclient.makedirs(server_ip + path_server, exist_ok=True)
					
				with open(selected_model, 'rb') as source_file:
					file_contents = source_file.read()

				server_file_name = os.path.basename(selected_model)
				with smbclient.open_file(server_ip + path_server + server_file_name, mode='wb') as dest_file:
					dest_file.write(file_contents)
			

			########################################################################################
			#### upload training results to db
			########################################################################################
			# print(df_filtered)
			connection = utils_common.get_db_connection_obj()
			cursor = connection.cursor()

			create_training_results_table(cursor)

			db_columns = [
					'model_name', 'learner', 'time_horizon', 'trial_number', 'bt_ending_balance', 'bt_pnl_percent',
					'bt_r2', 'bt_sharpe', 'bt_sortino', 'bt_max_drawdown', 'bt_num_trades',
					'bt_num_profits', 'bt_num_losses', 'bt_total_profit_pct', 'bt_total_loss_pct',
					'bt_win_loss_ratio', 'bt_profit_ratio', 'bt_recovery_factor',
					'bt_pnl_long_accuracy', 'bt_pnl_short_accuracy', 'bt_long_trades_pct',
					'bt_short_trades_pct', 'bt_model_accuracy_pct', 'bt_model_long_accuracy_pct',
					'bt_model_short_accuracy_pct', 'ft_ending_balance', 'ft_pnl_percent', 'ft_r2',
					'ft_sharpe', 'ft_sortino', 'ft_max_drawdown', 'ft_num_trades', 'ft_num_profits',
					'ft_num_losses', 'ft_total_profit_pct', 'ft_total_loss_pct', 'ft_win_loss_ratio',
					'ft_profit_ratio', 'ft_recovery_factor', 'ft_pnl_long_accuracy', 'ft_pnl_short_accuracy',
					'ft_long_trades_pct', 'ft_short_trades_pct', 'ft_model_accuracy_pct', 'ft_model_long_accuracy_pct',
					'ft_model_short_accuracy_pct'
				]
			
			df_filtered = df_filtered[db_columns]
			insert_columns = ", ".join(df_filtered.columns)
			insert_sql = f"""
							INSERT INTO training_results ({insert_columns}) VALUES %s
							ON CONFLICT (model_name)DO UPDATE
							SET 
								learner = EXCLUDED.learner,
								time_horizon = EXCLUDED.time_horizon,
								trial_number = EXCLUDED.trial_number,
								bt_ending_balance = EXCLUDED.bt_ending_balance,
								bt_pnl_percent = EXCLUDED.bt_pnl_percent,
								bt_r2 = EXCLUDED.bt_r2,
								bt_sharpe = EXCLUDED.bt_sharpe,
								bt_sortino = EXCLUDED.bt_sortino,
								bt_max_drawdown = EXCLUDED.bt_max_drawdown,
								bt_num_trades = EXCLUDED.bt_num_trades,
								bt_num_profits = EXCLUDED.bt_num_profits,
								bt_num_losses = EXCLUDED.bt_num_losses,
								bt_total_profit_pct = EXCLUDED.bt_total_profit_pct,
								bt_total_loss_pct = EXCLUDED.bt_total_loss_pct,
								bt_win_loss_ratio = EXCLUDED.bt_win_loss_ratio,
								bt_profit_ratio = EXCLUDED.bt_profit_ratio,
								bt_recovery_factor = EXCLUDED.bt_recovery_factor,
								bt_pnl_long_accuracy = EXCLUDED.bt_pnl_long_accuracy,
								bt_pnl_short_accuracy = EXCLUDED.bt_pnl_short_accuracy,
								bt_long_trades_pct = EXCLUDED.bt_long_trades_pct,
								bt_short_trades_pct = EXCLUDED.bt_short_trades_pct,
								bt_model_accuracy_pct = EXCLUDED.bt_model_accuracy_pct,
								bt_model_long_accuracy_pct = EXCLUDED.bt_model_long_accuracy_pct,
								bt_model_short_accuracy_pct = EXCLUDED.bt_model_short_accuracy_pct,
								ft_ending_balance = EXCLUDED.ft_ending_balance,
								ft_pnl_percent = EXCLUDED.ft_pnl_percent,
								ft_r2 = EXCLUDED.ft_r2,
								ft_sharpe = EXCLUDED.ft_sharpe,
								ft_sortino = EXCLUDED.ft_sortino,
								ft_max_drawdown = EXCLUDED.ft_max_drawdown,
								ft_num_trades = EXCLUDED.ft_num_trades,
								ft_num_profits = EXCLUDED.ft_num_profits,
								ft_num_losses = EXCLUDED.ft_num_losses,
								ft_total_profit_pct = EXCLUDED.ft_total_profit_pct,
								ft_total_loss_pct = EXCLUDED.ft_total_loss_pct,
								ft_win_loss_ratio = EXCLUDED.ft_win_loss_ratio,
								ft_profit_ratio = EXCLUDED.ft_profit_ratio,
								ft_recovery_factor = EXCLUDED.ft_recovery_factor,
								ft_pnl_long_accuracy = EXCLUDED.ft_pnl_long_accuracy,
								ft_pnl_short_accuracy = EXCLUDED.ft_pnl_short_accuracy,
								ft_long_trades_pct = EXCLUDED.ft_long_trades_pct,
								ft_short_trades_pct = EXCLUDED.ft_short_trades_pct,
								ft_model_accuracy_pct = EXCLUDED.ft_model_accuracy_pct,
								ft_model_long_accuracy_pct = EXCLUDED.ft_model_long_accuracy_pct,
								ft_model_short_accuracy_pct = EXCLUDED.ft_model_short_accuracy_pct
						"""
			values = [tuple(row) for row in df_filtered.to_numpy()]

			execute_values(cursor, insert_sql, values)
			connection.commit()
			cursor.close()
			connection.close()

def get_metadata_from_model_name(model_name):
	parts = model_name.split('_')
	# Extract the relevant parts
	base_parts = parts[:4]  # ['btc1', '12h', 'binance', '27Jan24']
	timestamp_parts = parts[-2:]  # ['24150', '1229']
	# Construct the new filename with 'metadata'
	metadata_filename = '_'.join(base_parts) + '_metadata_' + '_'.join(timestamp_parts)
	return metadata_filename