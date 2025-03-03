import pandas as pd
import numpy as np
from StrategySphere.common import utils_common
from StrategySphere.data.utils import utils_data
from StrategySphere.ml.utils import utils_indicators, utils_ml, minute_data
from darts import TimeSeries
from sklearn.preprocessing import OneHotEncoder


class Dataset():
	def __init__(self, symbol, time_horizon, df_ohlcv, dict_config, df_1m):
		self.symbol = symbol
		self.time_horizon = time_horizon
		self.time_horizon_minutes = utils_common.convert_to_minutes(time_horizon)
		self.best_input_features = []
		
		self.exchange_name = exchange_name = dict_config['data']['exchange']
		utils_common.validate_str(self.exchange_name, utils_data.get_allowed_exchanges_for_data_downloader(), 'exchange')
		# path_symbol = utils.get_symbol_path(exchange_name, self.symbol, self.time_horizon)

		# # read ohlcv data
		# if os.path.exists(path_symbol):
			# df_ohlcv = pd.read_csv(path_symbol)
		df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['datetime'], utc=True)	
		if dict_config['technical_indicators']['enable']:
			df_ohlcv = utils_indicators.get_technical_indicators(df_ohlcv.copy(), dict_config)
		df_ohlcv.set_index('datetime', inplace=True)
		df_ohlcv.index = pd.to_datetime(df_ohlcv.index, utc=True)
		# print(df_ohlcv)

		# else:
		# 	print(f'Data not found, please create data for symbol={self.symbol}, time_horizon={self.time_horizon}.')
		# 	exit(0)	

		# get train, backtest, forwardtest split percentages, and their respective dataframes
		train_split_percent = dict_config['train']['train_split_percent']
		backtest_split_percent = dict_config['train']['backtest_split_percent'] 
		forwardtest_split_percent = dict_config['train']['forwardtest_split_percent']
		utils_ml.validate_train_test_split(train_split_percent, backtest_split_percent, forwardtest_split_percent)
		df_train, df_backtest, df_forwardtest, df_bt_ft = utils_ml.get_train_test_split(df_ohlcv, train_split_percent, backtest_split_percent, forwardtest_split_percent)
		# print(df_backtest)
		self.trained_until = df_train.index[-1].date().strftime("%d%b%y")
		self.minute_data_start_date = df_backtest.index[0]

		# create training data
		self.X_train, self.y_train_target, self.y_train_direction, self.darts_X_train, self.darts_y_train_target = self.create_dataset(df_train, 'train')
		self.X_backtest, self.y_backtest_target, self.y_backtest_direction, self.darts_X_backtest, self.darts_y_backtest_target = self.create_dataset(df_backtest, 'backtest')
		self.X_forwardtest, self.y_forwardtest_target, self.y_forwardtest_direction, self.darts_X_forwardtest, self.darts_y_forwardtest_target = self.create_dataset(df_forwardtest, 'forwardtest')
		self.X_bt_ft, self.y_bt_ft_target, self.y_bt_ft_direction, self.darts_X_bt_ft, self.darts_y_bt_ft_target = self.create_dataset(df_bt_ft, 'bt_ft')
		# print(self.X_backtest)
		# print(self.X_forwardtest)
		# print(self.X_bt_ft)

		if self.best_input_features == []:
			self.best_input_features = self.X_train.columns

		### get 1m data that will be used for backtesting and forwardtesting
		# self.index_open, self.index_high, self.index_low, self.index_datetime, self.np_1m = utils_ml.get_min_data_for_bt(df_1m, df_backtest.index[0])
		self.obj_minute_data = minute_data.MinuteData(df_1m, df_backtest.index[0])

	def create_dataset(self, df_tmp, df_type):
		df = df_tmp.copy()
		close_only = df_tmp[['close']].copy()

		new_order = ['close'] + [col for col in df.columns if col != 'close']
		df = df[new_order]

		df_train_log_transformed = df.copy()
		df_before_log_transform = df_train_log_transformed.copy()

		days, weeks, months = zip(*df_train_log_transformed.index.map(utils_ml.until_next_halving))
		df_train_log_transformed['days_to_next_halving'] = days
		df_train_log_transformed['weeks_to_next_halving'] = weeks
		df_train_log_transformed['months_to_next_halving'] = months

		list_non_logged_cols = []
		list_negative_cols = []

		for column in df_train_log_transformed.columns:
			if all(df_train_log_transformed[column].isin([-1, 0, 1])) or \
					all(df_train_log_transformed[column].isin([-1, 1])) or \
					all(df_train_log_transformed[column].isin([1, 0])):
				list_non_logged_cols.append(column)
			elif any(df_train_log_transformed[column] == 0) or any(df_train_log_transformed[column] < 0):
				list_negative_cols.append(column)
			else:
				if column == 'weeks_to_next_halving' or column == 'days_to_next_halving' or column == 'months_to_next_halving':
					df_train_log_transformed[column] = np.log(df_train_log_transformed[column] + 1).diff()
				else:
					df_train_log_transformed[column] = np.log(df_train_log_transformed[column]).diff()

		df_train_log_transformed.dropna(inplace=True)

		df_target_log_transformed = np.log(close_only) - np.log(close_only.shift(1))
		df_target_log_transformed = df_target_log_transformed.shift(-1)
		df_target_log_transformed = df_target_log_transformed.reindex(df_train_log_transformed.index)
		df_target_log_transformed.dropna(inplace=True)
		df_target_log_transformed.rename(columns={'close': 'target'}, inplace=True)
		df_target_log_transformed['direction'] = np.sign(df_target_log_transformed['target'])

		df_train_log_transformed = df_train_log_transformed.reindex(df_target_log_transformed.index)

		df_train_log_transformed['day_of_week'] = df_train_log_transformed.index.dayofweek
		encoder = OneHotEncoder(sparse=False)
		day_of_week_encoded = encoder.fit_transform(df_train_log_transformed[['day_of_week']])
		day_of_week_encoded_df = pd.DataFrame(day_of_week_encoded,
											  columns=[f'day_{i}' for i in range(day_of_week_encoded.shape[1])])
		day_of_week_encoded_df.index = df_train_log_transformed.index
		df_train_log_transformed = df_train_log_transformed.join(day_of_week_encoded_df)
		df_train_log_transformed.drop('day_of_week', axis=1, inplace=True)

		df_darts = df_train_log_transformed.copy()
		target_column = 'close'
		feature_columns = [col for col in df_darts.columns if col != target_column]
		df_darts = df_darts.reset_index()
		df_darts['datetime'] = pd.to_datetime(df_darts['datetime']).dt.tz_localize(None)
		feature_series = TimeSeries.from_dataframe(df_darts, 'datetime', feature_columns)
		target_series = TimeSeries.from_dataframe(df_darts, 'datetime', [target_column])

		df_bt_ = df_target_log_transformed.copy()
		df_bt_.index = pd.DatetimeIndex(df_bt_.index) + pd.DateOffset(minutes=self.time_horizon_minutes)

		if df_type == 'backtest':
			self.df_backtest = pd.DataFrame(df_target_log_transformed.index, columns=['datetime'])
			self.df_backtest['datetime'] = pd.to_datetime(self.df_backtest['datetime'])
			self.df_backtest['datetime'] = pd.DatetimeIndex(self.df_backtest['datetime']) + pd.DateOffset(
				minutes=self.time_horizon_minutes)
		elif df_type == 'forwardtest':
			self.df_forwardtest = pd.DataFrame(df_target_log_transformed.index, columns=['datetime'])
			self.df_forwardtest['datetime'] = pd.to_datetime(self.df_forwardtest['datetime'])
			self.df_forwardtest['datetime'] = pd.DatetimeIndex(self.df_forwardtest['datetime']) + pd.DateOffset(
				minutes=self.time_horizon_minutes)
		elif df_type == 'bt_ft':
			self.df_bt_ft = pd.DataFrame(df_target_log_transformed.index, columns=['datetime'])
			self.df_bt_ft['datetime'] = pd.to_datetime(self.df_bt_ft['datetime'])
			self.df_bt_ft['datetime'] = pd.DatetimeIndex(self.df_bt_ft['datetime']) + pd.DateOffset(
				minutes=self.time_horizon_minutes)

		has_nan_or_inf = df_train_log_transformed.isna().any().any() or np.isinf(df_train_log_transformed.values).any()
		if has_nan_or_inf:
			df_train_log_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
			df_train_log_transformed.fillna(0, inplace=True)

		# Create copies for Excel export
		df_before_log_transform_excel = df_before_log_transform.copy()
		df_train_log_transformed_excel = df_train_log_transformed.copy()
		df_target_log_transformed_excel = df_target_log_transformed.copy()

		# Convert datetime index to timezone unaware
		df_before_log_transform_excel.index = df_before_log_transform_excel.index.tz_localize(None)
		df_train_log_transformed_excel.index = df_train_log_transformed_excel.index.tz_localize(None)
		df_target_log_transformed_excel.index = df_target_log_transformed_excel.index.tz_localize(None)

		# Export data to Excel
		excel_filename = f'{self.symbol}_{self.time_horizon}_{df_type}_log_transformation_analysis.xlsx'
		with pd.ExcelWriter(excel_filename) as writer:
			df_before_log_transform_excel.to_excel(writer, sheet_name='Before_Log_Transformation')
			df_train_log_transformed_excel.to_excel(writer, sheet_name='After_Log_Transformation')
			df_target_log_transformed_excel.to_excel(writer, sheet_name='Target_Transformation')

		return df_train_log_transformed, df_target_log_transformed['target'], df_target_log_transformed[
			'direction'], feature_series, target_series


		

