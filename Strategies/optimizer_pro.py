# Backtesting Algorithm Refactor - Class Design
import pandas as pd
import datetime
import copy
import itertools
from datetime import datetime
from joblib import Parallel, delayed,cpu_count, Memory
import psutil
import gc
import shutil
import os
# from backtester import Backtester
# Define global variables for storing DataFrames

class Optimizer:
    def __init__(self, backtester, period_info, settings, Optimization):
        self.backtester = backtester  # Store Backtester instance
        self.period_info = period_info
        self.settings = settings
        self.Optimization = Optimization
        self.df = None  # Global dataframe to be set before optimization
        self.df_lw = None  # Global lw dataframe to be set before optimization

    def optimize_strategy(self, optimization_metric='Return (%)', df=None, df_lw=None):
        time1 = datetime.now()
        print(f"Memory usage before optimization: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

        # Set global DataFrames to avoid passing large objects in parallel processing
        self.df = df
        self.df_lw = df_lw
        gc.collect()  # Free memory before starting
        best_performance = None
        best_settings = None
        optimization_history = []
        optimization_history_splits = {}
        optimization_summary = []
        optimization_iter_mapping = []

        # Identify active parameters and generate combinations
        active_params = self.get_active_parameters()
        param_combinations = self.generate_parameter_combinations(active_params)

        total_cpu_cores = psutil.cpu_count(logical=False)
        num_jobs = min(total_cpu_cores, max(1, total_cpu_cores - 1))  # Use (total cores - 1) to keep system stable
        print(f'total cores in this server is {total_cpu_cores}, we will use {num_jobs} to keep system stable')

        # Run parallel processing without passing df and df_lw
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing", verbose=10)(# multiprocessing,
            delayed(self.run_single_iteration)(combination, idx, active_params, optimization_metric)
            for idx, combination in enumerate(param_combinations, start=1)
        )
        gc.collect()  # Free memory after parallel execution
        # Process results
        for res in results:
            if res is None:
                continue

            identifier = res["identifier"]
            param_dict = res["param_dict"]
            performance_summary_df = res["performance_summary_df"]
            aggregated_trades = res["aggregated_trades"]

            optimization_history.append(performance_summary_df)
            optimization_iter_mapping.append({'Identifier': identifier, **copy.deepcopy(param_dict)})

            for map_name, splits in self.backtester.get_stats_with_splits(
                aggregated_trades,
                pd.Timestamp(self.period_info["start_date"]),
                pd.Timestamp(self.period_info["end_date"]),
                self.period_info["cross_validation_map"]
            ).items():
                for split_name, split_data in splits.items():
                    split_perf_summary = split_data["performance_summary"].copy()
                    split_perf_summary.insert(0, 'Identifier', identifier)
                    split_perf_summary.insert(1, 'Sheet_Name', split_name)

                    for i, (key, value) in enumerate(param_dict.items()):
                        split_perf_summary.insert(i + 1, key, [value] * len(split_perf_summary))

                    optimization_summary.append(split_perf_summary)

                    if split_name not in optimization_history_splits:
                        optimization_history_splits[split_name] = []
                    optimization_history_splits[split_name].append(split_perf_summary)

            metric_value = performance_summary_df[optimization_metric].values[0]
            if best_performance is None or metric_value > best_performance:
                best_performance = metric_value
                best_settings = param_dict.copy()
                best_aggregated_trades = aggregated_trades
                best_performance_summary_df = performance_summary_df
            # Free memory after processing results
            del performance_summary_df, aggregated_trades
            gc.collect()
        # Create DataFrames from results
        optimization_history_df = pd.concat(optimization_history, ignore_index=True) if optimization_history else pd.DataFrame()
        optimization_summary_df = pd.concat(optimization_summary, ignore_index=True) if optimization_summary else pd.DataFrame()
        optimization_iter_mapping_df = pd.DataFrame(optimization_iter_mapping)
        optimization_summary_df2 = self.restructure_dataframe(optimization_summary_df, optimization_iter_mapping_df)
        optimization_history_splits_df = {
            key: pd.concat(history, ignore_index=True) for key, history in optimization_history_splits.items() if history
        }

        # Clear joblib temp files
        memory = Memory(location=None, verbose=0)
        memory.clear(warn=False)
        gc.collect()

        # Manually remove temp folders
        temp_folder = "C:\\Users\\azzam\\AppData\\Local\\Temp"
        for folder in os.listdir(temp_folder):
            if folder.startswith("joblib_memmapping_folder_"):
                shutil.rmtree(os.path.join(temp_folder, folder), ignore_errors=True)

        time2 = datetime.now()
        print(f"Time for optimization: {time2 - time1}")
        print(f"Memory usage after optimization: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
        # print(optimization_summary_df2)

        return (best_settings, best_aggregated_trades, best_performance_summary_df, optimization_history_df,
                optimization_history_splits_df, optimization_summary_df, optimization_summary_df2,
                optimization_iter_mapping_df)

    def run_single_iteration(self, combination, identifier, active_params, optimization_metric):
        """ Runs a single iteration of the optimization process without passing large DataFrames """

        # Update strategy settings with the given combination
        self.apply_parameter_combination(active_params, combination)

        # Skip iterations based on logic conditions
        if not self.settings['RSI']['Entry']['active'] and self.settings['RSI']['Exit']['active']:
            return None
        if not self.settings['long_trade'] and not self.settings['short_trade']:
            return None
        if (not self.settings['long_trade'] and self.settings['close_opposite_position']) or (
                not self.settings['short_trade'] and self.settings['close_opposite_position']):
            return None
        if self.settings['ema']['Exit']['active'] and self.settings['EMAdev_exit']:
            return None

        param_dict = {param_name: combination[i] for i, (param_name, _) in enumerate(active_params)}

        # Use globally stored DataFrames to avoid memory duplication
        df, df_lw = self.df, self.df_lw

        df, df_lw, df_results, aggregated_trades_init = self.backtester.run_backtest_with_settings(df, df_lw)
        start_date = pd.Timestamp(self.period_info["start_date"])
        end_date = pd.Timestamp(self.period_info["end_date"])
        cross_validation_map = self.period_info["cross_validation_map"]

        split_results = self.backtester.get_stats_with_splits(aggregated_trades_init, start_date, end_date, cross_validation_map)
        if split_results:# and split_results['map0_']:
            # print(f'combination: {combination}')
            # print(split_results)
            performance_summary_df = split_results['map0_']['full_backtest']['performance_summary']
            aggregated_trades = split_results['map0_']['full_backtest']['aggregated_trades']

            if performance_summary_df is not None and not performance_summary_df.empty:
                return {
                    "identifier": identifier,
                    "performance_summary_df": performance_summary_df,
                    "aggregated_trades": aggregated_trades,
                    "param_dict": param_dict
                }
            return None
        else:
            return None

    # Other helper methods remain unchanged
    def validate_current_settings(self):
        """ Validate current parameter combination """
        settings = self.settings
        if not settings['RSI']['Entry']['active'] and settings['RSI']['Exit']['active']:
            return False
        if not settings['long_trade'] and not settings['short_trade']:
            return False
        if (not settings['long_trade'] and settings['close_opposite_position']) or \
                (not settings['short_trade'] and settings['close_opposite_position']):
            return False
        if settings['ema']['Exit']['active'] and settings['EMAdev_exit']:
            return False
        return True

    def restructure_dataframe(self, df: pd.DataFrame, optimization_iter_mapping_df):
        selected_columns = [
            "nbTrad", "WinRate(%)", "Mart_nbTrad", "Mart_WinRate(%)", "Max_draw", "Profit($)", "AvgYr_Profit($)",
            "RoMaD",
            "nbTrad_Mart1", "WinRate_Mart1(%)", "nbTrad_Mart2", "WinRate_Mart2(%)", "nbTrad_Mart3", "WinRate_Mart3(%)",
            "nbTrad_Mart4", "WinRate_Mart4(%)", "nbTrad_Mart5", "WinRate_Mart5(%)", "nbTrad_Mart6", "WinRate_Mart6(%)"
        ]

        # Extract iter_mapping_columns excluding 'Identifier'
        iter_mapping_columns = [col for col in optimization_iter_mapping_df.columns if col != 'Identifier']

        # Ensure only existing columns are selected
        existing_columns = ['Identifier', 'Sheet_Name'] + iter_mapping_columns + [col for col in selected_columns if
                                                                                  col in df.columns]

        # Filter the dataframe to keep only necessary columns
        df = df[existing_columns]

        # Set unique columns as index (excluding selected columns)
        unique_columns = ['Identifier', 'Sheet_Name'] + iter_mapping_columns
        df.set_index(unique_columns, inplace=True)

        # Pivot only the selected columns
        new_df = df.unstack(level='Sheet_Name')

        # Flatten the multi-index columns for selected columns
        new_df.columns = ["_".join(map(str, col)).strip() for col in new_df.columns.values]

        # Reset index to keep unique columns as part of the DataFrame
        new_df.reset_index(inplace=True)

        return new_df
    def get_active_parameters(self):
        """ Identify all active parameters for optimization. """
        active_params = []


        if self.Optimization['trailing_sl']['active']:
            active_params.append(('trailing_sl', self.Optimization['trailing_sl']['use_trailing_sl']))
        if self.Optimization['long_trade']['active']:
            active_params.append(('long_trade', self.Optimization['long_trade']['use_long']))
        if self.Optimization['short_trade']['active']:
            active_params.append(('short_trade', self.Optimization['short_trade']['use_short']))

        for param in ['ema', 'SL_%', 'TP_%', 'deviation']:
            if self.Optimization[param]['active']:
                active_params.append((param, self.Optimization[param]['level']))

        if self.Optimization['Martingale']['active']:
            active_params.extend([
                ('Martingale_active', [True, False]),
                ('Martingale_n_trade', self.Optimization['Martingale']['n_trade']),
                ('Martingale_multiplier', self.Optimization['Martingale']['multiplier'])
            ])

        default_level = (0, 0)
        for ind in ['RSI', 'CCI', 'STOCH']:
            if self.Optimization[ind]['active']:
                use_key = f"use_{ind}"
                if self.Optimization[ind][use_key]:
                    active_params.append((ind, self.Optimization[ind][use_key]))
                    active_params.append((f"{ind}_levels", self.Optimization[ind]['levels']))
                else:
                    active_params.append((ind, [False]))
                    active_params.append((f"{ind}_levels", [default_level]))

        if self.Optimization['Exit_rsi']['active']:
            active_params.append(('Exit_rsi', self.Optimization['Exit_rsi']['use_rsi']))
        if self.Optimization['Exit_ema']['active']:
            active_params.append(('Exit_ema', self.Optimization['Exit_ema']['use_ema']))
        if self.Optimization['close_opposite_position']['active']:
            active_params.append(('close_opposite_position', self.Optimization['close_opposite_position']['use_close_opposite_position']))
        if self.Optimization['EMAdev_exit']['active']:
            active_params.append(('EMAdev_exit', self.Optimization['EMAdev_exit']['use_EMAdev_exit']))

        # for param in ['Exit_rsi', 'Exit_ema', 'close_opposite_position', 'EMAdev_exit']:
        #     if self.Optimization[param]['active']:
        #         active_params.append((param, self.Optimization[param][f"use_{param}"]))

        return active_params


    def apply_parameter_combination(self, active_params, combination):
        """ Apply a specific parameter combination to strategy settings. """
        for i, (param_name, _) in enumerate(active_params):
            if isinstance(combination[i], tuple):  # Handle (oversold, overbought) pairs
                self.settings[param_name.replace('_levels', '')]['Entry']['oversold'], \
                    self.settings[param_name.replace('_levels', '')]['Entry']['overbought'] = combination[i]
            else:
                if param_name.endswith('_levels'):
                    continue  # Skip levels if indicator is inactive

                if param_name == 'Martingale_active':
                    self.settings['Martingale']['active'] = combination[i]
                elif param_name == 'Martingale_n_trade':
                    self.settings['Martingale']['n_trade'] = combination[i]
                elif param_name == 'Martingale_multiplier':
                    self.settings['Martingale']['multiplier'] = combination[i]
                elif param_name == 'long_trade':
                    self.settings['long_trade'] = combination[i]
                elif param_name == 'short_trade':
                    self.settings['short_trade'] = combination[i]
                elif param_name == 'trailing_sl':
                    self.settings['trailing_sl'] = combination[i]

                elif param_name == 'RSI':
                    self.settings['RSI']['Entry']['active'] = combination[i]
                elif param_name == 'RSI_levels':
                    self.settings['RSI']['Entry']['oversold'], self.settings['RSI']['Entry']['overbought'] = \
                        combination[i]
                elif param_name == 'CCI':
                    self.settings['CCI']['Entry']['active'] = combination[i]
                elif param_name == 'CCI_levels':
                    self.settings['CCI']['Entry']['oversold'], self.settings['CCI']['Entry']['overbought'] = \
                        combination[i]
                elif param_name == 'STOCH':
                    self.settings['STOCH']['Entry']['active'] = combination[i]
                elif param_name == 'STOCH_levels':
                    self.settings['STOCH']['Entry']['oversold'], self.settings['STOCH']['Entry']['overbought'] = \
                        combination[i]
                elif param_name == 'Exit_rsi':
                    self.settings['RSI']['Exit']['active'] = combination[i]
                elif param_name == 'ema':
                    self.settings['ema']['level'] = combination[i]
                elif param_name == 'Exit_ema':
                    self.settings['ema']['Exit']['active'] = combination[i]
                elif param_name == 'close_opposite_position':
                    self.settings['close_opposite_position'] = combination[i]
                elif param_name == 'EMAdev_exit':
                    self.settings['EMAdev_exit'] = combination[i]
                else:
                    self.settings[param_name] = combination[i]


    def generate_parameter_combinations(self, active_params):
        """
        Generates all valid parameter combinations based on active parameters, handling Martingale settings
        and oversold/overbought levels for inactive indicators.

        - list: A list of unique parameter combinations after filtering invalid ones.
        """
        default_level = (0, 0)
        param_combinations = []
        optimization_settings = self.Optimization
        for combination in itertools.product(*[param[1] for param in active_params]):
            filtered_combination = list(combination)

            # Adjust Martingale parameters based on Martingale_active
            martingale_active_index = [i for i, param in enumerate(active_params) if param[0] == 'Martingale_active']
            if martingale_active_index:
                index = martingale_active_index[0]
                if not combination[index]:  # If Martingale_active is False
                    # Set n_trade and multiplier to default values
                    filtered_combination[
                        active_params.index(('Martingale_n_trade', optimization_settings['Martingale']['n_trade']))] = 0
                    filtered_combination[
                        active_params.index(
                            ('Martingale_multiplier', optimization_settings['Martingale']['multiplier']))] = 0

            # Adjust oversold/overbought levels for inactive indicators
            for i, (param_name, _) in enumerate(active_params):
                if param_name.endswith('_levels') and combination[i - 1] is False:
                    filtered_combination[i] = default_level

            param_combinations.append(tuple(filtered_combination))

        # Remove duplicates and unnecessary combinations
        param_combinations = list(set(param_combinations))

        print('-------------------------------------------------------------------------------------------')
        print(f"Will test the following {len(param_combinations)} iterations :")

        return param_combinations
