from dataprocessor import *
from Indicator import *
from backtester import *
# Set option to display all columns
pd.set_option('display.max_columns', None)

ORIGINAL_CONFIGS = {
    # ------ Time & Data Settings ------
    "period": {
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "validation": {
            "map1": {"method": "percent", "train": 60, "test": 40}
        }
    },

    # ------ Core Strategy Configuration ------
    "strategy": {
        "symbols": ["EURCHF"],
        "timeframes": {
            "main": "30m",
            "lower": "15m"
        },

        "trade_directions": {
            "long": True,
            "short": True
        },

        "martingale": {
            "enabled": True,
            "max_trades": 3,
            "multiplier": 2
        },

        "broker": {
            "name": "IC",
            "account_type": "future",
            "use_db": False
        },

        "indicators": {
            "ema": {
                "enabled": True,
                "period": 100,
                "exit": False
            },
            "rsi": {
                "enabled": True,
                "period": 14,
                "entry_levels": (30, 70),
                "exit_level": 50
            },
            "cci": {
                "enabled": True,
                "period": 14,
                "entry_levels": (-150, 150)
            },
            "stoch": {
                "enabled": True,
                "period": 14,
                "entry_levels": (30, 70)
            }
        }
    },

    # ------ Risk & Money Management ------
    "risk": {
        "initial_capital": 100000,
        "per_trade_risk": 1000,
        "stop_loss": 0.3,
        "take_profit": 0.3,
        "trailing_stop": {
            "enabled": False,
            "levels": [{"trigger": 50, "sl": 0}]
        },
        "instrument": {
            "lot_size": 100000,
            "spread": 0.0,
            "commission": 6
        },
        "deviation": 0.8
    },

    # ------ Optimization Parameters ------
    "optimization": {
        "enabled": False,
        "target_metric": "WinRate(%)",

        "parameters": {
            "general": {
                "trailing_stop": [True, False],
                "trade_directions": {
                    "long": [True, False],
                    "short": [True, False]
                }
            },
            "risk": {
                "stop_loss": [0.6],
                "take_profit": [0.5, 0.7],
                "deviation": [0.6, 0.9]
            },
            "indicators": {
                "ema": {
                    "periods": [50, 100],
                    "exit": [True, False]
                },
                "rsi": {
                    "enabled": [True, False],
                    "levels": [(30, 70)]
                }
            }
        }
    },

    # ------ System Settings ------
    "system": {
        "charts": True,
        "close_opposite": False,
        "ema_exit": True
    }
}

if __name__ == '__main__':
    for symbol in ORIGINAL_CONFIGS['settings']['symbols']:
        for interval in ORIGINAL_CONFIGS['settings']['intervals']:
            print('\n\n\n----------------------------------------------------------------')
            print(f">>>>>> Processing {symbol} on {interval} timeframe <<<<<<")
            print('----------------------------------------------------------------\n')

            # Reset CONFIGS to original before each iteration
            CONFIGS = copy.deepcopy(ORIGINAL_CONFIGS)

            # Update settings dynamically
            CONFIGS['settings']['symbol'] = symbol
            CONFIGS['settings']['interval'] = interval

            # Initialize components
            datahandler_obj = DataProcessor(period_info=CONFIGS['period_info'], settings=CONFIGS['settings'])
            Indica_obj = IndicatorManager(period_info=CONFIGS['period_info'], settings=CONFIGS['settings'])
            backtester_obj = Backtester(period_info=CONFIGS['period_info'], settings=CONFIGS['settings'], Optimization=CONFIGS['Optimization'])
            Visualizer_obj = Visualizer(settings=CONFIGS['settings'])

            # Prepare data
            data, data_lw, exchange_data, exchange_data_lw = datahandler_obj.get_data()
            df, df_lw = Indica_obj.prepare_data(df=data, exchange_data=exchange_data, df_lw=data_lw, exchange_data_lw=exchange_data_lw)
            df = datahandler_obj.optimize_financial_data(df)
            df_lw = datahandler_obj.optimize_financial_data(df_lw)

            # Run backtesting
            df, df_lw, df_results, split_results = backtester_obj._main_(df, df_lw)

            # Plot trade history
            if not CONFIGS['Optimization']['optimization'] and CONFIGS['settings']['charts']:
                Visualizer_obj.plot_interactive_chart(df_lw, df_results)
                Visualizer_obj.plot_drawdown_with_points_interactive(split_results["map0_"]["full_backtest"]["aggregated_trades"])

    print("\nProcessing complete for all symbols and intervals.")
