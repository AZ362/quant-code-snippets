import os
import optuna
from darts.models import AutoARIMA
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error


class AutoARIMA_Model:
    def __init__(
        self, dir_current, symbol, time_horizon, obj_dataset, optimisation_metric
    ):
        self.model_name = "ts_autoarima"
        self.dir_current = dir_current
        self.dir_models = os.path.join(dir_current, symbol, time_horizon, "models")
        self.dir_results = os.path.join(dir_current, symbol, time_horizon, "results")
        self.symbol = symbol
        self.time_horizon = time_horizon
        self.obj_dataset = obj_dataset
        self.best_metric = optimisation_metric
        self.best_model = None
        self.best_model_metric = None
        self.use_gpu = False  # AutoARIMA does not utilize GPU

    def train(self, optuna_trials):
        # Set up the Optuna study with a more efficient sampler
        sampler = TPESampler(seed=42)  # Use a fixed seed for reproducibility
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        # Use a lambda function to pass additional arguments to the objective function
        study.optimize(
            lambda trial: self.objective(trial), n_trials=optuna_trials, n_jobs=-1
        )  # Parallel execution

        # Save the best model found during the study
        if self.best_model is not None:
            model_path = os.path.join(self.dir_models, f"{self.model_name}.pkl")
            self.save_model(self.best_model, model_path)
            print(f"Best model saved to {model_path}")
        else:
            print("No model was optimized. Please check the objective function.")

        # Optionally, output the best trial's hyperparameters and the achieved metric
        print(f"Best trial parameters: {study.best_trial.params}")
        print(f"Best trial objective value: {study.best_trial.value}")

    def get_hyperparameters(self, trial):
        # AutoARIMA parameter suggestions
        # Note: AutoARIMA's automatic selection makes many of these optional.
        # You can choose to fix some parameters and let others be automatically determined.
        max_p = trial.suggest_int("max_p", 1, 5)
        max_d = trial.suggest_int("max_d", 0, 2)
        max_q = trial.suggest_int("max_q", 1, 5)
        max_P = trial.suggest_int("max_P", 0, 2)
        max_D = trial.suggest_int("max_D", 0, 1)
        max_Q = trial.suggest_int("max_Q", 0, 2)
        max_order = trial.suggest_int("max_order", 5, 10)
        seasonal = trial.suggest_categorical("seasonal", [True, False])
        m = trial.suggest_categorical(
            "m", [0, 4, 12]
        )  # Monthly or quarterly data, 0 if not seasonal
        trend = trial.suggest_categorical("trend", [None, "n", "c", "t", "ct"])

        # Stationarity and seasonal tests options could be considered but are typically part of the model's internal checks.

        return {
            "max_p": max_p,
            "max_d": max_d,
            "max_q": max_q,
            "max_P": max_P,
            "max_D": max_D,
            "max_Q": max_Q,
            "max_order": max_order,
            "seasonal": seasonal,
            "m": m,
            "trend": trend,
        }

    def objective(self, trial):
        # Extract hyperparameters
        params = self.get_hyperparameters(trial)

        # Initialize the model with the current set of hyperparameters
        model = AutoARIMA(**params)

        # Fit the model
        model.fit(self.obj_dataset.darts_y_train_target)

        # Evaluate the model (this part is highly dependent on your specific use case)
        predictions = model.predict(len(self.obj_dataset.darts_y_test_target))
        actual = self.obj_dataset.darts_y_test_target

        # Example metric: Mean Squared Error (you could use any relevant metric)
        mse = mean_squared_error(actual.values(), predictions.values())

        # Update the best model if the current model is better
        if (self.best_metric == "minimize" and mse < self.best_model_metric) or (
            self.best_metric == "maximize" and mse > self.best_model_metric
        ):
            self.best_model = model
            self.best_model_metric = mse

        return mse


# Note: This code focuses on the structural changes needed to use AutoARIMA from Darts.
# It assumes that `self.obj_dataset` and utility functions like `utils.get_ts_predictions`
# and `utils.perform_backtest` are adapted to work with the output of AutoARIMA predictions.
