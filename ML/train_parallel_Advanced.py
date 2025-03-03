import warnings
warnings.filterwarnings("ignore")
import os
import psutil
import multiprocessing
import threading
from multiprocessing import Manager, Value, Lock
from StrategySphere.common import utils_common
from StrategySphere.data import data_downloader
from StrategySphere.ml.learners import base_learner as learner
from StrategySphere.ml.utils import dataset, utils_ml
import torch
import time
import sys
import pynvml  # NVIDIA Management Library for GPU usage
import logging
from colorama import Back, Fore, Style, init
from datetime import datetime

# Initialize colorama
init(autoreset=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Estimated CPU and GPU load for each model (values are hypothetical and should be adjusted based on actual profiling)
MODEL_CPU_LOAD = {
	'abr': 20,
	'etr': 25,
	'mlpr': 30,
	'sgdr': 10,
	'ardr': 10,
	'bnbc': 5,
	'knc': 20,
	'svc': 40,
	'tm': 25
}

MODEL_GPU_LOAD = {
	'brnn': 90,
	'tide': 50,
	'nhits': 50,
	'tft': 50,
	'tcn': 50
}

# Categorize models into CPU and GPU
cpu_models = ['abr', 'etr', 'mlpr', 'sgdr', 'ardr', 'bnbc', 'knc', 'svc', 'tm']
gpu_models = ['brnn', 'tide', 'nhits', 'tft', 'tcn']

def train_model(model_name, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue):
	print("\rCreating directories")
	utils_ml.create_models_results_dirs(os.path.join(dir_current, symbol, time_horizon))
	print("\rDirectories created")
	try:

		if model_name in cpu_models:
			print(Fore.WHITE + Back.BLUE + f"\rTraining {model_name} for {symbol} {time_horizon}..." + Style.RESET_ALL)
		else:
			print(Fore.WHITE + Back.GREEN + f"\rTraining {model_name} for {symbol} {time_horizon}..." + Style.RESET_ALL)        
		
		sys.stdout.flush()  # Ensure output is flushed
		obj_learner = learner.BaseLearner(dir_current, symbol, time_horizon, obj_dataset, dict_config)
		obj_learner.train(model_name)
	except Exception as e:
		print(Fore.WHITE + Back.RED + f"\rError during training {model_name}: {e}" + Style.RESET_ALL)
	finally:
		output_queue.put((model_name, ""))  # Put an empty string since we're not capturing the output
		result_queue.put((model_name, MODEL_CPU_LOAD.get(model_name, 0), MODEL_GPU_LOAD.get(model_name, 0)))  # Signal that training for this model is complete

def get_available_cpus():
	return psutil.cpu_count(logical=False)

def get_available_gpus():
	return torch.cuda.device_count()

def get_gpu_load():
	pynvml.nvmlInit()
	handle = pynvml.nvmlDeviceGetHandleByIndex(0)
	utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
	return utilization.gpu

def display_load(current_cpu_load, current_gpu_load):
	while True:
		actual_cpu_load = psutil.cpu_percent(interval=1)
		actual_gpu_load = get_gpu_load()
		print(Fore.WHITE + Back.CYAN + f"\rActual CPU Load: {actual_cpu_load}%, Actual GPU Load: {actual_gpu_load}%" + Style.RESET_ALL, end="")
		time.sleep(5)

def start_process(model_name, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue):
	print(Fore.WHITE + Back.BLUE + f"\nStarting process for model and timehorizon {model_name} {time_horizon}"+ Style.RESET_ALL)
	p = multiprocessing.Process(target=train_model, args=(model_name, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue))
	p.start()
	return (p, model_name, MODEL_CPU_LOAD.get(model_name, 0), MODEL_GPU_LOAD.get(model_name, 0))

def manage_training(models_to_train, max_concurrent, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue, max_cpu_usage, current_cpu_load, cpu_load_lock, max_gpu_usage, current_gpu_load, gpu_load_lock, cpu_models, gpu_models, gpu_available=False):
	active_processes = []
	last_print_time = time.time()

	def try_start_process(model_name):
		nonlocal current_cpu_load, current_gpu_load, last_print_time
		actual_cpu_load = psutil.cpu_percent(interval=0.1)
		actual_gpu_load = get_gpu_load()
		if (current_cpu_load.value + MODEL_CPU_LOAD.get(model_name, 0) <= max_cpu_usage and
			current_gpu_load.value + MODEL_GPU_LOAD.get(model_name, 0) <= max_gpu_usage and
			actual_cpu_load < max_cpu_usage and
			actual_gpu_load < 75):  # Adjusted GPU threshold to 75%
			with cpu_load_lock, gpu_load_lock:
				process_info = start_process(model_name, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue)
				active_processes.append(process_info)
				current_cpu_load.value += MODEL_CPU_LOAD.get(model_name, 0)
				current_gpu_load.value += MODEL_GPU_LOAD.get(model_name, 0)
				# print(Fore.WHITE + Back.BLUE +f"\nStarted process for {('CPU' if model_name in cpu_models else 'GPU')} model: {model_name}"+ Style.RESET_ALL)
				return True
		return False

	def start_models(models, max_concurrent_processes, model_type):
		remaining_models = models[:]
		while remaining_models and len(active_processes) < max_concurrent_processes:
			model_name = remaining_models.pop(0)
			if try_start_process(model_name):
				if model_type == 'CPU':
					time.sleep(60)  # Wait for 1 minute before starting the next CPU model
			else:
				remaining_models.append(model_name)

	# Start CPU models
	start_models([m for m in models_to_train if m in cpu_models], max_concurrent, 'CPU')

	if gpu_available:
		# Start GPU models one by one
		for model_name in [m for m in models_to_train if m in gpu_models]:
			try_start_process(model_name)
			while True:
				finished_model, cpu_load_released, gpu_load_released = result_queue.get()  # Wait for any GPU model to finish
				print(Fore.BLACK + Back.BLUE + f"Finished training for GPU model: {finished_model}"+ Style.RESET_ALL)
				with gpu_load_lock:
					current_gpu_load.value -= gpu_load_released
				if finished_model == model_name:
					break

	while active_processes:
		finished_model, cpu_load_released, gpu_load_released = result_queue.get()  # Wait for any process to finish
		print(Fore.BLACK + Back.BLUE + f"Finished training for model: {finished_model}"+ Style.RESET_ALL)
		active_processes = [(p, m, c, g) for p, m, c, g in active_processes if p.is_alive()]  # Remove finished process from active list
		with cpu_load_lock:
			current_cpu_load.value -= cpu_load_released
		with gpu_load_lock:
			current_gpu_load.value -= gpu_load_released

		# Check if we can start more models
		start_models([m for m in models_to_train if m in cpu_models], max_concurrent, 'CPU')

	for p, _, _, _ in active_processes:
		p.join()

	while not output_queue.empty():
		model_name, output = output_queue.get()
		print(f"\nOutput for model {model_name}:\n{output}")

def train_symbol(symbol, dict_config, max_concurrent=None, max_cpu_usage=95, max_gpu_usage=75):  # Adjusted max_gpu_usage to 75
	if max_concurrent is None:
		max_concurrent = max(1, get_available_cpus() - 1)  # Leave one core free
	print(f"Using {max_concurrent} concurrent processes for CPU models.")

	dir_current = os.path.dirname(os.path.realpath(__file__))
	list_time_horizons = dict_config['data']['time_horizons']
	if not isinstance(list_time_horizons, list):
		list_time_horizons = [list_time_horizons]
	utils_common.validate_time_horizons(list_time_horizons)
	
	list_models_to_train = dict_config['train']['models_to_train']
	if not isinstance(list_models_to_train, list):
		list_models_to_train = [list_models_to_train]
	utils_common.validate_list(list_models_to_train, utils_ml.get_allowed_models(), 'models_to_train')

	# Create a symbol folder if it doesn't exist
	utils_common.create_if_not_exists(os.path.join(dir_current, symbol))


	# Filter models to train based on configuration
	cpu_models_to_train = [model for model in list_models_to_train if model in cpu_models]
	gpu_models_to_train = [model for model in list_models_to_train if model in gpu_models]

	available_gpus = get_available_gpus()
	gpu_available = available_gpus > 0
	if gpu_available:
		print(f"Using {available_gpus} GPUs for GPU models.")
	else:
		print("No GPUs available. Only CPU models will be trained.")

	for time_horizon in list_time_horizons:
		
		now = datetime.now()
		year = now.strftime('%y')       # Last two digits of the year
		day_of_year = now.strftime('%j')  # Day of the year (001 to 366)
		time = now.strftime('%H%M')     # Time in HHMM format
		training_started_time = f"{year}{day_of_year}_{time}"
		dict_config['train']['started_at'] = training_started_time
  
		obj_data_downloader = data_downloader.DataDownloader(
			dict_config['data']['exchange'], symbol, time_horizon,
			dict_config['data']['start_date'], dict_config['data']['end_date'],
			dict_config['data']['fill_missing_method'], dict_config['data']['interpolation_method'],
			dict_config['data']['fill_zero_volume'], dict_config['data']['retries'],
			dict_config['data']['retry_delay'], dict_config['data']['override_existing_data']
		)
		df_symbol, df_1m = obj_data_downloader.get_data_df(return_1m=True)
		utils_common.create_if_not_exists(os.path.join(dir_current, symbol, time_horizon))
		obj_dataset = dataset.Dataset(symbol, time_horizon, df_symbol, dict_config, df_1m)

		# Queue to signal completion of training
		result_queue = multiprocessing.Queue()
		output_queue = multiprocessing.Queue()

		with Manager() as manager:
			current_cpu_load = manager.Value('i', 0)  # Shared variable to track current CPU load
			current_gpu_load = manager.Value('i', 0)  # Shared variable to track current GPU load
			cpu_load_lock = Lock()  # Lock for safely updating the CPU load
			gpu_load_lock = Lock()  # Lock for safely updating the GPU load

			# Start load display in a separate thread
			load_display_thread = threading.Thread(target=display_load, args=(current_cpu_load, current_gpu_load))
			load_display_thread.daemon = True  # Ensures the thread will exit when the main program exits
			load_display_thread.start()

			# Start CPU model training in a separate thread
			cpu_training_thread = threading.Thread(target=manage_training, args=(cpu_models_to_train, max_concurrent, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue, max_cpu_usage, current_cpu_load, cpu_load_lock, max_gpu_usage, current_gpu_load, gpu_load_lock, cpu_models, gpu_models, gpu_available))
			cpu_training_thread.start()

			# Start GPU model training if GPUs are available
			if gpu_available and gpu_models_to_train:
				gpu_max_concurrent = available_gpus
				for gpu_model in gpu_models_to_train:
					print(Fore.BLACK + Back.GREEN + f"Started process for GPU model: {gpu_model}"+ Style.RESET_ALL)
					p, model_name, cpu_load, gpu_load = start_process(gpu_model, dir_current, symbol, time_horizon, obj_dataset, dict_config, result_queue, output_queue)
					p.join()  # Wait for the GPU model to complete
					print(Fore.WHITE + Back.GREEN +f"Finished process for GPU model: {model_name}"+ Style.RESET_ALL)
					current_gpu_load.value -= gpu_load

			# Wait for CPU training to complete
			cpu_training_thread.join()
			print("Finished CPU training thread.")

			# Print output for each model
			while not output_queue.empty():
				model_name, output = output_queue.get()
				print(f"\nOutput for model {model_name}:\n{output}")
