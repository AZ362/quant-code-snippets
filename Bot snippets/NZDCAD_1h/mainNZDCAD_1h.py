import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# Import the necessary modules
from Ini_Ticket_new_multi4_2 import *
if __name__ == "__main__":
    bot = MeanReversionBot("config.json")
    bot.run()