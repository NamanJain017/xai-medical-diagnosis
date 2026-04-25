"""
Logging utilities for clean terminal output.
"""

import logging
import sys

def setup_logging(log_file=None):
    """Setup standard logging to console and optionally to a file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    
    # Silence third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

class TermColor:
    """Terminal colors for pretty printing."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{TermColor.BOLD}{TermColor.OKCYAN}=== {text} ==={TermColor.ENDC}")

def print_success(text):
    print(f"{TermColor.OKGREEN}✓ {text}{TermColor.ENDC}")

def print_warning(text):
    print(f"{TermColor.WARNING}⚠ {text}{TermColor.ENDC}")

def print_error(text):
    print(f"{TermColor.FAIL}✗ {text}{TermColor.ENDC}")
