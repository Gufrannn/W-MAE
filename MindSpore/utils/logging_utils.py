import os
import logging
import mindspore
import x2ms_adapter

_format = '%(asctime)s-%(name)s-%(levelname)s-{}-%(message)s'

def config_logger(log_level=logging.INFO, notes=""):
  new_format = _format.format(notes)
  logging.basicConfig(format=new_format, level=log_level)

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log', notes=""):
  
  new_format = _format.format(notes)

  if not os.path.exists(os.path.dirname(log_filename)):
    try:
      os.makedirs(os.path.dirname(log_filename))
    except:
      pass 
  if logger_name is not None:
    log = logging.getLogger(logger_name)
  else:
    log = logging.getLogger()

  fh = logging.FileHandler(log_filename)
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter(new_format))
  log.addHandler(fh)

  # print_on_screen:
  # console = logging.StreamHandler()
  # console.setLevel(logging.INFO)
  # formatter = logging.Formatter(new_format)
  # console.setFormatter(formatter)
  # log.addHandler(console)

def log_versions():
  import subprocess

  logging.info('--------------- Versions ---------------')
  try:
    logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
    logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
  except:
    logging.info("No Git")
  logging.info('Torch: ' + str(mindspore.__version__))
  logging.info('----------------------------------------')
