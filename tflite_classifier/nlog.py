from typing import Optional
import json
import queue
import logging
import datetime
import threading
import traceback
from collections import OrderedDict
import os
import shutil

from config import read_config

LOGGING_FORMAT = '%(asctime)s: %(module)s: %(funcName)s: %(levelname)s:  %(message)s'

config = read_config()


class LogFormatter(logging.Formatter):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def format(self, record):
        data = OrderedDict()
        data['timestamp'] = datetime.datetime.utcnow().strftime(
            '%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        data['level'] = record.levelname
        data['version'] = '0.0.1'
        data['module'] = 'semantic'
        data['environment'] = 'dev'
        data['function'] = record.funcName
        data['filename'] = record.filename
        data['lineno'] = record.lineno
        if type(record.msg) == tuple and len(record.msg) > 1:
            data['description'] = record.msg[0]
            data['message'] = record.msg[1]
        elif type(record.msg) == str:
            data['description'] = 'External log'
            data['message'] = record.msg % record.args
        return json.dumps(data)


class LogEventHandlerBase:
    def on_log_event(self, log_record):
        pass

    def close(self):
        pass


class AsyncLogger(logging.Handler):
    q = queue.Queue()  # type: queue.Queue
    worker_thread = None

    def __init__(self, log_event_handler):
        super().__init__()
        self.log_event_handler = log_event_handler

    def start(self):
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        if self.worker_thread:
            print('Stopping async logger')
            self.q.put(None)
            self.worker_thread.join()
            self.worker_thread = None

        if self.log_event_handler:
            self.log_event_handler.close()

    def emit(self, record):
        if not self.worker_thread:
            self.start()

        msg = self.format(record)
        self.q.put(msg)

    def _worker(self):
        while True:
            msg = self.q.get()
            if msg is None:
                break
            self._process_message(msg)
            self.q.task_done()

        print('Exiting async logger worker..')

    def _process_message(self, msg):
        if not self.log_event_handler:
            return

        try:
            self.log_event_handler.on_log_event(msg)
        except Exception:  # noqa
            print('Exception while processing the log event.')
            print(traceback.format_exc())


current_handler = None  # type: Optional[logging.Handler]


def get_current_ts():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def init_stdout_logger(log_level):
    logging.getLogger().setLevel(log_level)
    logging.basicConfig(
        format='%(levelname)s:%(funcName)s:%(message)s', level=log_level)


def init_async_logger(log_level, log_event_handler, config):
    global current_handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    handler = AsyncLogger(log_event_handler)
    handler.setFormatter(LogFormatter(config))

    if not current_handler:
        root_logger.removeHandler(current_handler)

    current_handler = handler
    root_logger.addHandler(handler)


def stop_logger():
    if not current_handler:
        return

    if not isinstance(current_handler, AsyncLogger):
        return

    async_logger = current_handler  # type: AsyncLogger
    async_logger.stop()


log_filepath = None
if os.path.exists('training_dummy_file'):
    log_dir = config.get('LOG_DIR', 'logs/')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print('Cleared logs directory')
    os.makedirs(log_dir)
    print('Created logs directory')
    log_filepath = os.path.join(log_dir, f'{get_current_ts()}.log')

logging.basicConfig(
    format=LOGGING_FORMAT,
    level=config.get('LOG_LEVEL', 'INFO'),
    filename=log_filepath)

if log_filepath:
    strm_handler = logging.StreamHandler()
    fmtr = logging.Formatter(LOGGING_FORMAT)
    strm_handler.setFormatter(fmtr)
    logging.getLogger().addHandler(strm_handler)

logger = logging.getLogger(__name__)

debug = logger.debug
info = logger.info
warning = logger.warning
warn = logger.warning
error = logger.error
exception = logger.exception
log = logger.log
