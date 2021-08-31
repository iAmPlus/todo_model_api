import cProfile
import pstats
import io
import logging
import os
import time
import multiprocessing
from pympler import tracker
from tflite_classifier.application import get_application

PROFILE_LIMIT = int(os.environ.get("PROFILE_LIMIT", 30))
PROFILER = bool(int(os.environ.get("PROFILER", 1)))
PROFILE_TIMING = bool(int(os.environ.get("PROFILE_TIMING", 1)))
PROFILE_MEMORY = bool(int(os.environ.get("PROFILE_MEMORY", 1)))

USE_GEVENT = bool(int(os.environ.get("USE_GEVENT", 0)))

#workers = int(os.environ.get("NUM_WORKERS", multiprocessing.cpu_count()))
workers = get_application().get_config('NUM_WORKERS')

if USE_GEVENT is True:
        worker_class = 'gevent'
        worker_connections = 2000
        backlog = 1000
else:
        #print("GEVENT is False. coming in else")
        #threads = int(os.environ.get("WORKER_CONCURRENCY", multiprocessing.cpu_count()))
        #print("no of threads are::", threads)
        threads = get_application().get_config('WORKER_CONCURRENCY')

bind = "0.0.0.0:8185"
timeout = 6000

if (PROFILER is True) and (PROFILE_MEMORY is True):
        tr = tracker.SummaryTracker()

def profiler_enable(worker, req):
    worker.profile = cProfile.Profile()
    worker.profile.enable()
    worker.log.info("PROFILING %d: %s" % (worker.pid, req.uri))

def profiler_summary(worker, req):
    s = io.StringIO()
    worker.profile.disable()
    ps = pstats.Stats(worker.profile, stream=s).sort_stats('time', 'cumulative')
    ps.print_stats(PROFILE_LIMIT)

    logging.error("\n[%d] [INFO] [%s] URI %s" % (worker.pid, req.method, req.uri))
    logging.error("[%d] [INFO] %s" % (worker.pid, s.getvalue()))
    if PROFILE_MEMORY is True:
        logging.error("[%d] [INFO] %s" % (worker.pid, "----- Memory Heap Stats -----"))
        memDiffLines = tr.format_diff()
        for line in memDiffLines:
                logging.error("[%d] [INFO] %s" % (worker.pid, line))
        logging.error("[%d] [INFO] %s" % (worker.pid, "-----------------------------"))

def pre_request(worker, req):    
    if PROFILER is True:
        worker.start_time = time.time()    
        if PROFILE_TIMING is True:
                profiler_enable(worker, req)

def post_request(worker, req, *args):
    if PROFILER is True:
        total_time = time.time() - worker.start_time
        logging.error("\n[%d] [INFO] [%s] Load Time: %.3fs\n" % (
                worker.pid, req.method, total_time))
        if PROFILE_TIMING is True:                        
                profiler_summary(worker, req)