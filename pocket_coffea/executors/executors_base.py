import os
from abc import ABC, abstractmethod
from coffea import processor as coffea_processor
from pocket_coffea.utils.network import get_proxy_path

from coffea.processor import Runner, accumulate, executor
from collections.abc import Awaitable, Generator, Iterable, Mapping, MutableMapping
from typing import (
    Iterable,
    Callable,
    Optional,
    List,
    Set,
    Generator,
    Dict,
    Union,
    Tuple,
    Awaitable,
)
from coffea.processor import ProcessorABC, DaskExecutor, TaskVineExecutor, WorkQueueExecutor, Accumulatable

ExecutorBase = executor.ExecutorBase


import threading
import awkward as ak

from functools import partial
from dataclasses import dataclass, field
import lz4.frame as lz4f

from rich import print as rprint

from coffea.util import save, rich_bar


@dataclass
class IterativeExecutorAlt(ExecutorBase):
    """Execute in one thread iteratively

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Ignored for iterative executor
    """

    workers: int = 1

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        """
        print("start iterative executor")
        print(f"=== THREADING DEBUG ===")
        print(f"Active threads: {threading.active_count()}")
        print(f"Thread names: {[t.name for t in threading.enumerate()]}")
        print(f"Main thread: {threading.current_thread().name}")
        print(f"Current thread ID: {threading.get_ident()}")
        print(f"=======================")

        with rich_bar() as progress:
            print("start rich bar")
            print(f"=== THREADING DEBUG ===")
            print(f"Active threads: {threading.active_count()}")
            print(f"Thread names: {[t.name for t in threading.enumerate()]}")
            print(f"Main thread: {threading.current_thread().name}")
            print(f"Current thread ID: {threading.get_ident()}")
            print(f"=======================")
            p_id = progress.add_task(
                self.desc, total=len(items), unit=self.unit, disable=not self.status
            )
            print("start accumulate")
            print(f"=== THREADING DEBUG ===")
            print(f"Active threads: {threading.active_count()}")
            print(f"Thread names: {[t.name for t in threading.enumerate()]}")
            print(f"Main thread: {threading.current_thread().name}")
            print(f"Current thread ID: {threading.get_ident()}")
            print(f"=======================")
            return (
                accumulate(
                    progress.track(
                        map(function, (c for c in items)),
                        total=len(items),
                        task_id=p_id,
                    ),
                    accumulator,
                ),
                0,
            )
        """

        print("start iterative executor")
        print(f"=== THREADING DEBUG ===")
        print(f"Active threads: {threading.active_count()}")
        print(f"Thread names: {[t.name for t in threading.enumerate()]}")
        print(f"Main thread: {threading.current_thread().name}")
        print(f"Current thread ID: {threading.get_ident()}")
        print(f"=======================")
        results = []
        for item in items:
            print("start function")
            print(f"=== THREADING DEBUG ===")
            print(f"Active threads: {threading.active_count()}")
            print(f"Thread names: {[t.name for t in threading.enumerate()]}")
            print(f"Main thread: {threading.current_thread().name}")
            print(f"Current thread ID: {threading.get_ident()}")
            print(f"=======================")
            print("type of function:", type(function))
            result = function(item)
            print("end function")
            print(f"=== THREADING DEBUG ===")
            print(f"Active threads: {threading.active_count()}")
            print(f"Thread names: {[t.name for t in threading.enumerate()]}")
            print(f"Main thread: {threading.current_thread().name}")
            print(f"Current thread ID: {threading.get_ident()}")
            print(f"=======================")
            results.append(result)
        print("end iterative executor")

        return (
                accumulate(
                    results,
                    accumulator,
                ),
                0,
            )


class ExecutorFactoryABC(ABC):

    def __init__(self, run_options, **kwargs):
        self.run_options = run_options
        self.setup()
        # If handles_submission == False, the executor is not responsible for submitting the jobs
        self.handles_submission = False

    @abstractmethod
    def get(self):
        pass

    def setup(self):
        self.setup_proxyfile()
        self.set_env()

    def setup_proxyfile(self):
        if self.run_options['ignore-grid-certificate']: return
        if vomsproxy:=self.run_options.get('voms-proxy', None) is not None:
             self.x509_path = vomsproxy
        else:
             _x509_localpath = get_proxy_path()
             # Copy the proxy to the home from the /tmp to be used by workers
             self.x509_path = os.environ['HOME'] + f'/{_x509_localpath.split("/")[-1]}'
             if _x509_localpath != self.x509_path:
                 print("Copying proxy file to $HOME.")
                 os.system(f'scp {_x509_localpath} {self.x509_path}')  # scp makes sure older file is overwritten without prompting
             
    def set_env(self):
        # define some environmental variable
        # that are general enought to be always useful
        vars= {
            "XRD_RUNFORKHANDLER": "1",
            "MALLOC_TRIM_THRESHOLD_" : "0",
        }
        if not self.run_options['ignore-grid-certificate']:
            vars["X509_USER_PROXY"] = self.x509_path
        for k,v in vars.items():
            os.environ[k] = v

    def customized_args(self):
        return {}

    def close(self):
        pass
    
class IterativeExecutorFactory(ExecutorFactoryABC):

    def __init__(self, run_options,  **kwargs):
        super().__init__(run_options, **kwargs)

    def get(self):
        return IterativeExecutorAlt(**self.customized_args())


class FuturesExecutorFactory(ExecutorFactoryABC):

    def __init__(self, run_options, **kwargs):
        super().__init__(run_options, **kwargs)

    def get(self):
        return coffea_processor.futures_executor(**self.customized_args())

    def customized_args(self):
        args = super().customized_args()
        # in the futures executor Nworkers == N scaleout
        args["workers"] = self.run_options["scaleout"]
        return args

        

def get_executor_factory(executor_name, **kwargs):
    if executor_name == "iterative":
        return IterativeExecutorFactory(**kwargs)
    elif executor_name == "futures":
        return FuturesExecutorFactory(**kwargs)
