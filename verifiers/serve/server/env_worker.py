"""Environment worker.

Owns a single environment instance, a client cache, and three ZMQ
sockets (PULL for requests, PUSH for responses, PUSH for stats).
Receives requests from the router, runs rollouts, and pushes
responses + stats back.
"""

import asyncio
import faulthandler
import gc
import logging
import os
import signal
import sys
import threading
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import msgpack
import zmq
import zmq.asyncio
from pydantic import BaseModel

import verifiers as vf
from verifiers.clients import Client, resolve_client
from verifiers.serve.types import (
    BaseResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.types import ClientConfig
from verifiers.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.loop_debug import looptime
from verifiers.utils.process_utils import monitor_death_pipe, set_proc_title
from verifiers.utils.serve_utils import msgpack_encoder


# ── Event-loop diagnostics (VF_LOOP_DEBUG) ──────────────────────────────────
# Attribute env-worker event-loop stalls. ON by default; set VF_LOOP_DEBUG=0 to
# disable. Low overhead: a faulthandler watchdog only dumps when the loop is
# *already* stalled past the threshold, and the gc callback only logs slow
# collections. VF_LOOP_DEBUG_ASYNCIO=1 additionally enables asyncio debug mode
# (slow-callback logging with source) — heavier, opt-in.
_loopdbg_log = logging.getLogger("vf.loopdbg")

# Kept alive so faulthandler's fatal-signal file isn't GC-closed (set in run_worker).
_FAULTHANDLER_FILE = None


def _loopdbg_on() -> bool:
    return os.getenv("VF_LOOP_DEBUG", "1").strip().lower() not in (
        "0", "off", "false", "no", "",
    )


def _loopdbg_lag_s() -> float:
    try:
        return float(os.getenv("VF_LOOP_DEBUG_LAG_S", "1.5") or 1.5)
    except ValueError:
        return 1.5


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * (os.sysconf("SC_PAGE_SIZE") / 1e6)
    except Exception:
        return -1.0


def _install_gc_debug(worker_name: str) -> None:
    """Log every gc collection that takes >VF_LOOP_DEBUG_GC_S (default 0.2s),
    with generation, duration, and current RSS — to confirm/refute growing
    stop-the-world gen-2 pauses as a loop-stall source."""
    try:
        thresh = float(os.getenv("VF_LOOP_DEBUG_GC_S", "0.2") or 0.2)
    except ValueError:
        thresh = 0.2
    state = {"t0": 0.0}

    def _gc_cb(phase: str, info: dict) -> None:
        if phase == "start":
            state["t0"] = time.perf_counter()
            return
        dt = time.perf_counter() - state["t0"]
        if dt >= thresh:
            counts = gc.get_count()
            _loopdbg_log.warning(
                "GC %s gen=%s dt=%.2fs rss=%.0fMB counts=%s collected=%s",
                worker_name, info.get("generation"), dt, _rss_mb(),
                counts, info.get("collected"),
            )

    gc.callbacks.append(_gc_cb)


class EnvWorkerStats(BaseModel):
    worker_id: int
    timestamp: float
    active_tasks: int
    lag: EventLoopLagStats = EventLoopLagStats()

    def __str__(self) -> str:
        parts = []
        if self.lag.n > 0:
            parts.append(f"Lag: {self.lag}")
        return " | ".join(parts) if parts else "no lag data"


def _cap_native_threads() -> None:
    """Best-effort runtime cap for env-worker native thread pools."""

    from verifiers.utils.native_threads import configure_runtime_native_threads

    configure_runtime_native_threads()


class EnvWorker:
    """Executes environment logic."""

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_dir: str | None = None,
        console_logging: bool = True,
        json_logging: bool = False,
        *,
        worker_id: int,
        worker_name: str,
        request_address: str,
        response_address: str,
        stats_address: str,
        death_pipe: Connection | None = None,
    ):
        set_proc_title(f"EnvWorker{worker_id}")
        _cap_native_threads()
        self.death_pipe = death_pipe
        self.env_id = env_id
        self.worker_id = worker_id
        self.worker_name = worker_name

        # setup logging — each worker gets its own log file
        logger_kwargs: dict[str, Any] = {
            "console_logging": console_logging,
            "file_logging": log_dir is not None,
            "json_logging": json_logging,
        }
        if log_level is not None:
            logger_kwargs["level"] = log_level
        if log_dir is not None:
            worker_log_file = EnvWorker.get_log_file(log_dir, worker_id)
            worker_log_file.parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["log_file"] = str(worker_log_file)
        vf.setup_logging(**logger_kwargs)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # setup env
        self.logger.debug(
            f"Loading environment {env_id} for worker {worker_name} ({env_args=}, {extra_env_kwargs=})"
        )
        self.env = vf.load_environment(env_id, **(env_args or {}))
        if extra_env_kwargs:
            self.env.set_kwargs(**extra_env_kwargs)

        # setup zmq sockets
        self.ctx = zmq.asyncio.Context()

        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 0)
        self.pull_socket.setsockopt(zmq.LINGER, 0)
        self.pull_socket.bind(request_address)

        self.response_socket = self.ctx.socket(zmq.PUSH)
        self.response_socket.setsockopt(zmq.SNDHWM, 0)
        self.response_socket.setsockopt(zmq.LINGER, 5000)
        self.response_socket.connect(response_address)

        self.stats_socket = self.ctx.socket(zmq.PUSH)
        self.stats_socket.setsockopt(zmq.SNDHWM, 100)
        self.stats_socket.setsockopt(zmq.LINGER, 0)
        self.stats_socket.connect(stats_address)

        # state tracking
        self.clients: dict[str, Client] = {}
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.shutting_down: bool = False

        # stats
        self.lag_monitor = EventLoopLagMonitor()

        self.logger.info(f"Initialized worker {worker_name} on {request_address}")

    async def resolve_client(self, client_config: ClientConfig) -> Client:
        """Resolve the client instance given the request client config."""
        resolved = resolve_client_config(client_config)
        key = resolved.model_dump_json()
        if key not in self.clients:
            self.clients[key] = resolve_client(resolved)
        return self.clients[key]

    async def handle_run_rollout(
        self, request: RunRolloutRequest
    ) -> RunRolloutResponse:
        client = await self.resolve_client(request.client_config)
        output = await self.env.run_rollout(
            input=request.input,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
        )
        return RunRolloutResponse(output=output)

    async def handle_run_group(self, request: RunGroupRequest) -> RunGroupResponse:
        client = await self.resolve_client(request.client_config)
        outputs = await self.env.run_group(
            group_inputs=request.group_inputs,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
        )
        return RunGroupResponse(outputs=outputs)

    async def process_request(
        self,
        client_id: bytes,
        request_id_bytes: bytes,
        payload_bytes: bytes,
    ) -> None:
        request_id = request_id_bytes.decode()
        response: BaseResponse

        async def send_error_response(error: str) -> None:
            """Serialize and send an error response. Best-effort."""
            try:
                response_bytes = cast(
                    bytes,
                    msgpack.packb(
                        BaseResponse(success=False, error=error).model_dump(
                            mode="python", warnings=False
                        ),
                        default=msgpack_encoder,
                        use_bin_type=True,
                    ),
                )
                await self.response_socket.send_multipart(
                    [client_id, request_id.encode(), response_bytes]
                )
            except Exception:
                pass

        try:
            with looptime("req_msgpack_unpack"):
                raw = await asyncio.to_thread(msgpack.unpackb, payload_bytes, raw=False)
            request_type = raw.get("request_type")
            request_id = raw.get("request_id", request_id)

            if request_type == "run_rollout":
                with looptime("req_model_validate"):
                    request = await asyncio.to_thread(
                        RunRolloutRequest.model_validate, raw
                    )
                response = await self.handle_run_rollout(request)
            elif request_type == "run_group":
                with looptime("req_model_validate"):
                    request = await asyncio.to_thread(
                        RunGroupRequest.model_validate, raw
                    )
                response = await self.handle_run_group(request)
            else:
                self.logger.warning(f"Unknown request type: {request_type}")
                response = BaseResponse(
                    success=False, error=f"Unknown request type: {request_type}"
                )

        except asyncio.CancelledError:
            if self.shutting_down:
                return
            # shield prevents the still-set cancellation flag from killing the await
            await asyncio.shield(send_error_response("Request was cancelled"))
            return

        except Exception as e:
            self.logger.error(
                f"Error processing request {request_id}: {e}", exc_info=True
            )
            await send_error_response(repr(e))
            return

        try:
            with looptime("req_response_serialize"):
                response_bytes = await asyncio.to_thread(
                    lambda: cast(
                        bytes,
                        msgpack.packb(
                            response.model_dump(mode="python", warnings=False),
                            default=msgpack_encoder,
                            use_bin_type=True,
                        ),
                    )
                )
        except Exception as e:
            self.logger.error(
                f"Failed to serialize response for {request_id}: {e}",
                exc_info=True,
            )
            await send_error_response(f"Response serialization failed: {repr(e)}")
            return

        try:
            await self.response_socket.send_multipart(
                [client_id, request_id.encode(), response_bytes]
            )
        except zmq.ZMQError as e:
            self.logger.warning(f"Failed to send response for {request_id[:7]}: {e}")

    async def stats_loop(self, interval: float = 10.0) -> None:
        """Loop to push worker stats to the router."""
        while True:
            await asyncio.sleep(interval)

            # Force-RE-REGISTER faulthandler so a lazily-imported native lib
            # (torch/vLLM) that installed its own SIGSEGV handler can't keep it:
            # enable() while already-enabled is a no-op (won't re-sigaction), so
            # disable() first. Last sigaction wins -> ours. Writes to fd 2
            # (redirected to the crash file in run_worker).
            try:
                faulthandler.disable()
                faulthandler.enable(all_threads=True)
            except Exception:
                pass

            stats = EnvWorkerStats(
                worker_id=self.worker_id,
                timestamp=time.time(),
                active_tasks=len(self.active_tasks),
                lag=EventLoopLagStats.from_monitor(self.lag_monitor),
            )

            try:
                data = msgpack.packb(
                    stats.model_dump(mode="python"),
                    default=msgpack_encoder,
                    use_bin_type=True,
                )
                await self.stats_socket.send(data, zmq.NOBLOCK)
            except zmq.Again:
                pass  # best-effort

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Main worker loop."""
        self.logger.info(f"Starting worker {self.worker_name}")

        gc.collect()
        gc.freeze()
        gc.set_threshold(150_000, 10, 10)

        loopdbg = _loopdbg_on()
        watchdog_task: asyncio.Task | None = None
        if loopdbg:
            _install_gc_debug(self.worker_name)
            # asyncio debug (per-callback slow-call logging w/ source) default-ON
            # in this diagnostic build; set VF_LOOP_DEBUG_ASYNCIO=0 to disable.
            # NOTE: adds per-callback overhead — diagnostic builds only.
            if os.getenv("VF_LOOP_DEBUG_ASYNCIO", "1").strip().lower() not in (
                "0", "off", "false", "no", "",
            ):
                loop = asyncio.get_event_loop()
                loop.set_debug(True)
                try:
                    loop.slow_callback_duration = float(
                        os.getenv("VF_LOOP_DEBUG_SLOW_S", "0.25") or 0.25
                    )
                except ValueError:
                    loop.slow_callback_duration = 0.25
            watchdog_task = asyncio.create_task(self._loop_watchdog())
            self.logger.info(
                f"VF_LOOP_DEBUG on for {self.worker_name}: lag-dump>="
                f"{_loopdbg_lag_s():.1f}s + gc logging"
            )

        lag_task = asyncio.create_task(self.lag_monitor.run())
        stats_task = asyncio.create_task(self.stats_loop())

        poller = zmq.asyncio.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                try:
                    events = dict(await poller.poll(timeout=100))
                    if self.pull_socket not in events:
                        continue

                    frames = await self.pull_socket.recv_multipart()
                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                        continue

                    raw_client_id, raw_request_id, raw_payload = frames
                    request_id = raw_request_id.decode()

                    if not raw_payload:
                        # Cancel signal
                        task = self.active_tasks.get(request_id)
                        if task is not None:
                            task.cancel()
                        continue

                    task = asyncio.create_task(
                        self.process_request(raw_client_id, raw_request_id, raw_payload)
                    )
                    self.active_tasks[request_id] = task

                    def cleanup_task(task: asyncio.Task, request_id: str) -> None:
                        if self.active_tasks.get(request_id) is task:
                            self.active_tasks.pop(request_id, None)

                    task.add_done_callback(
                        lambda t, rid=request_id: cleanup_task(t, rid)
                    )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in serve loop: {e}", exc_info=True)
        finally:
            poller.unregister(self.pull_socket)
            tasks = [stats_task, lag_task]
            if watchdog_task is not None:
                tasks.append(watchdog_task)
                try:
                    faulthandler.cancel_dump_traceback_later()
                except Exception:
                    pass
                stall_stop = getattr(self, "_stall_stop", None)
                if stall_stop is not None:
                    stall_stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _loop_watchdog(self) -> None:
        """Catch actual event-loop blocks via TWO independent mechanisms:

        1) A background daemon THREAD (``_stall_watchdog_thread``) that watches a
           loop-updated heartbeat; when it goes stale > threshold it logs every
           thread's stack via the logger (the *captured* channel — faulthandler's
           raw stderr writes weren't showing in `prime train logs`). This is the
           reliable capture of "what is the loop thread executing right now".
        2) ``faulthandler.dump_traceback_later`` to **stdout** as a C-level backup
           that fires even under total GIL starvation (when the python watchdog
           thread itself can't get the GIL).

        This coroutine just keeps the loop heartbeat fresh + re-arms faulthandler.
        """
        lag_s = _loopdbg_lag_s()
        interval = max(0.25, lag_s / 4.0)
        self._loop_hb = time.monotonic()
        self._loop_tid = threading.get_ident()  # the event-loop thread
        # start the python watchdog thread once
        if getattr(self, "_stall_thread", None) is None:
            self._stall_stop = threading.Event()
            self._stall_thread = threading.Thread(
                target=self._stall_watchdog_thread,
                args=(lag_s,),
                name=f"loop-watchdog-{self.worker_id}",
                daemon=True,
            )
            self._stall_thread.start()
        while True:
            self._loop_hb = time.monotonic()
            try:
                faulthandler.dump_traceback_later(
                    lag_s, repeat=False, file=sys.stdout, exit=False
                )
            except Exception as exc:
                _loopdbg_log.warning("faulthandler arm failed: %r", exc)
            await asyncio.sleep(interval)

    def _stall_watchdog_thread(self, lag_s: float) -> None:
        """Daemon thread: when the loop heartbeat is stale > lag_s, log all-thread
        stacks via the logger so the blocking frame appears in captured logs."""
        cooldown = 5.0
        last_dump = 0.0
        while not self._stall_stop.is_set():
            loop_tid = getattr(self, "_loop_tid", None) or threading.main_thread().ident
            time.sleep(min(0.25, lag_s / 4.0))
            hb = getattr(self, "_loop_hb", None)
            if hb is None:
                continue
            gap = time.monotonic() - hb
            now = time.monotonic()
            if gap >= lag_s and (now - last_dump) > cooldown:
                last_dump = now
                frames = sys._current_frames()
                _loopdbg_log.warning(
                    "LOOP STALL %s gap=%.1fs rss=%.0fMB — thread stacks follow",
                    self.worker_name, gap, _rss_mb(),
                )
                # loop thread first (the blocker), then the rest
                tids = [loop_tid] + [t for t in frames if t != loop_tid]
                for tid in tids:
                    frame = frames.get(tid)
                    if frame is None:
                        continue
                    stack = "".join(traceback.format_stack(frame, limit=25))
                    tag = "LOOP-THREAD" if tid == loop_tid else f"thread-{tid}"
                    _loopdbg_log.warning(
                        "LOOP STALL %s [%s]:\n%s", self.worker_name, tag, stack
                    )

    async def close(self) -> None:
        self.shutting_down = True
        if self.active_tasks:
            tasks = list(self.active_tasks.values())
            self.logger.info(
                f"Cancelling {len(tasks)} active tasks on worker {self.worker_name}"
            )
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks.clear()

        for client in self.clients.values():
            await client.close()
        self.clients.clear()

        await self.env._teardown()

        self.pull_socket.close()
        self.response_socket.close()
        self.stats_socket.close()
        self.ctx.term()

        self.logger.info(f"Shut down worker {self.worker_name}")

    async def run(self) -> None:
        if self.death_pipe is not None:
            monitor_death_pipe(self.death_pipe)

        from verifiers.utils.thread_utils import (
            install_default_executor,
            scale_executors,
        )

        from verifiers.utils.native_threads import env_worker_max_threads

        # Scale the default executor BEFORE install_default_executor so the
        # event loop picks up a properly-sized pool. Keep this bounded because
        # multimodal renderers use asyncio.to_thread for large image buffers;
        # hundreds of simultaneous preprocessing jobs can create a high RSS
        # watermark even when native BLAS/OpenMP teams are capped.
        max_threads = env_worker_max_threads()
        self.logger.info(
            f"Setting env-worker default executor max_workers={max_threads}"
        )
        scale_executors(concurrency=max_threads)
        install_default_executor()

        stop_event = asyncio.Event()

        def signal_handler(sig, _frame):
            stop_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            await self.serve(stop_event=stop_event)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            await self.close()

    @staticmethod
    def get_log_file(log_dir: str, worker_id: int) -> Path:
        """Return the log file path for a given worker."""
        return Path(log_dir) / f"env_worker_{worker_id}.log"

    @classmethod
    def run_worker(cls, *args, **kwargs) -> None:
        # Workers have died with exitcode=-11 (SIGSEGV): a native crash with no
        # Python traceback. Register faulthandler for fatal signals writing to
        # STDOUT (the captured channel; stderr is not captured by the log
        # pipeline) so the next crash dumps every thread's Python stack and
        # points at the native call site. faulthandler re-raises the signal, so
        # the process still dies with -11 — we just learn where.
        # stdout-at-crash is lost (the process dies before the pipe is shipped),
        # so write the fatal dump to a pod-local FILE the router reads after the
        # worker dies (same container → shared /tmp). Keep the file object alive
        # (module global) so it isn't GC-closed out from under faulthandler.
        # The prior faulthandler->file produced empty dumps (torch/vLLM install
        # their own fatal handler on lazy import, overriding ours, and write to
        # stderr which isn't captured). So REDIRECT the process stderr fd (2) to a
        # pod-local file: then EVERY crash writer — faulthandler, torch/c10's C++
        # backtrace, glibc abort, vLLM — lands in that file, which the router
        # reads after death. Robust to whichever handler wins.
        global _FAULTHANDLER_FILE
        try:
            _FAULTHANDLER_FILE = open(f"/tmp/vf_crash_{os.getpid()}.txt", "w", buffering=1)
            os.dup2(_FAULTHANDLER_FILE.fileno(), 2)  # fd 2 = stderr -> file
            faulthandler.enable(all_threads=True)     # default target = stderr = file
        except Exception as exc:  # never block worker startup on diagnostics
            logging.getLogger("vf.loopdbg").warning("stderr-redirect/faulthandler failed: %r", exc)

        # ROOT-CAUSE PROBE: the -11 stack shows the worker SIGSEGVs importing
        # scipy.stats (pulled by vllm.multimodal on first render) — a native ABI
        # crash. Log the actual image versions of the ABI-coupled packages, then
        # do a guarded `import scipy.stats` self-test. If the worker dies between
        # SELFTEST START and OK, scipy import segfaults deterministically at
        # startup (and we have the version skew to pin).
        # Use raw print->stdout (fd 1, captured): logging handlers aren't
        # configured this early in run_worker, so logger calls go nowhere.
        def _probe(msg):
            print(f"PROBE {msg}", flush=True)
        try:
            import importlib.metadata as _md
            for _p in ("numpy", "scipy", "scikit-image", "torch", "vllm",
                       "transformers", "pillow", "vllm-flash-attn"):
                try:
                    _probe(f"VERSIONS {_p}=={_md.version(_p)}")
                except Exception as _e:
                    _probe(f"VERSIONS {_p}=missing({_e!r})")
            _probe("SELFTEST scipy.stats START")
            import scipy.stats as _scipy_stats  # noqa: F401
            _probe("SELFTEST scipy.stats OK")
            _probe("SELFTEST vllm.multimodal START")
            import vllm.multimodal as _vmm  # noqa: F401
            _probe("SELFTEST vllm.multimodal OK")
        except Exception as _e:
            _probe(f"SELFTEST exception {_e!r}")

        try:
            import uvloop

            uvloop.install()
        except ImportError:
            pass
        worker = cls(*args, **kwargs)
        asyncio.run(worker.run())
