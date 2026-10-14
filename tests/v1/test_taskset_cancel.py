"""`Taskset.cancel`: the channel a taskset uses to ask its runner to stop a dispatched
task, and the `on_cancelled` hook the runner answers on."""

import itertools
import threading

import verifiers.v1 as vf


class CountTask(vf.Task[vf.TaskData]):
    pass


class Stream(vf.Taskset[CountTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self):
        for i in itertools.count():
            yield CountTask(vf.TaskData(id=f"id-{i}", name=f"t{i}", prompt=f"task {i}"))


def test_cancelled_returns_each_requested_key_once_in_order() -> None:
    taskset = Stream(vf.TasksetConfig())
    assert taskset.cancelled() == []
    taskset.cancel("b")
    taskset.cancel("a")
    assert taskset.cancelled() == ["b", "a"]
    assert taskset.cancelled() == []


def test_a_view_shares_the_cancel_channel() -> None:
    taskset = Stream(vf.TasksetConfig())
    view = taskset.take(3)
    # the taskset asks on the object that runs load(); the runner polls the view it iterates
    taskset.cancel("x")
    assert view.cancelled() == ["x"]
    view.cancel("y")
    assert taskset.cancelled() == ["y"]


def test_cancel_is_safe_from_other_threads() -> None:
    taskset = Stream(vf.TasksetConfig())
    threads = [
        threading.Thread(
            target=lambda n=n: [taskset.cancel(f"{n}-{i}") for i in range(200)]
        )
        for n in range(4)
    ]
    for thread in threads:
        thread.start()
    drained: list[str] = []
    while any(thread.is_alive() for thread in threads):
        drained.extend(taskset.cancelled())
    for thread in threads:
        thread.join()
    drained.extend(taskset.cancelled())
    assert sorted(drained) == sorted(f"{n}-{i}" for n in range(4) for i in range(200))


def test_on_cancelled_defaults_to_a_no_op() -> None:
    taskset = Stream(vf.TasksetConfig())
    task = next(iter(taskset))
    assert taskset.on_cancelled(task) is None


def test_a_taskset_whose_init_skips_super_still_cancels() -> None:
    class Bare(Stream):
        def __init__(self, config: vf.TasksetConfig) -> None:
            self.config = config  # no super().__init__()

    taskset = Bare(vf.TasksetConfig())
    assert taskset.cancelled() == []
    taskset.cancel("a")
    assert taskset.cancelled() == ["a"]
