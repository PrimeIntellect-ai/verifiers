"""Native v1 evaluation: each independent invocation produces one agent graph."""

import asyncio
import contextlib
import logging
import random
import time

from verifiers.v1.cli.dashboard import dashboard
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import append_graph, output_path, save_config
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.topology import AgentGraph, resolve_topology_runner

logger = logging.getLogger(__name__)

_SHUFFLE_SEED = 0


def _selected(items: list, config: EvalConfig) -> list:
    if config.shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(items)
    return items if config.num_tasks is None else items[: config.num_tasks]


def graph_traces(graphs: list[AgentGraph]) -> list:
    return [trace for graph in graphs for trace in graph.traces]


async def run_eval(config: EvalConfig) -> list[AgentGraph]:
    """Run taskset syntax or an explicit topology through the same in-process runner."""
    logger.info("eval config:\n%s", config.model_dump_json(indent=2))
    runner = resolve_topology_runner(config)
    tasks = _selected(list(runner.tasks), config)
    out = output_path(config)
    finished: list[AgentGraph] = []
    owed: dict[int, int] | None = None
    if config.resume is not None:
        finished, owed = resume.load(
            out, [task.data.idx for task in tasks], config.num_rollouts
        )
        if not owed:
            print(resume.nothing_to_resume_msg(out, len(tasks), config.num_rollouts))
            raise SystemExit(0)
        tasks = [task for task in tasks if owed.get(task.data.idx)]
        logger.info(
            "resuming %s: %d task(s), %d invocation(s) owed",
            out,
            len(tasks),
            sum(owed.values()),
        )
    else:
        save_config(config, out)
        logger.info(
            "running %dx%d topology invocation(s) on %s",
            len(tasks),
            config.num_rollouts,
            config.model,
        )
    logger.info("results: %s", out)
    client = resolve_client(config.client)
    ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    start = time.time()
    write_lock = asyncio.Lock()
    rollouts = [resume.Finished(trace) for trace in graph_traces(finished)]

    async def run_instance(task) -> AgentGraph:
        graph = await runner.run_instance(
            task,
            ctx,
            semaphore,
            on_rollout=rollouts.append if config.rich else None,
        )
        await append_graph(out, graph, write_lock)
        return graph

    push_state = None
    if config.push and config.rich:
        from verifiers.v1.push import PushState

        push_state = PushState()
    display = (
        dashboard(rollouts, config, start, push=push_state)
        if config.rich
        else contextlib.nullcontext()
    )
    try:
        async with runner.serving(), display:
            instances = [
                run_instance(task)
                for task in tasks
                for _ in range(owed[task.data.idx] if owed else config.num_rollouts)
            ]
            completed = await asyncio.gather(*instances)
            graphs = finished + completed
            if push_state is not None:
                from verifiers.v1.push import push_traces

                push_state.started = True
                await asyncio.to_thread(
                    push_traces, graph_traces(graphs), config, push_state
                )
        return graphs
    finally:
        await client.close()


async def run_eval_server(config: EvalConfig) -> list[AgentGraph]:
    """Run independent graph invocations through the existing env-server worker pool."""
    import multiprocessing as mp
    from functools import partial

    from verifiers.v1.env import pool_serve_kwargs
    from verifiers.v1.serve import EnvClient, env_config_data, serve_env
    from verifiers.v1.utils.logging import setup_logging

    server_kwargs = {"config_data": env_config_data(config)}
    level = "DEBUG" if config.verbose else "INFO"
    log_file = str(output_path(config) / "eval.log")
    mpctx = mp.get_context("spawn")
    address_queue: mp.Queue = mpctx.Queue()
    parent_conn, child_conn = mpctx.Pipe()
    proc = mpctx.Process(
        target=serve_env,
        kwargs=dict(
            **pool_serve_kwargs(config.pool),
            legacy=False,
            address="tcp://127.0.0.1:0",
            address_queue=address_queue,
            death_pipe=child_conn,
            log_setup=partial(setup_logging, level, log_file),
            **server_kwargs,
        ),
        daemon=False,
    )
    proc.start()
    child_conn.close()
    client = None
    try:
        address = await asyncio.to_thread(address_queue.get, timeout=600)
        client = EnvClient(address=address)
        await client.wait_for_server_startup(timeout=600)
        info = await client.info()
        positions = _selected(list(range(info.num_tasks)), config)
        task_ids = [info.task_ids[position] for position in positions]
        out = output_path(config)
        finished: list[AgentGraph] = []
        if config.resume is not None:
            finished, owed = resume.load(out, task_ids, config.num_rollouts)
            if not owed:
                print(
                    resume.nothing_to_resume_msg(
                        out, len(positions), config.num_rollouts
                    )
                )
                raise SystemExit(0)
            selected = [
                (position, task_id)
                for position, task_id in zip(positions, task_ids)
                if owed.get(task_id)
            ]
            logger.info(
                "resuming %s: %d task(s), %d invocation(s) owed",
                out,
                len(selected),
                sum(owed.values()),
            )
        else:
            selected = list(zip(positions, task_ids))
            owed = {task_id: config.num_rollouts for task_id in task_ids}
            save_config(config, out)
            logger.info(
                "running %dx%d topology invocation(s) via the env-server %s pool on %s",
                len(selected),
                config.num_rollouts,
                config.pool.type,
                config.model,
            )
        logger.info("results: %s", out)
        semaphore = (
            asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
        )
        write_lock = asyncio.Lock()

        async def run_unit(position: int) -> AgentGraph:
            async with semaphore or contextlib.nullcontext():
                graph = await client.run(
                    task_idx=position,
                    client=config.client,
                    model=config.model,
                    sampling=config.sampling,
                )
            await append_graph(out, graph, write_lock)
            return graph

        units = [
            run_unit(position)
            for position, task_id in selected
            for _ in range(owed[task_id])
        ]
        completed = await asyncio.gather(*units)
        return finished + completed
    finally:
        if client is not None:
            await client.close()
        proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.join, 10)
        with contextlib.suppress(Exception):
            parent_conn.close()
