"""
LiveCodeBench Real Dataset Implementation for Verifiers

This implementation uses the ACTUAL LiveCodeBench dataset from HuggingFace
with proper Docker-based sandboxing for secure code execution.
"""

import atexit
import base64
import json
import pickle
import re
import signal
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import docker  # type: ignore
from datasets import Dataset, load_dataset

import verifiers as vf


class SecureSandboxExecutor:
    """Secure Docker-based code execution with container pooling"""

    def __init__(self, pool_size: int = 20):
        """Initialize Docker-only sandbox with container pool"""
        try:
            self.docker_client = docker.from_env()

            # Test Docker access
            self.docker_client.ping()

            # Clean up any orphaned containers from previous runs
            self._cleanup_orphaned_containers()

            # Pull Python image if not available
            try:
                self.docker_client.images.get("python:3.11-slim")
            except docker.errors.ImageNotFound:  # type: ignore
                print("Pulling Python Docker image...")
                self.docker_client.images.pull("python:3.11-slim")

            # Initialize container pool
            self.container_pool = ContainerPool(self.docker_client, pool_size=pool_size)

            # Register cleanup handlers
            self._register_cleanup_handlers()

            print("Docker sandbox with container pool initialized successfully")

        except Exception as e:
            raise RuntimeError(
                f"Docker is required but not available: {e}\n"
                "Please ensure Docker is installed and running:\n"
                "  - Install Docker: https://docs.docker.com/get-docker/\n"
                "  - Start Docker daemon: sudo systemctl start docker\n"
                "  - Add user to docker group: sudo usermod -aG docker $USER"
            )

    def _cleanup_orphaned_containers(self):
        """Clean up any containers from previous runs"""
        try:
            # Find all containers with our label
            containers = self.docker_client.containers.list(
                all=True, filters={"label": "livecodebench=true"}
            )

            if containers:
                print(
                    f"Cleaning up {len(containers)} orphaned containers from previous runs..."
                )
                for container in containers:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: Failed to clean up orphaned containers: {e}")

    def _register_cleanup_handlers(self):
        """Register cleanup handlers for various exit scenarios"""
        # Register atexit handler
        atexit.register(self.cleanup)

        # Register signal handlers
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up...")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def cleanup(self):
        """Explicitly clean up all resources"""
        print("Cleaning up Docker resources...")

        # Shutdown container pool
        if hasattr(self, "container_pool"):
            self.container_pool.shutdown()

        # Run Docker system prune to clean up unused resources
        try:
            # Remove unused containers, networks, and build cache
            self.docker_client.containers.prune(filters={"label": "livecodebench=true"})

            # Optional: More aggressive cleanup (commented out by default)
            # self.docker_client.images.prune()
            # self.docker_client.volumes.prune()
            # self.docker_client.networks.prune()

        except Exception as e:
            print(f"Warning: Failed to prune Docker resources: {e}")

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False

    def execute_in_container(
        self, container, code: str, test_input: str = "", timeout: int = 10
    ) -> Tuple[str, str, int]:
        """Execute code in a pre-allocated container"""

        # Create exec command with code and input
        exec_command = f"""
import sys
from io import StringIO

# Set up stdin
sys.stdin = StringIO({repr(test_input)})

# Capture stdout
old_stdout = sys.stdout
sys.stdout = StringIO()

exit_code = 0
stderr_output = ""

try:
    exec({repr(code)})
    stdout_output = sys.stdout.getvalue()
except Exception as e:
    import traceback
    stderr_output = traceback.format_exc()
    stdout_output = sys.stdout.getvalue()
    exit_code = 1
finally:
    sys.stdout = old_stdout

# Output results in a parseable format
print("===STDOUT_START===")
print(stdout_output)
print("===STDOUT_END===")
print("===STDERR_START===") 
print(stderr_output)
print("===STDERR_END===")
print("===EXIT_CODE===")
print(exit_code)
"""

        try:
            # Execute in container
            exec_result = container.exec_run(
                cmd=["python", "-c", exec_command],
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                privileged=False,
                user="1000:1000",
                detach=False,
                stream=False,
                socket=False,
            )

            output = exec_result.output.decode("utf-8")

            # Parse output
            stdout = ""
            stderr = ""
            exit_code = 1

            if "===STDOUT_START===" in output:
                stdout = (
                    output.split("===STDOUT_START===")[1]
                    .split("===STDOUT_END===")[0]
                    .strip()
                )
            if "===STDERR_START===" in output:
                stderr = (
                    output.split("===STDERR_START===")[1]
                    .split("===STDERR_END===")[0]
                    .strip()
                )
            if "===EXIT_CODE===" in output:
                exit_code = int(output.split("===EXIT_CODE===")[1].strip())

            return stdout, stderr, exit_code

        except Exception as e:
            return "", str(e), 1


def load_livecodebench_dataset(
    version_tag: str = "release_v5", split: str = "test", num_examples: int = -1
) -> Dataset:
    """Load the ACTUAL LiveCodeBench dataset from HuggingFace"""

    dataset_name = "livecodebench/code_generation_lite"

    print(f"Loading LiveCodeBench dataset: {dataset_name}")
    print(f"Version: {version_tag}")

    # Map version tags to JSONL files
    version_to_file = {
        "release_v1": "test.jsonl",
        "release_v2": "test2.jsonl",
        "release_v3": "test3.jsonl",
        "release_v4": "test4.jsonl",
        "release_v5": "test5.jsonl",
        "release_v6": "test6.jsonl",
    }

    # Get the appropriate file
    jsonl_file = version_to_file.get(version_tag, "test5.jsonl")

    print(f"Downloading {jsonl_file} from HuggingFace...")

    try:
        import json
        import urllib.request

        # Download JSONL file from HuggingFace
        url = (
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{jsonl_file}"
        )

        examples = []
        with urllib.request.urlopen(url) as response:
            # Read line by line to handle large files
            for line_num, line in enumerate(response):
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)

                        # Stop if we've reached the desired number
                        if num_examples > 0 and len(examples) >= num_examples:
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue

        print(f"Dataset loaded: {len(examples)} total problems")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(examples)

        return dataset

    except Exception as e:
        print(f"Error loading LiveCodeBench dataset: {e}")

        # Fallback: Try the standard datasets library approaches
        print("Attempting standard loading approaches...")

        # Try various loading methods
        for attempt_name, load_func in [
            (
                "with trust_remote_code",
                lambda: load_dataset(dataset_name, split=split, trust_remote_code=True),
            ),
            (
                "without trust_remote_code",
                lambda: load_dataset(dataset_name, split=split),
            ),
            (
                "with config",
                lambda: load_dataset(dataset_name, version_tag, split=split),
            ),
            ("basic", lambda: load_dataset(dataset_name, split=split)),
        ]:
            try:
                print(f"Trying {attempt_name}...")
                dataset = load_func()
                print(f"Success with {attempt_name}")

                # Limit examples if specified
                if num_examples > 0 and len(dataset) > num_examples:  # type: ignore
                    dataset = dataset.select(range(num_examples))  # type: ignore

                return dataset  # type: ignore
            except Exception as e2:
                print(f"Failed {attempt_name}: {e2}")
                continue

        raise Exception("All approaches to load LiveCodeBench dataset failed")


def decode_private_test_cases(encoded_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Decode private test cases using the official LiveCodeBench decoding method.

    The private test cases are encoded using:
    1. JSON -> 2. Pickle -> 3. Zlib compression -> 4. Base64 encoding

    Args:
        encoded_str: Base64-encoded and compressed string

    Returns:
        List of test cases or None if decoding fails
    """
    try:
        # Follow the exact decoding steps from LiveCodeBench
        # 1. Base64 decode
        decoded = base64.b64decode(encoded_str.encode("utf-8"))

        # 2. Zlib decompress
        decompressed = zlib.decompress(decoded)

        # 3. Pickle loads
        unpickled = pickle.loads(decompressed)

        # 4. JSON loads (unpickled is already a string)
        test_cases = json.loads(unpickled)

        return test_cases if isinstance(test_cases, list) else None

    except Exception:
        # If decoding fails, return None
        return None


def load_environment(
    dataset_name: str = "livecodebench/code_generation_lite",
    version_tag: str = "release_v5",
    num_examples: int = -1,  # -1 means use all examples
    pool_size: int = 20,  # Configurable container pool size
) -> vf.SingleTurnEnv:
    """Load LiveCodeBench environment with automatic cleanup

    Args:
        dataset_name: Name of the dataset to load
        version_tag: Version of the dataset to use
        num_examples: Number of examples to load (-1 for all)
        pool_size: Number of pre-allocated Docker containers

    Returns:
        SingleTurnEnv: Configured environment with automatic cleanup
    """
    # Load dataset
    dataset = load_livecodebench_dataset(dataset_name, version_tag, num_examples)

    # Initialize sandbox executor
    sandbox = SecureSandboxExecutor(pool_size=pool_size)

    # Convert dataset to verifiers format
    converted_dataset = []
    for idx, example in enumerate(dataset):
        # Extract problem information
        problem_id = example.get("question_id", f"problem_{idx}")  # type: ignore
        problem_desc = example.get("question_content", "")  # type: ignore
        starter_code = example.get("starter_code", "")  # type: ignore

        # Extract test cases - they're split between public and private
        test_cases = []

        # Add public test cases
        if "public_test_cases" in example and example["public_test_cases"]:  # type: ignore
            try:
                public_tests = json.loads(example["public_test_cases"])  # type: ignore
                if isinstance(public_tests, list):
                    test_cases.extend(public_tests)
            except json.JSONDecodeError:
                print(f"Failed to parse public test cases for {problem_id}")

        # Add private test cases
        if "private_test_cases" in example and example["private_test_cases"]:  # type: ignore
            private_raw = example["private_test_cases"]  # type: ignore

            # Check if it's already a list
            if isinstance(private_raw, list):
                test_cases.extend(private_raw)
            elif isinstance(private_raw, str):
                # Try direct JSON parsing first
                try:
                    private_tests = json.loads(private_raw)
                    if isinstance(private_tests, list):
                        test_cases.extend(private_tests)
                except json.JSONDecodeError:
                    # Try to decode using the official LiveCodeBench method
                    decoded_tests = decode_private_test_cases(private_raw)
                    if decoded_tests:
                        test_cases.extend(decoded_tests)

        # Format the prompt
        prompt = f"{problem_desc}\n\n"
        if starter_code:
            prompt += f"Starter code:\n```python\n{starter_code}\n```\n\n"
        if example.get("public_tests"):  # type: ignore
            prompt += f"Example test cases:\n{example['public_tests']}\n\n"  # type: ignore

        # Parse the test cases into the expected format
        parsed_tests = []
        for test in test_cases:
            if isinstance(test, dict) and "input" in test and "output" in test:
                parsed_tests.append(
                    {
                        "input": test["input"],
                        "output": test["output"],
                        "testtype": test.get("testtype", "public"),
                    }
                )

        converted_dataset.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": "",  # Will be filled by the model
                "info": {
                    "problem_id": problem_id,
                    "problem_desc": problem_desc,
                    "starter_code": starter_code,
                    "test_cases": parsed_tests,
                    "difficulty": example.get("difficulty", "unknown"),  # type: ignore
                    "contest_date": example.get("contest_date", ""),  # type: ignore
                },
            }
        )

    # Create dataset
    dataset = Dataset.from_list(converted_dataset)

    # Create parser and rubric
    def extract_code(completion: str) -> str:
        """Extract code from model completion"""
        if not completion:
            return ""

        # Try to extract code between ```python and ```
        code_match = re.search(r"```python\n(.*?)```", completion, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try to extract code between ``` and ```
        code_match = re.search(r"```\n(.*?)```", completion, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # If no code blocks, look for function definition
        if "def " in completion:
            # Extract from first def to the end
            def_index = completion.find("def ")
            return completion[def_index:]

        # Return the entire completion as last resort
        return completion

    parser = vf.Parser(extract_fn=extract_code)

    # Define rubric functions
    def correctness_score(completion, info, parser, state) -> float:
        """Evaluate code correctness using test cases - returns fraction passed"""
        code = parser.parse_answer(completion)

        if not code or not info.get("test_cases"):
            state["raw_score"] = 0.0
            return 0.0

        # Add starter code if provided
        if info.get("starter_code"):
            code = info["starter_code"] + "\n\n" + code

        # Execute test cases in parallel using pre-allocated containers
        test_cases = info["test_cases"]

        def run_test(test):
            container = sandbox.container_pool.get_container()
            try:
                input_data = test.get("input", "")
                stdout, stderr, exit_code = sandbox.execute_in_container(
                    container, code, input_data
                )
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "expected": test.get("output", "").strip(),
                }
            finally:
                sandbox.container_pool.return_container(container)

        # Run tests in parallel with thread pool
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all test cases
            futures = [executor.submit(run_test, test) for test in test_cases]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(
                        {"stdout": "", "stderr": str(e), "exit_code": 1, "expected": ""}
                    )

        # Check results
        passed = sum(
            1
            for r in results
            if r["exit_code"] == 0 and r["stdout"].strip() == r["expected"]
        )
        score = passed / len(test_cases) if test_cases else 0.0

        # Store raw score in state for pass_score to use
        state["raw_score"] = score
        state["total_tests"] = len(test_cases)
        state["passed_tests"] = passed

        return score

    def pass_score(completion, info, parser, state) -> float:
        """Returns 1.0 if ALL test cases pass, 0.0 otherwise"""
        # Use the raw score stored by correctness_score
        raw_score = state.get("raw_score", 0.0)
        return 1.0 if raw_score == 1.0 else 0.0

    # Create rubric with both metrics
    rubric = vf.Rubric(
        funcs=[correctness_score, pass_score],
        weights=[0.0, 1.0],  # Only pass_score contributes to final reward
        parser=parser,
        parallelize_scoring=False,  # Important: run sequentially so pass_score can use correctness_score's state
    )

    # Create environment
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric, parser=parser)

    # Attach sandbox to environment for cleanup
    env._sandbox = sandbox  # type: ignore

    # Override the original del method if it exists
    original_del = getattr(env, "__del__", None)

    def env_del(self):
        # Clean up sandbox
        if hasattr(self, "_sandbox"):
            self._sandbox.cleanup()
        # Call original del if it existed
        if original_del:
            original_del()

    env.__del__ = env_del.__get__(env, type(env))  # type: ignore

    # Register additional cleanup for safety
    atexit.register(lambda: sandbox.cleanup() if sandbox else None)

    return env


class ContainerPool:
    """Pool of pre-allocated Docker containers for fast execution"""

    def __init__(
        self, docker_client, pool_size: int = 20, image: str = "python:3.11-slim"
    ):
        self.docker_client = docker_client
        self.image = image
        self.pool_size = pool_size
        self.available = Queue()
        self.all_containers = []
        self._shutdown = False
        self._lock = Lock()

        # Start background thread to maintain pool
        self._maintainer = Thread(target=self._maintain_pool, daemon=True)
        self._maintainer.start()

        # Pre-allocate initial containers
        print(f"Pre-allocating {pool_size} containers...")
        for _ in range(pool_size):
            self._create_container()

    def _create_container(self):
        """Create a new container and add it to the pool"""
        try:
            # Create container in paused state
            container = self.docker_client.containers.create(
                image=self.image,
                command=[
                    "python",
                    "-c",
                    "import time; time.sleep(3600)",
                ],  # Keep alive for 1 hour
                detach=True,
                mem_limit="512m",
                memswap_limit="512m",
                cpu_quota=50000,  # 0.5 CPU
                cpu_period=100000,
                pids_limit=50,
                network_disabled=True,
                tmpfs={"/tmp": "size=64M,mode=1777"},
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                user="1000:1000",
                stdin_open=True,
                tty=False,
                labels={"livecodebench": "true"},  # Add label for cleanup
            )
            container.start()

            with self._lock:
                self.all_containers.append(container)

            self.available.put(container)

        except Exception as e:
            print(f"Failed to create container: {e}")

    def _maintain_pool(self):
        """Background thread to maintain pool size"""
        while not self._shutdown:
            try:
                # Check pool size every second
                time.sleep(1)

                # Count healthy containers
                current_size = self.available.qsize()

                # Replenish if needed
                if current_size < self.pool_size // 2:  # Refill when below 50%
                    for _ in range(self.pool_size - current_size):
                        self._create_container()

            except Exception as e:
                print(f"Pool maintenance error: {e}")

    def get_container(self, timeout: int = 5):
        """Get a container from the pool"""
        try:
            return self.available.get(timeout=timeout)
        except Empty:
            # If pool is empty, create one on demand
            print("Container pool empty, creating on demand...")
            self._create_container()
            return self.available.get(timeout=timeout)

    def return_container(self, container):
        """Return a container to the pool or destroy if unhealthy"""
        try:
            # Check if container is still healthy
            container.reload()
            if container.status == "running":
                # Reset container state (optional - could implement cleanup here)
                self.available.put(container)
            else:
                # Remove unhealthy container
                container.remove(force=True)
        except Exception:
            # Container is dead, just try to remove it
            try:
                container.remove(force=True)
            except Exception:
                pass

    def shutdown(self):
        """Clean up all containers"""
        self._shutdown = True

        # Wait for maintainer thread to finish
        if hasattr(self, "_maintainer") and self._maintainer.is_alive():
            self._maintainer.join(timeout=5)

        # Clear the available queue
        while not self.available.empty():
            try:
                self.available.get_nowait()
            except Exception:
                pass

        # Remove all containers
        with self._lock:
            print(f"Shutting down {len(self.all_containers)} containers...")
            for container in self.all_containers:
                try:
                    container.stop(timeout=2)
                    container.remove(force=True)
                except Exception:
                    pass
            self.all_containers.clear()


def cleanup_all_docker_resources():
    """Utility function to clean up ALL Docker resources related to LiveCodeBench

    This is useful for manual cleanup if resources were left behind.
    """
    try:
        client = docker.from_env()

        # Remove all containers with our label
        containers = client.containers.list(
            all=True, filters={"label": "livecodebench=true"}
        )
        print(f"Found {len(containers)} LiveCodeBench containers to clean up...")

        for container in containers:
            try:
                container.stop(timeout=2)
                container.remove(force=True)
                print(f"Removed container {container.short_id}")
            except Exception as e:
                print(f"Failed to remove container {container.short_id}: {e}")

        # Prune system resources
        print("Pruning Docker system resources...")
        client.containers.prune()

        # Optional aggressive cleanup (uncomment if needed)
        # client.images.prune()
        # client.volumes.prune()
        # client.networks.prune()

        print("Docker cleanup complete!")

    except Exception as e:
        print(f"Error during Docker cleanup: {e}")
