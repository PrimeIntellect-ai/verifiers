import asyncio
import functools
import random
import re
import string
from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, Field

import verifiers.v1 as vf

DSPyToolResult: TypeAlias = (
    str
    | None
    | BaseModel
    | list[BaseModel]
    | tuple[str, BaseModel]
    | tuple[str, dict[str, BaseModel]]
)


class DSPyFlightsHarnessConfig(vf.HarnessConfig):
    max_iters: int = 8


class Date(BaseModel):
    year: int
    month: int
    day: int
    hour: int


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float


class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight


class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile


class FlightRunResult(vf.Config):
    process_result: str
    reasoning: str = ""
    dspy_calls: int = 0
    dspy_trajectory: vf.JsonValue | None = None
    itinerary_database: dict[str, vf.JsonData]
    ticket_database: dict[str, vf.JsonData]


class ExpectedFlightChange(BaseModel, extra="forbid"):
    kind: Literal["book", "cancel", "ticket"]
    user: str | None = None
    flight_id: str | None = None
    confirmation_number: str | None = None
    contains: str | None = None


class DSPyFlightsTask(vf.Task):
    user_request: str
    expected: ExpectedFlightChange
    initial_itineraries: dict[str, Itinerary] = Field(default_factory=dict)


def user_database() -> dict[str, UserProfile]:
    return {
        "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
        "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
        "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
        "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
    }


def flight_database() -> dict[str, Flight]:
    return {
        "DA123": Flight(
            flight_id="DA123",
            origin="SFO",
            destination="JFK",
            date_time=Date(year=2025, month=9, day=1, hour=1),
            duration=3,
            price=200,
        ),
        "DA125": Flight(
            flight_id="DA125",
            origin="SFO",
            destination="JFK",
            date_time=Date(year=2025, month=9, day=1, hour=7),
            duration=9,
            price=500,
        ),
        "DA456": Flight(
            flight_id="DA456",
            origin="SFO",
            destination="SNA",
            date_time=Date(year=2025, month=10, day=1, hour=1),
            duration=2,
            price=100,
        ),
        "DA460": Flight(
            flight_id="DA460",
            origin="SFO",
            destination="SNA",
            date_time=Date(year=2025, month=10, day=1, hour=9),
            duration=2,
            price=120,
        ),
    }


def load_tasks(split: vf.TaskSplit = "train") -> list[vf.JsonData]:
    _ = split

    def record(
        example_id: int,
        user_request: str,
        expected: vf.JsonData,
        initial_itineraries: dict[str, vf.JsonData] | None = None,
    ) -> vf.JsonData:
        task: vf.JsonData = {
            "example_id": example_id,
            "user_request": user_request,
            "prompt": [{"role": "user", "content": user_request}],
            "expected": expected,
        }
        if initial_itineraries is not None:
            task["initial_itineraries"] = initial_itineraries
        return task

    return [
        record(
            0,
            (
                "please help me book a flight from SFO to JFK on 09/01/2025, "
                "my name is Adam"
            ),
            {"kind": "book", "user": "Adam", "flight_id": "DA123"},
        ),
        record(
            1,
            (
                "please help me book a flight from SFO to SNA on 10/01/2025, "
                "my name is Bob"
            ),
            {"kind": "book", "user": "Bob", "flight_id": "DA456"},
        ),
        record(
            2,
            (
                "please cancel itinerary CH123 for Chelsie; she no longer wants "
                "to travel"
            ),
            {"kind": "cancel", "confirmation_number": "CH123"},
            {"CH123": itinerary("CH123", "Chelsie", "DA125").model_dump()},
        ),
        record(
            3,
            (
                "my name is David and I need wheelchair assistance added to my "
                "reservation"
            ),
            {"kind": "ticket", "user": "David", "contains": "wheelchair assistance"},
        ),
        record(
            4,
            "my name is Adam and I need a vegetarian meal noted for my upcoming trip",
            {"kind": "ticket", "user": "Adam", "contains": "vegetarian meal"},
        ),
        record(
            5,
            "please cancel itinerary BO456 for Bob because his plans changed",
            {"kind": "cancel", "confirmation_number": "BO456"},
            {"BO456": itinerary("BO456", "Bob", "DA456").model_dump()},
        ),
        record(
            6,
            (
                "please help me book a flight from SFO to JFK on 09/01/2025, "
                "my name is Chelsie"
            ),
            {"kind": "book", "user": "Chelsie", "flight_id": "DA123"},
        ),
        record(
            7,
            (
                "please help me book a flight from SFO to SNA on 10/01/2025, "
                "my name is David"
            ),
            {"kind": "book", "user": "David", "flight_id": "DA456"},
        ),
        record(
            8,
            "cancel confirmation AD460 for Adam; he will rebook later",
            {"kind": "cancel", "confirmation_number": "AD460"},
            {"AD460": itinerary("AD460", "Adam", "DA460").model_dump()},
        ),
        record(
            9,
            "my name is Chelsie and I need to travel with a service animal",
            {"kind": "ticket", "user": "Chelsie", "contains": "service animal"},
        ),
    ]


def itinerary(confirmation_number: str, user_name: str, flight_id: str) -> Itinerary:
    users = user_database()
    flights = flight_database()
    return Itinerary(
        confirmation_number=confirmation_number,
        user_profile=users[user_name],
        flight=flights[flight_id],
    )


def build_airline_tools(
    task: DSPyFlightsTask,
) -> tuple[list[Callable[..., DSPyToolResult]], dict[str, dict[str, BaseModel]]]:
    users = user_database()
    flights = flight_database()
    itineraries = dict(task.initial_itineraries)
    tickets: dict[str, Ticket] = {}

    def fetch_flight_info(date: Date, origin: str, destination: str) -> list[Flight]:
        date = Date.model_validate(date)
        matching_flights = [
            flight
            for flight in flights.values()
            if flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ]
        if not matching_flights:
            raise ValueError("No matching flight found.")
        return matching_flights

    def fetch_itinerary(confirmation_number: str) -> Itinerary | None:
        return itineraries.get(confirmation_number)

    def pick_flight(flights: list[Flight]) -> Flight:
        return sorted(flights, key=lambda flight: (flight.duration, flight.price))[0]

    def generate_id(length: int = 8) -> str:
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choices(chars, k=length))

    def book_flight(flight: Flight, user_profile: UserProfile) -> tuple[str, Itinerary]:
        flight = Flight.model_validate(flight)
        user_profile = UserProfile.model_validate(user_profile)
        confirmation_number = generate_id()
        while confirmation_number in itineraries:
            confirmation_number = generate_id()
        itineraries[confirmation_number] = Itinerary(
            confirmation_number=confirmation_number,
            user_profile=user_profile,
            flight=flight,
        )
        return confirmation_number, itineraries[confirmation_number]

    def cancel_itinerary(confirmation_number: str, user_profile: UserProfile) -> None:
        UserProfile.model_validate(user_profile)
        if confirmation_number in itineraries:
            del itineraries[confirmation_number]
            return
        raise ValueError(
            "Cannot find the itinerary, please check your confirmation number."
        )

    def get_user_info(name: str) -> UserProfile | None:
        return users.get(name)

    def file_ticket(user_request: str, user_profile: UserProfile) -> str:
        user_profile = UserProfile.model_validate(user_profile)
        ticket_id = generate_id(length=6)
        tickets[ticket_id] = Ticket(
            user_request=user_request,
            user_profile=user_profile,
        )
        return ticket_id

    tools: list[Callable[..., DSPyToolResult]] = [
        async_tool(fetch_flight_info),
        async_tool(fetch_itinerary),
        async_tool(pick_flight),
        async_tool(book_flight),
        async_tool(cancel_itinerary),
        async_tool(get_user_info),
        async_tool(file_ticket),
    ]
    return tools, {"itinerary_database": itineraries, "ticket_database": tickets}


def async_tool(
    fn: Callable[..., DSPyToolResult],
) -> Callable[..., DSPyToolResult]:
    @functools.wraps(fn)
    async def wrapped(*args: object, **kwargs: object) -> DSPyToolResult:
        return await asyncio.to_thread(fn, *args, **kwargs)

    return wrapped


def dump_database(database: dict[str, BaseModel]) -> dict[str, vf.JsonData]:
    return {
        key: cast(vf.JsonData, value.model_dump(mode="json"))
        for key, value in database.items()
    }


def jsonable_dspy(value: object) -> vf.JsonValue:
    if isinstance(value, BaseModel):
        return jsonable_dspy(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): jsonable_dspy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable_dspy(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def dspy_iteration_count(result: object) -> int:
    trajectory = getattr(result, "trajectory", None)
    if isinstance(trajectory, Mapping):
        indices = {
            match.group(1)
            for key in trajectory
            if (match := re.search(r"_(\d+)$", str(key)))
        }
        if indices:
            return len(indices)
        return len(trajectory)
    if isinstance(trajectory, list | tuple):
        return len(trajectory)
    return 0


async def run_dspy_flight_agent(
    *,
    task: DSPyFlightsTask,
    base_url: str,
    api_key: str,
    model: str,
    max_iters: int,
) -> FlightRunResult:
    import dspy

    class DSPyAirlineCustomerService(dspy.Signature):
        """Airline customer service agent for booking and managing flights."""

        user_request: str = dspy.InputField()
        process_result: str = dspy.OutputField(
            desc=(
                "Message that summarizes the process result and any information "
                "the user needs, such as a confirmation number."
            )
        )

    tools, databases = build_airline_tools(task)
    lm = dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key=api_key,
        cache=False,
    )
    agent = dspy.ReAct(DSPyAirlineCustomerService, tools=tools, max_iters=max_iters)
    with dspy.context(lm=lm):
        result = await agent.acall(user_request=task.user_request)

    return FlightRunResult(
        process_result=str(result.process_result),
        reasoning=str(getattr(result, "reasoning", "")),
        dspy_calls=dspy_iteration_count(result),
        dspy_trajectory=jsonable_dspy(getattr(result, "trajectory", None)),
        itinerary_database=dump_database(databases["itinerary_database"]),
        ticket_database=dump_database(databases["ticket_database"]),
    )


class DSPyFlightsTasksetConfig(vf.TasksetConfig):
    id: str = "dspy-flights"


class DSPyFlightsTaskset(vf.Taskset[DSPyFlightsTasksetConfig]):
    task_type = DSPyFlightsTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(split)

    @vf.reward
    async def expected_database_change(
        self, task: DSPyFlightsTask, state: vf.State
    ) -> float:
        expected = task.expected
        itineraries = state.artifacts.get("itinerary_database")
        tickets = state.artifacts.get("ticket_database")
        itinerary_map = itineraries if isinstance(itineraries, Mapping) else {}
        ticket_map = tickets if isinstance(tickets, Mapping) else {}

        if expected.kind == "book":
            return float(
                len(itinerary_map) == 1
                and any(
                    isinstance(item, Mapping)
                    and isinstance(item.get("user_profile"), Mapping)
                    and isinstance(item.get("flight"), Mapping)
                    and item["user_profile"].get("name") == expected.user
                    and item["flight"].get("flight_id") == expected.flight_id
                    for item in itinerary_map.values()
                )
            )
        if expected.kind == "cancel":
            return float(expected.confirmation_number not in itinerary_map)
        if expected.kind == "ticket":
            contains = str(expected.contains or "").lower()
            return float(
                len(ticket_map) == 1
                and any(
                    isinstance(item, Mapping)
                    and isinstance(item.get("user_profile"), Mapping)
                    and item["user_profile"].get("name") == expected.user
                    and contains in str(item.get("user_request", "")).lower()
                    for item in ticket_map.values()
                )
            )
        raise ValueError(f"Unknown expected kind: {expected.kind!r}.")

    @vf.metric
    async def dspy_calls(self, state: vf.State) -> float:
        value = state.artifacts.get("dspy_calls")
        return float(value) if isinstance(value, int | float) else 0.0


class DSPyFlightsHarness(vf.Harness[DSPyFlightsHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = DSPyFlightsTask.model_validate(context.task.model_dump())
        state = context.state
        runtime = context.runtime
        if runtime is None:
            raise RuntimeError("DSPyFlightsHarness requires a runtime.")
        prompt = self.initial_messages(task)

        async def stop_check() -> str | None:
            if await self.is_completed(context):
                return state.stop_condition or "stop"
            return None

        async with vf.InterceptionServer(
            context,
            task,
            state,
            protocols=self.protocols,
            stop_check=stop_check,
        ) as endpoint:
            endpoint_url = await runtime.expose(endpoint.port)
            endpoint_env = endpoint.env(base_url=endpoint_url, model=context.model)
            result = await run_dspy_flight_agent(
                task=task,
                base_url=endpoint_env["OPENAI_BASE_URL"],
                api_key=endpoint_env["OPENAI_API_KEY"],
                model=endpoint_env["OPENAI_MODEL"],
                max_iters=self.config.max_iters,
            )

        state.artifacts.update(result.model_dump(mode="json", exclude_none=True))
        message = vf.AssistantMessage(content=result.process_result)
        state.add_turn(vf.Turn(prompt=prompt, completion=[message]))
        state.stop("dspy_completed")


def load_taskset(config: DSPyFlightsTasksetConfig) -> DSPyFlightsTaskset:
    return DSPyFlightsTaskset(config=config)


def load_harness(config: DSPyFlightsHarnessConfig) -> DSPyFlightsHarness:
    return DSPyFlightsHarness(config=config)
