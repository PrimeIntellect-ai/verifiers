from __future__ import annotations

import asyncio
import functools
import random
import string
from collections.abc import Callable, Mapping

from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.utils.endpoint_utils import openai_endpoint_config


class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
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


@vf.reward(weight=1.0)
async def expected_database_change(task, state) -> float:
    expected = task["expected"]
    if expected["kind"] == "book":
        itineraries = state.get("itinerary_database", {})
        return float(
            len(itineraries) == 1
            and any(
                item["user_profile"]["name"] == expected["user"]
                and item["flight"]["flight_id"] == expected["flight_id"]
                for item in itineraries.values()
            )
        )
    if expected["kind"] == "cancel":
        return float(
            expected["confirmation_number"] not in state.get("itinerary_database", {})
        )
    if expected["kind"] == "ticket":
        tickets = state.get("ticket_database", {})
        return float(
            len(tickets) == 1
            and any(
                item["user_profile"]["name"] == expected["user"]
                and expected["contains"].lower() in item["user_request"].lower()
                for item in tickets.values()
            )
        )
    raise ValueError(f"Unknown expected kind: {expected['kind']}")


@vf.metric
async def dspy_calls(task, state) -> float:
    return float(len(state.get("trajectory", [])))


def source():
    def row(
        example_id: int,
        user_request: str,
        expected: dict[str, object],
        initial_itineraries: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, object]:
        task = {
            "example_id": example_id,
            "user_request": user_request,
            "prompt": [{"role": "user", "content": user_request}],
            "expected": expected,
        }
        if initial_itineraries is not None:
            task["initial_itineraries"] = initial_itineraries
        return task

    return [
        row(
            0,
            (
                "please help me book a flight from SFO to JFK on 09/01/2025, "
                "my name is Adam"
            ),
            {"kind": "book", "user": "Adam", "flight_id": "DA123"},
        ),
        row(
            1,
            (
                "please help me book a flight from SFO to SNA on 10/01/2025, "
                "my name is Bob"
            ),
            {"kind": "book", "user": "Bob", "flight_id": "DA456"},
        ),
        row(
            2,
            (
                "please cancel itinerary CH123 for Chelsie; she no longer wants "
                "to travel"
            ),
            {"kind": "cancel", "confirmation_number": "CH123"},
            {"CH123": itinerary("CH123", "Chelsie", "DA125").model_dump()},
        ),
        row(
            3,
            (
                "my name is David and I need wheelchair assistance added to my "
                "reservation"
            ),
            {
                "kind": "ticket",
                "user": "David",
                "contains": "wheelchair assistance",
            },
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


def build_airline_tools(task) -> tuple[list[object], dict[str, dict[str, BaseModel]]]:
    users = user_database()
    flights = flight_database()
    itineraries = {
        key: Itinerary.model_validate(value)
        for key, value in (task.get("initial_itineraries") or {}).items()
    }
    tickets: dict[str, Ticket] = {}

    def fetch_flight_info(date: Date, origin: str, destination: str):
        """Fetch flight information from origin to destination on the given date"""
        date = Date.model_validate(date)
        matching_flights = []

        for flight in flights.values():
            if (
                flight.date_time.year == date.year
                and flight.date_time.month == date.month
                and flight.date_time.day == date.day
                and flight.origin == origin
                and flight.destination == destination
            ):
                matching_flights.append(flight)
        if len(matching_flights) == 0:
            raise ValueError("No matching flight found!")
        return matching_flights

    def fetch_itinerary(confirmation_number: str):
        """Fetch a booked itinerary information from database"""
        return itineraries.get(confirmation_number)

    def pick_flight(flights: list[Flight]):
        """Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties."""
        sorted_flights = sorted(
            flights,
            key=lambda x: (
                x.get("duration") if isinstance(x, dict) else x.duration,
                x.get("price") if isinstance(x, dict) else x.price,
            ),
        )
        return sorted_flights[0]

    def _generate_id(length=8):
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choices(chars, k=length))

    def book_flight(flight: Flight, user_profile: UserProfile):
        """Book a flight on behalf of the user."""
        flight = Flight.model_validate(flight)
        user_profile = UserProfile.model_validate(user_profile)
        confirmation_number = _generate_id()
        while confirmation_number in itineraries:
            confirmation_number = _generate_id()
        itineraries[confirmation_number] = Itinerary(
            confirmation_number=confirmation_number,
            user_profile=user_profile,
            flight=flight,
        )
        return confirmation_number, itineraries[confirmation_number]

    def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
        """Cancel an itinerary on behalf of the user."""
        _ = UserProfile.model_validate(user_profile)
        if confirmation_number in itineraries:
            del itineraries[confirmation_number]
            return
        raise ValueError(
            "Cannot find the itinerary, please check your confirmation number."
        )

    def get_user_info(name: str):
        """Fetch the user profile from database with given name."""
        return users.get(name)

    def file_ticket(user_request: str, user_profile: UserProfile):
        """File a customer support ticket if this is something the agent cannot handle."""
        user_profile = UserProfile.model_validate(user_profile)
        ticket_id = _generate_id(length=6)
        tickets[ticket_id] = Ticket(
            user_request=user_request,
            user_profile=user_profile,
        )
        return ticket_id

    return [
        async_tool(fetch_flight_info),
        async_tool(fetch_itinerary),
        async_tool(pick_flight),
        async_tool(book_flight),
        async_tool(cancel_itinerary),
        async_tool(get_user_info),
        async_tool(file_ticket),
    ], {
        "itinerary_database": itineraries,
        "ticket_database": tickets,
    }


def async_tool(fn: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(fn)
    async def wrapped(*args: object, **kwargs: object) -> object:
        return await asyncio.to_thread(fn, *args, **kwargs)

    return wrapped


def dump_database(database: Mapping[str, BaseModel]) -> dict[str, dict[str, object]]:
    return {key: value.model_dump() for key, value in database.items()}


async def run_dspy_flight_program(task, state, client):
    import dspy

    class DSPyAirlineCustomerService(dspy.Signature):
        """You are an airline customer service agent that helps user book and manage flights.

        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.
        """

        user_request: str = dspy.InputField()
        process_result: str = dspy.OutputField(
            desc=(
                "Message that summarizes the process result, and the information users need, e.g., the "
                "confirmation_number if a new flight is booked."
            )
        )

    tools, databases = build_airline_tools(task)
    endpoint_config = openai_endpoint_config(state, client)
    lm = dspy.LM(
        f"openai/{endpoint_config['model']}",
        api_base=endpoint_config["api_base"],
        api_key=endpoint_config["api_key"],
        cache=False,
    )
    agent = dspy.ReAct(DSPyAirlineCustomerService, tools=tools, max_iters=8)
    with dspy.context(lm=lm):
        result = await agent.acall(user_request=task["user_request"])

    state["process_result"] = str(result.process_result)
    state["reasoning"] = str(getattr(result, "reasoning", ""))
    state["dspy_trajectory"] = stringify_nested(getattr(result, "trajectory", {}))
    state["itinerary_database"] = dump_database(databases["itinerary_database"])
    state["ticket_database"] = dump_database(databases["ticket_database"])
    state["completion"] = [
        {"role": "assistant", "content": state["process_result"]},
    ]
    return state


def stringify_nested(value: object) -> object:
    if isinstance(value, BaseModel):
        return stringify_nested(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): stringify_nested(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [stringify_nested(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[expected_database_change],
        metrics=[dspy_calls],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program={"fn": "dspy_flights:run_dspy_flight_program"},
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
