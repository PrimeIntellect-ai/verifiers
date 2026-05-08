import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-search-env")

RECORDS = {
    "kiln_battery_loop": {
        "title": "Kiln Battery Loop",
        "keywords": ["ceramic", "battery", "recycling", "kiln"],
        "summary": "A ceramic separator recovery process for spent solid-state batteries.",
    },
    "tide_scout": {
        "title": "Tide Scout",
        "keywords": ["ocean", "drone", "algae", "bloom"],
        "summary": "A coastal drone survey that maps algae bloom movement in real time.",
    },
    "stacks_navigator": {
        "title": "Stacks Navigator",
        "keywords": ["library", "robot", "sorting", "stacks"],
        "summary": "A shelf-scanning robot that sorts returned books by aisle and section.",
    },
    "moss_blanket": {
        "title": "Moss Blanket",
        "keywords": ["green", "roof", "insulation", "moss"],
        "summary": "A moss-based roof layer that improves summer insulation for warehouses.",
    },
    "ember_atlas": {
        "title": "Ember Atlas",
        "keywords": ["satellite", "wildfire", "mapping", "ember"],
        "summary": "A satellite mapping workflow for spotting wildfire ember corridors.",
    },
    "yeast_whisper": {
        "title": "Yeast Whisper",
        "keywords": ["fermentation", "sensor", "brewery", "yeast"],
        "summary": "A brewery sensor package that predicts fermentation stalls early.",
    },
    "tunnel_pulse": {
        "title": "Tunnel Pulse",
        "keywords": ["rail", "tunnel", "airflow", "ventilation"],
        "summary": "An airflow model for balancing ventilation across long rail tunnels.",
    },
    "gallery_grid": {
        "title": "Gallery Grid",
        "keywords": ["museum", "climate", "microgrid", "gallery"],
        "summary": "A microgrid controller that stabilizes gallery climate systems.",
    },
    "frost_lantern": {
        "title": "Frost Lantern",
        "keywords": ["orchard", "frost", "prediction", "lantern"],
        "summary": "A frost prediction network for timing orchard heaters and fans.",
    },
    "curb_queue": {
        "title": "Curb Queue",
        "keywords": ["city", "curb", "delivery", "routing"],
        "summary": "A curbside routing system that schedules short delivery stops.",
    },
}


@mcp.tool()
def search_records(query: str) -> str:
    normalized_tokens = set(query.lower().split())
    matches = []
    for record_id, record in RECORDS.items():
        keywords = set(record["keywords"])
        title_tokens = set(str(record["title"]).lower().split())
        if normalized_tokens & (keywords | title_tokens):
            matches.append({"record_id": record_id, "title": record["title"]})
    return json.dumps(matches)


@mcp.tool()
def read_record(record_id: str) -> str:
    if record_id not in RECORDS:
        raise ValueError(f"Unknown record_id: {record_id}")
    record = RECORDS[record_id]
    return f"{record['title']}\n\n{record['summary']}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
