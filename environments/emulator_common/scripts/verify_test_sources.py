import argparse
import json
import sys
from pathlib import Path

ENVIRONMENTS_ROOT = Path(__file__).resolve().parents[2]
if str(ENVIRONMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENTS_ROOT))

from emulator_common.test_sources import (  # noqa: E402
    load_public_test_manifest,
    verify_source_reachable,
)


def iter_manifests(root: Path):
    yield from sorted(root.glob("emulator_*/tasks/public_tests.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ENVIRONMENTS_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = []
    ok = True
    for path in iter_manifests(args.root):
        manifest = load_public_test_manifest(path)
        for source in manifest.sources:
            reachable, detail = verify_source_reachable(source)
            results.append(
                {
                    "environment_id": manifest.environment_id,
                    "source": source.name,
                    "url": source.url,
                    "kind": source.kind,
                    "ok": reachable,
                    "detail": detail,
                }
            )
            ok = ok and reachable

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for row in results:
            status = "OK" if row["ok"] else "FAIL"
            print(
                f"{status} {row['environment_id']} {row['source']} {row['url']} {row['detail']}"
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
