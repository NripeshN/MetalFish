#!/usr/bin/env python3
"""List or delete stale GCP CUDA gate instances."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence


DEFAULT_NAME_REGEX = r"^metalfish-(cuda|win-cuda)"


@dataclass(frozen=True)
class GcpInstance:
    name: str
    zone: str
    status: str
    created: datetime | None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def zone_name(zone: str | None) -> str:
    if not zone:
        return ""
    return zone.rsplit("/", 1)[-1]


def parse_instances(payload: str) -> list[GcpInstance]:
    records = json.loads(payload or "[]")
    instances: list[GcpInstance] = []
    for record in records:
        instances.append(
            GcpInstance(
                name=str(record.get("name", "")),
                zone=zone_name(record.get("zone")),
                status=str(record.get("status", "")),
                created=parse_timestamp(record.get("creationTimestamp")),
            )
        )
    return instances


def filter_instances(
    instances: Sequence[GcpInstance],
    *,
    name_regex: str,
    older_than_hours: float,
    now: datetime,
) -> list[GcpInstance]:
    pattern = re.compile(name_regex)
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=older_than_hours)
    selected: list[GcpInstance] = []
    for instance in instances:
        if not pattern.search(instance.name):
            continue
        if older_than_hours > 0 and instance.created and instance.created > cutoff:
            continue
        selected.append(instance)
    return selected


def run_gcloud(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gcloud", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def list_instances(project: str) -> list[GcpInstance]:
    proc = run_gcloud(
        [
            "compute",
            "instances",
            "list",
            "--project",
            project,
            "--format=json(name,zone,status,creationTimestamp)",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "gcloud compute instances list failed")
    return parse_instances(proc.stdout)


def delete_instance(project: str, instance: GcpInstance) -> None:
    proc = run_gcloud(
        [
            "compute",
            "instances",
            "delete",
            instance.name,
            "--project",
            project,
            "--zone",
            instance.zone,
            "--quiet",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip()
            or f"gcloud compute instances delete failed for {instance.name}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="metalfish")
    parser.add_argument("--name-regex", default=DEFAULT_NAME_REGEX)
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=0.0,
        help="Only match instances older than this many hours; 0 disables age filtering.",
    )
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args(argv)


def instance_to_json(instance: GcpInstance) -> dict[str, str | None]:
    return {
        "name": instance.name,
        "zone": instance.zone,
        "status": instance.status,
        "created": instance.created.isoformat() if instance.created else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(timezone.utc)
    matches = filter_instances(
        list_instances(args.project),
        name_regex=args.name_regex,
        older_than_hours=args.older_than_hours,
        now=now,
    )

    if args.json:
        print(json.dumps([instance_to_json(instance) for instance in matches], indent=2))
    elif not matches:
        print("No matching CUDA GCP instances.")
    else:
        for instance in matches:
            created = instance.created.isoformat() if instance.created else "unknown"
            print(f"{instance.name}\t{instance.zone}\t{instance.status}\t{created}")

    if args.delete:
        for instance in matches:
            delete_instance(args.project, instance)
        if matches:
            print(f"Deleted {len(matches)} matching CUDA GCP instance(s).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
