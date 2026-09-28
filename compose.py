#!/usr/bin/env python3
"""
Renders a fully-resolved docker-compose file straight from a JSON config and
runs `docker compose up` against it.

Design notes (why it looks like this):
- No intermediate .env file: every value is baked directly into the compose
  file's `environment:` blocks as a literal, not as a `${VAR}` reference.
- The JSON config is split so that values which are IDENTICAL across the
  three app containers (backend / inference / ocr_inference) and are NOT
  credentials (hosts, ports, bucket names, logging, queue names) live once,
  at the top level, alongside the container keys. Anything that differs
  per container, or is a credential (usernames, passwords, access/secret
  keys) - even if the value happens to be the same everywhere - stays
  nested inside that container's own section and is never hoisted.
- Queue names are defined once under "queues" and used by both the backend
  (which only knows about them as publish/consume targets) and the
  matching worker (which reads/writes the same physical queue) - this is
  what previously caused a real bug: the backend and the workers used to
  have independently-typed-out queue names that could silently drift apart.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "project_name": "vino-service",
    "compose": {"detach": True, "build": True},
    # --- infrastructure services -------------------------------------------------
    "postgres": {
        "image_tag": "16-alpine",
        "port": 5432,
        "db": "mydb",
        "user": "postgres",
        "password": "postgres",
    },
    "minio": {
        "image_tag": "latest",
        "api_port": 9000,
        "console_port": 9001,
        "license_path": "./licenses",
        "root_user": "minioadmin",
        "root_password": "minioadmin",
        # shared, non-credential connection settings used by every app container
        "host": "minio",
        "port": 9000,
        "image_bucket": "images",
        "secure": False,
    },
    "rabbitmq": {
        "image_tag": "3.13-management",
        "port": 5672,
        "management_port": 15672,
        "vhost": "/",
        # shared, non-credential connection setting used by every app container
        "host": "rabbitmq",
    },
    "qdrant": {
        "image_tag": "latest",
        "port": 6333,
        "grpc_port": 6334,
        "prefer_grpc": False,
        "collection_name": "wines",
        "host": "qdrant",
    },
    # shared between the backend (publisher/consumer side) and the matching
    # worker (consumer/publisher side) - defined once so they can't drift apart
    "queues": {
        "cv": {
            "tasks": "wine.recognition.requests",
            "results": "wine.recognition.results",
        },
        "ocr": {
            "tasks": "ocr.recognition.tasks",
            "results": "ocr.recognition.results",
        },
    },
    # shared across backend + both workers (all three define the same LoggingConfig shape)
    "logging": {
        "level": "INFO",
        "logs_directory": "./logs/",
        "file": "",
        "serialize": False,
    },
    # --- app containers ------------------------------------------------------------
    "backend": {
        "app_name": "vino-service",
        "app_version": "0.1.0",
        "port": 8080,
        "internal_port": 8080,
        "debug": False,
        "title": "vino-back",
        "path_to_catalog": "./catalog.jsonl",
        # credentials: not hoisted even though they equal the infra services' own creds
        "database": {"user": "postgres", "password": "postgres"},
        "minio": {"access_key": "minioadmin", "secret_key": "minioadmin"},
        "rabbitmq": {"user": "guest", "password": "guest"},
        "parser": {
            "base_url": "https://vino-svoe.ru",
            "timeout": 10,
            "max_retries": 3,
            "retry_delay": 1,
        },
    },
    "inference": {
        "build_context": "./inference",
        "dockerfile": ".dockerfile",
        "worker_name": "cv_inference_worker",
        "worker_version": "0.0.1",
        "model_weights_path": "/app/models/ArcFace_v2.pth",
        "yolo_weights_path": "/app/models/YOLO_best_100.pt",
        "embedding_size": 512,
        "image_size": 224,
        "device": "cuda:0",
        "dataset_dir": "/app/dataset/train",
        "use_local_path": False,
        "prefetch_count": 1,
        # credentials: not hoisted
        "minio": {"access_key": "minioadmin", "secret_key": "minioadmin"},
        "rabbitmq": {"username": "guest", "password": "guest"},
    },
    "ocr_inference": {
        "build_context": "./ocr_inference",
        "dockerfile": ".dockerfile",
        "worker_name": "ocr_inference_worker",
        "worker_version": "0.0.1",
        "catalog_path": "/app/dataset/wine_catalog.jsonl",
        "checkpoint_path": "/app/weights/parseq_wine_best.pt",
        "device": "cuda:0",
        "craft_canvas_size": 2560,
        "recognition_passes": 3,
        "allow_multilingual": True,
        "top_k": 5,
        "prefetch_count": 1,
        # credentials: not hoisted
        "minio": {"access_key": "minioadmin", "secret_key": "minioadmin"},
        "rabbitmq": {"username": "guest", "password": "guest"},
    },
}


def deep_merge(defaults: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
            continue
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON config root must be an object.")

    return data


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _envs(mapping: dict[str, Any]) -> dict[str, str]:
    return {key: _env_value(value) for key, value in mapping.items()}


def render_compose(cfg: dict[str, Any]) -> dict[str, Any]:
    project = cfg["project_name"]
    postgres = cfg["postgres"]
    minio = cfg["minio"]
    rabbitmq = cfg["rabbitmq"]
    qdrant = cfg["qdrant"]
    queues = cfg["queues"]
    log_cfg = cfg["logging"]
    backend = cfg["backend"]
    inference = cfg["inference"]
    ocr = cfg["ocr_inference"]

    def cname(suffix: str) -> str:
        return f"{project}-{suffix}"

    healthcheck = lambda test, interval="10s", timeout="5s", retries=5: {
        "test": test,
        "interval": interval,
        "timeout": timeout,
        "retries": retries,
    }

    def gpu_reservation() -> dict[str, Any]:
        # a fresh dict per call - reusing one object across services would make
        # PyYAML emit anchors/aliases for it, which is needless noise here
        return {
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
                    }
                }
            }
        }

    services: dict[str, Any] = {
        "postgres": {
            "image": f"postgres:{postgres['image_tag']}",
            "container_name": cname("postgres"),
            "restart": "unless-stopped",
            "environment": _envs(
                {
                    "POSTGRES_DB": postgres["db"],
                    "POSTGRES_USER": postgres["user"],
                    "POSTGRES_PASSWORD": postgres["password"],
                }
            ),
            "ports": [f"{postgres['port']}:5432"],
            "volumes": ["postgres_data:/var/lib/postgresql/data"],
            "healthcheck": healthcheck(["CMD-SHELL", f"pg_isready -U {postgres['user']} -d {postgres['db']}"]),
        },
        "minio": {
            "image": f"quay.io/minio/aistor/minio:{minio['image_tag']}",
            "container_name": cname("minio"),
            "restart": "unless-stopped",
            "command": 'server /data --console-address ":9001" --license /licenses/minio.license',
            "environment": _envs(
                {
                    "MINIO_ROOT_USER": minio["root_user"],
                    "MINIO_ROOT_PASSWORD": minio["root_password"],
                }
            ),
            "ports": [f"{minio['api_port']}:9000", f"{minio['console_port']}:9001"],
            "volumes": ["minio_data:/data", f"{minio['license_path']}:/licenses:ro"],
            "healthcheck": healthcheck(["CMD", "mc", "ready", "local"]),
        },
        "rabbitmq": {
            "image": f"rabbitmq:{rabbitmq['image_tag']}",
            "container_name": cname("rabbitmq"),
            "restart": "unless-stopped",
            "environment": _envs(
                {
                    # infra-service admin credentials: intentionally kept here, not
                    # hoisted/shared with the app containers' own credential blocks
                    "RABBITMQ_DEFAULT_USER": backend["rabbitmq"]["user"],
                    "RABBITMQ_DEFAULT_PASS": backend["rabbitmq"]["password"],
                    "RABBITMQ_DEFAULT_VHOST": rabbitmq["vhost"],
                }
            ),
            "ports": [f"{rabbitmq['port']}:5672", f"{rabbitmq['management_port']}:15672"],
            "volumes": ["rabbitmq_data:/var/lib/rabbitmq"],
            "healthcheck": healthcheck(["CMD", "rabbitmq-diagnostics", "-q", "ping"]),
        },
        "qdrant": {
            "image": f"qdrant/qdrant:{qdrant['image_tag']}",
            "container_name": cname("qdrant"),
            "restart": "unless-stopped",
            "ports": [f"{qdrant['port']}:6333", f"{qdrant['grpc_port']}:6334"],
            "volumes": ["qdrant_data:/qdrant/storage"],
            "healthcheck": healthcheck(
                ["CMD-SHELL", "curl -f http://localhost:6333/healthz || exit 1"], retries=10
            ),
        },
        "backend": {
            "build": {"context": "./backend", "dockerfile": ".dockerfile"},
            "container_name": cname("backend"),
            "restart": "unless-stopped",
            "depends_on": {
                "postgres": {"condition": "service_healthy"},
                "minio": {"condition": "service_healthy"},
                "rabbitmq": {"condition": "service_healthy"},
            },
            "environment": _envs(
                {
                    "APP_NAME": backend["app_name"],
                    "APP_VERSION": backend["app_version"],
                    "PATH_TO_CATALOG": backend["path_to_catalog"],
                    "FASTAPI__HOST": "0.0.0.0",
                    "FASTAPI__PORT": backend["internal_port"],
                    "FASTAPI__DEBUG": backend["debug"],
                    "FASTAPI__TITLE": backend["title"],
                    "DATABASE__HOST": "postgres",
                    "DATABASE__PORT": 5432,
                    "DATABASE__USER": backend["database"]["user"],
                    "DATABASE__PASSWORD": backend["database"]["password"],
                    "DATABASE__DATABASE": postgres["db"],
                    "MINIO__HOST": minio["host"],
                    "MINIO__PORT": minio["port"],
                    "MINIO__ACCESS_KEY": backend["minio"]["access_key"],
                    "MINIO__SECRET_KEY": backend["minio"]["secret_key"],
                    "MINIO__IMAGE_BUCKET": minio["image_bucket"],
                    "RABBITMQ__HOST": rabbitmq["host"],
                    "RABBITMQ__PORT": rabbitmq["port"],
                    "RABBITMQ__USER": backend["rabbitmq"]["user"],
                    "RABBITMQ__PASSWORD": backend["rabbitmq"]["password"],
                    "RABBITMQ__VIRTUAL_HOST": rabbitmq["vhost"],
                    # nested inference_worker / inference_ocr_worker settings - the
                    # queue names here MUST match the corresponding worker's own
                    # rabbitmq.publish_queue / consume_queue, so they're taken from
                    # the single shared "queues" block instead of being typed twice
                    "RABBITMQ__INFERENCE_WORKER__WORKER_NAME": inference["worker_name"],
                    "RABBITMQ__INFERENCE_WORKER__PUBLISH_QUEUE": queues["cv"]["tasks"],
                    "RABBITMQ__INFERENCE_WORKER__CONSUME_QUEUE": queues["cv"]["results"],
                    "RABBITMQ__INFERENCE_OCR_WORKER__WORKER_NAME": ocr["worker_name"],
                    "RABBITMQ__INFERENCE_OCR_WORKER__PUBLISH_QUEUE": queues["ocr"]["tasks"],
                    "RABBITMQ__INFERENCE_OCR_WORKER__CONSUME_QUEUE": queues["ocr"]["results"],
                    "PARSER__BASE_URL": backend["parser"]["base_url"],
                    "PARSER__TIMEOUT": backend["parser"]["timeout"],
                    "PARSER__MAX_RETRIES": backend["parser"]["max_retries"],
                    "PARSER__RETRY_DELAY": backend["parser"]["retry_delay"],
                    "LOGGING__LEVEL": log_cfg["level"],
                    "LOGGING__LOGS_DIRECTORY": log_cfg["logs_directory"],
                    "LOGGING__FILE": log_cfg["file"],
                    "LOGGING__SERIALIZE": log_cfg["serialize"],
                }
            ),
            "ports": [f"{backend['port']}:{backend['internal_port']}"],
        },
        "inference": {
            "build": {"context": inference["build_context"], "dockerfile": inference["dockerfile"]},
            "container_name": cname("inference"),
            "restart": "unless-stopped",
            "depends_on": {
                "minio": {"condition": "service_healthy"},
                "rabbitmq": {"condition": "service_healthy"},
                "qdrant": {"condition": "service_healthy"},
            },
            "environment": _envs(
                {
                    "WORKER_NAME": inference["worker_name"],
                    "WORKER_VERSION": inference["worker_version"],
                    "INFERENCE__MODEL_WEIGHTS_PATH": inference["model_weights_path"],
                    "INFERENCE__YOLO_WEIGHTS_PATH": inference["yolo_weights_path"],
                    "INFERENCE__EMBEDDING_SIZE": inference["embedding_size"],
                    "INFERENCE__IMAGE_SIZE": inference["image_size"],
                    "INFERENCE__DEVICE": inference["device"],
                    "INFERENCE__DATASET_DIR": inference["dataset_dir"],
                    "QDRANT__USE_LOCAL_PATH": inference["use_local_path"],
                    "QDRANT__HOST": qdrant["host"],
                    "QDRANT__PORT": qdrant["port"],
                    "QDRANT__GRPC_PORT": qdrant["grpc_port"],
                    "QDRANT__PREFER_GRPC": qdrant["prefer_grpc"],
                    "QDRANT__COLLECTION_NAME": qdrant["collection_name"],
                    "RABBITMQ__HOST": rabbitmq["host"],
                    "RABBITMQ__PORT": rabbitmq["port"],
                    "RABBITMQ__USERNAME": inference["rabbitmq"]["username"],
                    "RABBITMQ__PASSWORD": inference["rabbitmq"]["password"],
                    "RABBITMQ__PUBLISH_QUEUE": queues["cv"]["tasks"],
                    "RABBITMQ__CONSUME_QUEUE": queues["cv"]["results"],
                    "RABBITMQ__PREFETCH_COUNT": inference["prefetch_count"],
                    "MINIO__ENDPOINT": f"{minio['host']}:{minio['port']}",
                    "MINIO__ACCESS_KEY": inference["minio"]["access_key"],
                    "MINIO__SECRET_KEY": inference["minio"]["secret_key"],
                    "MINIO__SECURE": minio["secure"],
                    "MINIO__IMAGE_BUCKET": minio["image_bucket"],
                    "LOGGING__LEVEL": log_cfg["level"],
                    "LOGGING__LOGS_DIRECTORY": log_cfg["logs_directory"],
                    "LOGGING__FILE": log_cfg["file"],
                    "LOGGING__SERIALIZE": log_cfg["serialize"],
                }
            ),
            **gpu_reservation(),
        },
        "ocr-inference": {
            "build": {"context": ocr["build_context"], "dockerfile": ocr["dockerfile"]},
            "container_name": cname("ocr-inference"),
            "restart": "unless-stopped",
            "depends_on": {
                "minio": {"condition": "service_healthy"},
                "rabbitmq": {"condition": "service_healthy"},
            },
            "environment": _envs(
                {
                    "WORKER_NAME": ocr["worker_name"],
                    "WORKER_VERSION": ocr["worker_version"],
                    "INFERENCE__CATALOG_PATH": ocr["catalog_path"],
                    "INFERENCE__CHECKPOINT_PATH": ocr["checkpoint_path"],
                    "INFERENCE__DEVICE": ocr["device"],
                    "INFERENCE__CRAFT_CANVAS_SIZE": ocr["craft_canvas_size"],
                    "INFERENCE__RECOGNITION_PASSES": ocr["recognition_passes"],
                    "INFERENCE__ALLOW_MULTILINGUAL": ocr["allow_multilingual"],
                    "INFERENCE__TOP_K": ocr["top_k"],
                    "RABBITMQ__HOST": rabbitmq["host"],
                    "RABBITMQ__PORT": rabbitmq["port"],
                    "RABBITMQ__USERNAME": ocr["rabbitmq"]["username"],
                    "RABBITMQ__PASSWORD": ocr["rabbitmq"]["password"],
                    "RABBITMQ__PUBLISH_QUEUE": queues["ocr"]["tasks"],
                    "RABBITMQ__CONSUME_QUEUE": queues["ocr"]["results"],
                    "RABBITMQ__PREFETCH_COUNT": ocr["prefetch_count"],
                    "MINIO__ENDPOINT": f"{minio['host']}:{minio['port']}",
                    "MINIO__ACCESS_KEY": ocr["minio"]["access_key"],
                    "MINIO__SECRET_KEY": ocr["minio"]["secret_key"],
                    "MINIO__SECURE": minio["secure"],
                    "MINIO__IMAGE_BUCKET": minio["image_bucket"],
                    "LOGGING__LEVEL": log_cfg["level"],
                    "LOGGING__LOGS_DIRECTORY": log_cfg["logs_directory"],
                    "LOGGING__FILE": log_cfg["file"],
                    "LOGGING__SERIALIZE": log_cfg["serialize"],
                }
            ),
            **gpu_reservation(),
        },
    }

    return {
        "services": services,
        "volumes": {"postgres_data": None, "minio_data": None, "rabbitmq_data": None, "qdrant_data": None},
    }


def build_run_command(
    compose_path: Path,
    project_name: str,
    detach: bool,
    build: bool,
) -> list[str]:
    command = ["docker", "compose", "-f", str(compose_path), "-p", project_name, "up"]

    if detach:
        command.append("-d")
    if build:
        command.append("--build")

    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a fully-resolved docker-compose file from a JSON config and run "
            "docker compose up (no intermediate .env file - values are baked in directly)."
        )
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Print the default JSON config to stdout and exit (e.g. redirect into docker-config.json to customize).",
    )
    parser.add_argument(
        "--config",
        default="docker-config.json",
        help="Path to a JSON config with overrides on top of the defaults. Optional.",
    )
    parser.add_argument(
        "--compose-output",
        default="docker-compose.generated.yml",
        help="Path to write the fully rendered compose file to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only render the compose file and print the command, without running it.",
    )
    parser.add_argument("--no-detach", action="store_true", help="Run docker compose up in the foreground.")
    parser.add_argument("--no-build", action="store_true", help="Skip --build when running docker compose up.")

    args = parser.parse_args()

    if args.generate_config:
        print(json.dumps(DEFAULTS, indent=2, ensure_ascii=False))
        return

    root = Path.cwd()
    config_path = (root / args.config).resolve()

    overrides: dict[str, Any] = {}
    if config_path.exists():
        overrides = load_json(config_path)
    else:
        print(f"No config file found at {config_path}, using built-in defaults only.")

    merged_config = deep_merge(DEFAULTS, overrides)
    compose_spec = render_compose(merged_config)

    compose_path = (root / args.compose_output).resolve()
    with compose_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(compose_spec, file, sort_keys=False, default_flow_style=False, allow_unicode=True)

    detach = not args.no_detach and bool(merged_config["compose"].get("detach", True))
    build = not args.no_build and bool(merged_config["compose"].get("build", True))

    command = build_run_command(compose_path, merged_config["project_name"], detach, build)

    print(f"Rendered compose file: {compose_path}")
    print("Running command:", " ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()