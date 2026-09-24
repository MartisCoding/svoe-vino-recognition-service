#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "project_name": "vino-service",

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
        "root_user": "minioadmin",
        "root_password": "minioadmin",
        "image_bucket": "images",
        "secure": False,
        "image_prefix": "client",
    },

    "rabbitmq": {
        "image_tag": "3.13-management",
        "port": 5672,
        "management_port": 15672,
        "user": "guest",
        "password": "guest",
        "vhost": "/",
        "recognition_queue": "recognition.tasks",
        "result_queue": "recognition.results",
        "inference_prefetch_count": 1,
        "inference_url": "amqp://guest:guest@rabbitmq:5672/",
    },

    "qdrant": {
        "image_tag": "latest",
        "port": 6333,
        "grpc_port": 6334,
        "prefer_grpc": False,
        "collection_name": "wines",
    },

    "inference": {
        "build_context": "./vino-inference",
        "dockerfile": "Dockerfile",
        "model_weights_path": "/app/best_arcface_model.pth",
        "yolo_weights_path": "/app/best.pt",
        "embedding_size": 512,
        "image_size": 224,
        "device": "cuda:0",
    },

    "backend": {
        "port": 8080,
        "internal_port": 8080,
        "app_name": "vino-service",
        "app_version": "0.1.0",
        "debug": False,
        "title": "vino-back",

        "parser_base_url": "https://vino-svoe.ru",
        "parser_timeout": 10,
        "parser_max_retries": 3,
        "parser_retry_delay": 1,

        "logging_level": "INFO",
        "logging_logs_directory": "./logs/",
        "logging_file": "",
        "logging_serialize": False,
    },

    "compose": {
        "detach": True,
        "build": True,
    },
}


def deep_merge(
    defaults: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    result = dict(defaults)

    for key, value in values.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_merge(result[key], value)
            continue

        result[key] = value

    return result


def to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def render_env(config: dict[str, Any]) -> str:
    postgres = config["postgres"]
    minio = config["minio"]
    rabbitmq = config["rabbitmq"]
    qdrant = config["qdrant"]
    inference = config["inference"]
    backend = config["backend"]

    values: dict[str, str] = {
        "PROJECT_NAME": str(config["project_name"]),

        "POSTGRES_IMAGE_TAG": str(postgres["image_tag"]),
        "POSTGRES_PORT": str(postgres["port"]),
        "POSTGRES_DB": str(postgres["db"]),
        "POSTGRES_USER": str(postgres["user"]),
        "POSTGRES_PASSWORD": str(postgres["password"]),

        "MINIO_IMAGE_TAG": str(minio["image_tag"]),
        "MINIO_API_PORT": str(minio["api_port"]),
        "MINIO_CONSOLE_PORT": str(minio["console_port"]),
        "MINIO_ROOT_USER": str(minio["root_user"]),
        "MINIO_ROOT_PASSWORD": str(minio["root_password"]),
        "MINIO_IMAGE_BUCKET": str(minio["image_bucket"]),
        "MINIO_SECURE": to_bool_str(bool(minio["secure"])),
        "MINIO_IMAGE_PREFIX": str(minio["image_prefix"]),

        "RABBITMQ_IMAGE_TAG": str(rabbitmq["image_tag"]),
        "RABBITMQ_PORT": str(rabbitmq["port"]),
        "RABBITMQ_MANAGEMENT_PORT": str(
            rabbitmq["management_port"]
        ),
        "RABBITMQ_DEFAULT_USER": str(rabbitmq["user"]),
        "RABBITMQ_DEFAULT_PASS": str(rabbitmq["password"]),
        "RABBITMQ_DEFAULT_VHOST": str(rabbitmq["vhost"]),
        "RABBITMQ_RECOGNITION_QUEUE": str(
            rabbitmq["recognition_queue"]
        ),
        "RABBITMQ_RESULT_QUEUE": str(
            rabbitmq["result_queue"]
        ),
        "RABBITMQ_INFERENCE_PREFETCH_COUNT": str(
            rabbitmq["inference_prefetch_count"]
        ),
        "RABBITMQ_INFERENCE_URL": str(
            rabbitmq["inference_url"]
        ),

        "QDRANT_IMAGE_TAG": str(qdrant["image_tag"]),
        "QDRANT_PORT": str(qdrant["port"]),
        "QDRANT_GRPC_PORT": str(qdrant["grpc_port"]),
        "QDRANT_INTERNAL_PORT": "6333",
        "QDRANT_INTERNAL_GRPC_PORT": "6334",
        "QDRANT_PREFER_GRPC": to_bool_str(
            bool(qdrant["prefer_grpc"])
        ),
        "QDRANT_COLLECTION_NAME": str(
            qdrant["collection_name"]
        ),

        "INFERENCE_BUILD_CONTEXT": str(
            inference["build_context"]
        ),
        "INFERENCE_DOCKERFILE": str(
            inference["dockerfile"]
        ),
        "INFERENCE_MODEL_WEIGHTS_PATH": str(
            inference["model_weights_path"]
        ),
        "INFERENCE_YOLO_WEIGHTS_PATH": str(
            inference["yolo_weights_path"]
        ),
        "INFERENCE_EMBEDDING_SIZE": str(
            inference["embedding_size"]
        ),
        "INFERENCE_IMAGE_SIZE": str(
            inference["image_size"]
        ),
        "INFERENCE_DEVICE": str(
            inference["device"]
        ),

        "BACKEND_PORT": str(backend["port"]),
        "BACKEND_INTERNAL_PORT": str(backend["internal_port"]),
        "APP_NAME": str(backend["app_name"]),
        "APP_VERSION": str(backend["app_version"]),
        "FASTAPI_DEBUG": to_bool_str(bool(backend["debug"])),
        "FASTAPI_TITLE": str(backend["title"]),

        "PARSER_BASE_URL": str(backend["parser_base_url"]),
        "PARSER_TIMEOUT": str(backend["parser_timeout"]),
        "PARSER_MAX_RETRIES": str(
            backend["parser_max_retries"]
        ),
        "PARSER_RETRY_DELAY": str(
            backend["parser_retry_delay"]
        ),

        "LOGGING_LEVEL": str(backend["logging_level"]),
        "LOGGING_LOGS_DIRECTORY": str(
            backend["logging_logs_directory"]
        ),
        "LOGGING_FILE": str(backend["logging_file"]),
        "LOGGING_SERIALIZE": to_bool_str(
            bool(backend["logging_serialize"])
        ),
    }

    lines = [
        f"{key}={value}"
        for key, value in values.items()
    ]

    return "\n".join(lines) + "\n"


def build_compose_command(
    compose_path: Path,
    env_path: Path,
    config: dict[str, Any],
) -> list[str]:
    compose_cfg = config["compose"]

    command = [
        "docker",
        "compose",
        "-f",
        str(compose_path),
        "--env-file",
        str(env_path),
        "up",
    ]

    if bool(compose_cfg["detach"]):
        command.append("-d")

    if bool(compose_cfg["build"]):
        command.append("--build")

    return command


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "JSON config root must be an object."
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Docker Compose env file from JSON "
            "and run docker compose up."
        )
    )

    parser.add_argument(
        "--config",
        default="docker-config.json",
        help=(
            "Path to JSON config file "
            "(default: docker-config.json)."
        ),
    )

    parser.add_argument(
        "--compose-file",
        default="docker-compose.yml",
        help=(
            "Path to docker compose file "
            "(default: docker-compose.yml)."
        ),
    )

    parser.add_argument(
        "--env-output",
        default=".docker.generated.env",
        help=(
            "Path to generated env file "
            "(default: .docker.generated.env)."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only generate env and print command, "
            "do not run docker compose."
        ),
    )

    args = parser.parse_args()

    root = Path.cwd()

    config_path = (root / args.config).resolve()
    compose_path = (root / args.compose_file).resolve()
    env_path = (root / args.env_output).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )

    if not compose_path.exists():
        raise FileNotFoundError(
            f"Compose file does not exist: {compose_path}"
        )

    user_config = load_json(config_path)

    merged_config = deep_merge(
        DEFAULTS,
        user_config,
    )

    env_content = render_env(merged_config)

    env_path.write_text(
        env_content,
        encoding="utf-8",
    )

    command = build_compose_command(
        compose_path,
        env_path,
        merged_config,
    )

    print(f"Generated env file: {env_path}")
    print(
        "Running command:",
        " ".join(command),
    )

    if args.dry_run:
        return

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()