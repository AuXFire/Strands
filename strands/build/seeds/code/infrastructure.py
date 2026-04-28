"""Infrastructure (IN) — container, ci-cd, cloud, monitoring."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: container
    (0, 0x00, ["docker", "container"]),
    (0, 0x01, ["kubernetes", "k8s"]),
    (0, 0x02, ["pod"]),
    (0, 0x03, ["service"]),
    (0, 0x04, ["ingress"]),
    (0, 0x05, ["volume"]),
    (0, 0x06, ["image"]),
    # Category 1: ci-cd
    (1, 0x00, ["build"]),
    (1, 0x01, ["test"]),
    (1, 0x02, ["deploy", "deployment"]),
    (1, 0x03, ["release"]),
    (1, 0x04, ["rollback"]),
    (1, 0x05, ["pipeline"]),
    (1, 0x06, ["workflow"]),
    (1, 0x07, ["artifact"]),
    # Category 2: cloud
    (2, 0x00, ["compute", "instance"]),
    (2, 0x01, ["storage", "bucket"]),
    (2, 0x02, ["queue"]),
    (2, 0x03, ["cache"]),
    (2, 0x04, ["cdn"]),
    (2, 0x05, ["serverless"]),
    (2, 0x06, ["lambda"]),
    (2, 0x07, ["function"]),
    # Category 3: monitoring
    (3, 0x00, ["metric"]),
    (3, 0x01, ["log"]),
    (3, 0x02, ["trace"]),
    (3, 0x03, ["alert"]),
    (3, 0x04, ["dashboard"]),
    (3, 0x05, ["healthcheck", "ping"]),
    (3, 0x06, ["sla", "slo"]),
    (3, 0x07, ["uptime"]),
]
