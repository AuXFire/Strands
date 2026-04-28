"""API (AP) — endpoint, auth, schema, rate."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: endpoint
    (0, 0x00, ["endpoint"]),
    (0, 0x01, ["route"]),
    (0, 0x02, ["path"]),
    (0, 0x03, ["handler"]),
    (0, 0x04, ["controller"]),
    (0, 0x05, ["middleware"]),
    (0, 0x06, ["router"]),
    # Category 1: auth
    (1, 0x00, ["token"]),
    (1, 0x01, ["session"]),
    (1, 0x02, ["cookie"]),
    (1, 0x03, ["oauth"]),
    (1, 0x04, ["jwt"]),
    (1, 0x05, ["apikey"]),
    (1, 0x06, ["login"]),
    (1, 0x07, ["logout"]),
    (1, 0x08, ["authorize"]),
    (1, 0x09, ["authenticate"]),
    # Category 2: schema
    (2, 0x00, ["schema"]),
    (2, 0x01, ["openapi", "swagger"]),
    (2, 0x02, ["graphql"]),
    (2, 0x03, ["protobuf"]),
    (2, 0x04, ["jsonschema"]),
    (2, 0x05, ["avro"]),
    # Category 3: rate
    (3, 0x00, ["limit", "ratelimit"]),
    (3, 0x01, ["throttle"]),
    (3, 0x02, ["retry"]),
    (3, 0x03, ["backoff"]),
    (3, 0x04, ["circuitbreaker"]),
    (3, 0x05, ["timeout"]),
]
