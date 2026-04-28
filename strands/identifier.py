"""Identifier splitting + abbreviation expansion (spec §7.3).

Splits programmer identifiers into their constituent words so the encoder
can route each part through the text codebook for semantic content while
the structural keywords route through the code codebook.
"""

from __future__ import annotations

import re

# Common programming abbreviations → expansions (spec §7.3 table).
ABBREVIATIONS: dict[str, str] = {
    "db": "database", "conn": "connection", "req": "request", "res": "response",
    "msg": "message", "btn": "button", "cfg": "config", "env": "environment",
    "auth": "authentication", "admin": "administrator", "api": "interface",
    "url": "address", "uri": "address", "err": "error", "fn": "function",
    "cb": "callback", "ctx": "context", "src": "source", "dst": "destination",
    "tmp": "temporary", "prev": "previous", "curr": "current", "idx": "index",
    "len": "length", "num": "number", "str": "string", "int": "integer",
    "bool": "boolean", "obj": "object", "arr": "array", "elem": "element",
    "attr": "attribute", "param": "parameter", "arg": "argument",
    "val": "value", "ref": "reference", "ptr": "pointer", "addr": "address",
    "doc": "document", "docs": "documents",
    "info": "information", "init": "initialize", "max": "maximum",
    "min": "minimum", "avg": "average", "calc": "calculate",
    "img": "image", "lib": "library", "pkg": "package", "mod": "module",
    "ver": "version", "regex": "pattern", "repo": "repository",
    "svc": "service", "cli": "interface", "ui": "interface", "ux": "interface",
    "id": "identifier", "uid": "identifier", "uuid": "identifier",
    "ws": "websocket", "ip": "address",
    "dir": "directory", "path": "path", "ext": "extension",
}


_ALL_CAPS_BLOCK = re.compile(r"^[A-Z0-9_]+$")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^A-Za-z0-9]+")


def split_identifier(identifier: str) -> list[str]:
    """Split a programming identifier into lowercase word parts.

    Handles camelCase, PascalCase, snake_case, kebab-case, SCREAMING_SNAKE,
    and digit boundaries. Returns lowercase tokens with abbreviations expanded.

    Examples:
      fetchUserData    -> ["fetch", "user", "data"]
      get_active_users -> ["get", "active", "users"]
      HttpResponseCode -> ["http", "response", "code"]
      MAX_RETRY_COUNT  -> ["max", "retry", "count"]
      user-profile-api -> ["user", "profile", "api"]
      getDBConn        -> ["get", "database", "connection"]
    """
    if not identifier:
        return []

    # Treat ALL-CAPS-with-underscores as snake_case after lowercasing.
    if _ALL_CAPS_BLOCK.fullmatch(identifier):
        identifier = identifier.lower()

    # Split on non-word separators first (snake/kebab/dot).
    chunks = [c for c in _NON_WORD.split(identifier) if c]

    parts: list[str] = []
    for chunk in chunks:
        for piece in _CAMEL_BOUNDARY.split(chunk):
            piece = piece.strip()
            if not piece:
                continue
            for sub in re.findall(r"[A-Za-z]+|[0-9]+", piece):
                parts.append(sub.lower())

    expanded: list[str] = []
    for p in parts:
        expanded.append(ABBREVIATIONS.get(p, p))
    return expanded
