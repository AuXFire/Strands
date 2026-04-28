"""Click-based CLI: strand encode | compare | search | inspect | build-codebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from strands import __version__
from strands.codebook import Codebook, default_codebook
from strands.comparator import compare_strands
from strands.encoder import encode
from strands.index import InMemoryIndex


@click.group()
@click.version_option(__version__, prog_name="strand")
def main() -> None:
    """Semantic Strands CLI."""


@main.command("encode")
@click.argument("text")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "binary", "json"]),
    default="text",
    help="Output format.",
)
def encode_cmd(text: str, fmt: str) -> None:
    """Encode TEXT into a semantic strand."""
    result = encode(text)
    if fmt == "text":
        click.echo(result.strand_text)
    elif fmt == "binary":
        sys.stdout.buffer.write(result.strand.to_binary())
    else:  # json
        click.echo(
            json.dumps(
                {
                    "strand": result.strand_text,
                    "byte_size": result.byte_size,
                    "codons": [
                        {
                            "word": e.word,
                            "codon": e.codon.to_str(),
                            "shade": f"{e.shade:02X}",
                        }
                        for e in result.strand.codons
                    ],
                    "unknowns": result.unknowns,
                },
                indent=2,
            )
        )


@main.command("compare")
@click.argument("text_a")
@click.argument("text_b")
@click.option("--explain", is_flag=True, help="Show per-codon match breakdown.")
def compare_cmd(text_a: str, text_b: str, explain: bool) -> None:
    """Compare two strings and report a similarity score."""
    a = encode(text_a)
    b = encode(text_b)
    result = compare_strands(a.strand, b.strand)
    if explain:
        click.echo(result.explain())
    else:
        click.echo(f"{result.score:.4f}")


@main.command("inspect")
@click.argument("token")
def inspect_cmd(token: str) -> None:
    """Show the codebook entry for a single TOKEN."""
    cb = default_codebook()
    entry = cb.lookup(token)
    if entry is None:
        click.echo(json.dumps({"found": False, "token": token}))
        sys.exit(1)
    click.echo(
        json.dumps(
            {
                "found": True,
                "token": token,
                "codon": entry.codon.to_str(),
                "domain": entry.domain_code,
                "category": entry.codon.category,
                "concept": entry.codon.concept,
                "shade_hint": entry.shade_hint,
                "synonyms": cb.synonyms_of(token)[:10],
            },
            indent=2,
        )
    )


@main.command("search")
@click.option("--corpus", type=click.Path(exists=True, dir_okay=False), required=True,
              help="JSONL file with {id, content} per line.")
@click.option("--query", required=True, help="Search query.")
@click.option("--top-k", type=int, default=10)
@click.option("--threshold", type=float, default=0.0)
def search_cmd(corpus: str, query: str, top_k: int, threshold: float) -> None:
    """Search a JSONL corpus for matches against QUERY."""
    index = InMemoryIndex()
    with open(corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            index.add(id=obj["id"], content=obj["content"])

    results = index.search(query, top_k=top_k, threshold=threshold)
    for r in results:
        click.echo(
            json.dumps(
                {"id": r.entry.id, "score": round(r.score, 4), "content": r.entry.content}
            )
        )


@main.command("build-codebook")
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=str(Path(__file__).parent / "data" / "codebook_v0.1.0.json"),
    show_default=True,
)
@click.option(
    "--frequency-threshold",
    type=float,
    default=None,
    help="Drop non-seed entries below this Zipf frequency. Default: include "
    "the full WordNet vocabulary.",
)
def build_codebook_cmd(output: str, frequency_threshold: float | None) -> None:
    """Build the codebook JSON from seed concepts + WordNet."""
    from strands.build.assemble import write

    codebook = write(output, frequency_threshold=frequency_threshold)
    click.echo(
        f"Wrote {len(codebook['entries'])} entries across "
        f"{len(codebook['domains'])} domains → {output}"
    )


if __name__ == "__main__":
    main()
