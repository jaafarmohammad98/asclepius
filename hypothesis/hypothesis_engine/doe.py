"""Two-level fractional factorial designs for the experiment builder (spec D1 to D4).

Each factor is a chosen compound or target at two levels, present (+1) and absent (-1). The design
is built from the standard generators, with no DoE library (D3):

- 2 to 4 factors: the full factorial (4, 8 or 16 runs), nothing aliased; no fraction is needed
  within 16 runs, so this meets D2's "up to 5 factors" with nothing confounded;
- 5 factors: the 2^(5-1) resolution V half fraction, E = ABCD (16 runs), so main effects and
  two-factor interactions are all separable;
- 6 to 8 factors: the 16-run resolution IV fractions (Box, Hunter and Hunter, table 6.22), whose
  main effects are clear of two-factor interactions but whose two-factor interactions are confounded
  with each other, which `Design.note` says;
- more than 8 is refused.

The letters A, B, C… name the factors in the order given; the base factors' columns are the full
factorial in standard order (A alternating fastest), and each generated factor is the product of
the base columns its generator names.

The run order is shuffled by Python's `random.Random(seed)`, so a seed reproduces its order only
here: the web app builds designs on the server, never in the browser.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from math import prod
from random import Random
from secrets import randbelow

from hypothesis_engine.hypotheses import CANDIDATE

MIN_FACTORS = 2
MAX_FACTORS = 8
SEED_LIMIT = 2**31  # a seed is 0 to SEED_LIMIT - 1
LETTERS = "ABCDEFGH"
LEVELS = {1: "present", -1: "absent"}
ROMAN = {4: "IV", 5: "V"}
SAFETY_WARNING = (
    f"# Safety warning: each combination is a {CANDIDATE}, untested, and needs a safety review "
    "before any wet-lab or animal use."
)

# The generated factors of each fraction: a letter, and the base letters whose product it is.
GENERATORS: dict[int, tuple[tuple[str, str], ...]] = {
    5: (("E", "ABCD"),),
    6: (("E", "ABC"), ("F", "BCD")),
    7: (("E", "ABC"), ("F", "BCD"), ("G", "ACD")),
    8: (("E", "BCD"), ("F", "ACD"), ("G", "ABC"), ("H", "ABD")),
}


def _word_product(a: str, b: str) -> str:
    """The product of two effect words: letters in both cancel, since every column squares to 1."""
    return "".join(sorted(set(a) ^ set(b)))


def _one_line(text: str) -> str:
    """The text on one line, so it stays inside its comment."""
    return " ".join(text.split())


@dataclass(frozen=True)
class Design:
    """A two-level design: the ±1 matrix in standard order, one column per factor, and the order
    to carry out its runs in. Run ids are 1-based positions in standard order; `run_order` lists
    them as they are to be run, shuffled by `seed` so the same seed gives the same order."""

    factors: tuple[str, ...]
    generated: tuple[tuple[str, str], ...]
    matrix: tuple[tuple[int, ...], ...]
    seed: int
    run_order: tuple[int, ...]
    replicates: int

    @property
    def generators(self) -> tuple[str, ...]:
        """The generators as written, E=ABCD; empty for a full factorial."""
        return tuple(f"{letter}={word}" for letter, word in self.generated)

    @property
    def defining_relation(self) -> tuple[str, ...]:
        """Every word equal to the identity column I, shortest first: all products of the
        generators' words (E = ABCD gives the word ABCDE)."""
        words = [_word_product(letter, word) for letter, word in self.generated]
        out: set[str] = set()
        for r in range(1, len(words) + 1):
            for chosen in combinations(words, r):
                product_word = ""
                for word in chosen:
                    product_word = _word_product(product_word, word)
                out.add(product_word)
        return tuple(sorted(out, key=lambda w: (len(w), w)))

    @property
    def resolution(self) -> int | None:
        """The length of the shortest word in the defining relation; None for a full factorial."""
        words = self.defining_relation
        return min(len(w) for w in words) if words else None

    @property
    def confounded(self) -> tuple[tuple[str, ...], ...]:
        """The two-factor interactions that cannot be told apart, as alias chains of letter pairs
        (AB, CE…); empty when every two-factor interaction is separable."""
        letters = LETTERS[: len(self.factors)]
        pairs = ["".join(p) for p in combinations(letters, 2)]
        words = self.defining_relation
        chains = []
        seen: set[str] = set()
        for pair in pairs:
            if pair in seen:
                continue
            chain = [pair] + sorted(q for w in words if len(q := _word_product(pair, w)) == 2)
            seen.update(chain)
            if len(chain) > 1:
                chains.append(tuple(chain))
        return tuple(chains)

    @property
    def label(self) -> str:
        """The design's name, its resolution and its runs: Full factorial 2^3, 8 runs; Fractional
        factorial 2^(5-1), resolution V, 16 runs."""
        k, runs = len(self.factors), len(self.matrix)
        if self.resolution is None:
            return f"Full factorial 2^{k}, {runs} runs"
        return f"Fractional factorial 2^({k}-{len(self.generated)}), resolution {ROMAN[self.resolution]}, {runs} runs"

    @property
    def note(self) -> str:
        """What the design can and cannot separate, in the factors' names, to show beside it."""
        if self.resolution is None or not self.confounded:
            return "Main effects and two-factor interactions are all separable."
        name = dict(zip(LETTERS, self.factors, strict=False))
        chains = "; ".join(" = ".join(f"{name[p[0]]} × {name[p[1]]}" for p in c) for c in self.confounded)
        return (
            f"Resolution {ROMAN[self.resolution]}: main effects are clear of two-factor interactions, "
            f"but two-factor interactions are confounded with each other: {chains}."
        )

    def csv(self, responses: Sequence[str] = ("response",), risks: Mapping[str, str] | None = None) -> str:
        """The runs in run order as CSV under the safety warning (D4): run id, run order, seed, one
        column per factor (present or absent), the replicate count, and an empty column for each
        response, one per replicate (`libido_1`, `libido_2`…) when there is more than one.

        `risks` says each factor's known risks (rule 6), written under the warning as a `#` line per
        factor labelled a candidate for study; it must say something for every factor and name no
        other. What it says (unknown, say, rather than none) is the caller's to choose."""
        if self.replicates == 1:
            measured = list(responses)
        else:
            measured = [f"{r}_{i}" for r in responses for i in range(1, self.replicates + 1)]
        header = ["run_id", "run_order", "seed", *self.factors, "replicates", *measured]
        if len(set(header)) < len(header):
            raise ValueError("Each column needs its own name: " + ", ".join(header))
        out = io.StringIO()
        out.write(SAFETY_WARNING + "\n")
        if risks is not None:
            said = {f: _one_line(r) for f, r in risks.items()}
            odd = sorted((set(said) ^ set(self.factors)) | {f for f in self.factors if not said.get(f)})
            if odd:
                raise ValueError(
                    "The risks must say something for every factor and no other: " + ", ".join(odd)
                )
            for factor in self.factors:
                out.write(f"# {_one_line(factor)} ({CANDIDATE}), known risks: {said[factor]}\n")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        for position, run_id in enumerate(self.run_order, start=1):
            levels = [LEVELS[level] for level in self.matrix[run_id - 1]]
            writer.writerow([run_id, position, self.seed, *levels, self.replicates, *[""] * len(measured)])
        return out.getvalue()


def design(factors: Sequence[str], *, seed: int | None = None, replicates: int = 1) -> Design:
    """The design for these factors (2 to 8), its run order shuffled by `seed` (drawn when None)."""
    names = tuple(f.strip() for f in factors)
    k = len(names)
    if not MIN_FACTORS <= k <= MAX_FACTORS:
        raise ValueError(f"A design takes {MIN_FACTORS} to {MAX_FACTORS} factors, not {k}.")
    if not all(names) or len(set(names)) < k:
        raise ValueError("Each factor needs its own name.")
    if replicates < 1:
        raise ValueError(f"A design needs at least one replicate, not {replicates}.")
    generated = GENERATORS.get(k, ())
    base_letters = LETTERS[: k - len(generated)]
    matrix = []
    for base_levels in product((-1, 1), repeat=len(base_letters)):
        run = dict(zip(base_letters, reversed(base_levels), strict=True))
        for letter, word in generated:
            run[letter] = prod(run[b] for b in word)
        matrix.append(tuple(run[letter] for letter in LETTERS[:k]))
    if seed is None:
        seed = randbelow(SEED_LIMIT)
    order = list(range(1, len(matrix) + 1))
    Random(seed).shuffle(order)
    return Design(
        factors=names,
        generated=generated,
        matrix=tuple(matrix),
        seed=seed,
        run_order=tuple(order),
        replicates=replicates,
    )
