"""Two-level fractional factorial designs for the experiment builder (spec D1 to D4).

Each factor is a chosen compound or target at two levels, present (+1) and absent (-1). The design
is built from the standard generators, with no DoE library (D3):

- 2 to 4 factors: the full factorial (4, 8 or 16 runs), nothing aliased;
- 5 factors: the 2^(5-1) resolution V half fraction, E = ABCD (16 runs), so main effects and
  two-factor interactions are all separable;
- 6 to 8 factors: the 16-run resolution IV fractions (Box, Hunter and Hunter, table 6.22), whose
  main effects are clear of two-factor interactions but whose two-factor interactions are confounded
  with each other, which `Design.note` says;
- more than 8 is refused.

The letters A, B, C… name the factors in the order given; the base factors' columns are the full
factorial in standard order (A alternating fastest), and each generated factor is the product of
the base columns its generator names.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from random import Random
from secrets import randbelow

MIN_FACTORS = 2
MAX_FACTORS = 8
LETTERS = "ABCDEFGH"
LEVELS = {1: "present", -1: "absent"}
SAFETY_WARNING = (
    "# Safety warning: these combinations are untested. Each needs a safety review before any "
    "wet-lab or animal use. Candidates for study only; not treatments."
)

# The generated factors of each fraction: a letter and the base letters whose product it is.
GENERATORS: dict[int, tuple[str, ...]] = {
    5: ("E=ABCD",),
    6: ("E=ABC", "F=BCD"),
    7: ("E=ABC", "F=BCD", "G=ACD"),
    8: ("E=BCD", "F=ACD", "G=ABC", "H=ABD"),
}


def _times(a: str, b: str) -> str:
    """The product of two effect words: letters in both cancel, since every column squares to 1."""
    return "".join(sorted(set(a) ^ set(b)))


@dataclass(frozen=True)
class Design:
    """A two-level design: the ±1 matrix in standard order, one column per factor, and the order
    to carry out its runs in. Run ids are 1-based positions in standard order; `run_order` lists
    them as they are to be run, shuffled by `seed` so the same seed gives the same order."""

    factors: tuple[str, ...]
    generators: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]
    seed: int
    run_order: tuple[int, ...]
    replicates: int

    @property
    def defining_relation(self) -> tuple[str, ...]:
        """Every word equal to the identity column I, shortest first: all products of the
        generators' words (E = ABCD gives the word ABCDE)."""
        words = [_times(g.split("=")[0], g.split("=")[1]) for g in self.generators]
        out: set[str] = set()
        for r in range(1, len(words) + 1):
            for chosen in combinations(words, r):
                w = ""
                for x in chosen:
                    w = _times(w, x)
                out.add(w)
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
        for p in pairs:
            if p in seen:
                continue
            chain = [p] + sorted(q for w in words if len(q := _times(p, w)) == 2)
            seen.update(chain)
            if len(chain) > 1:
                chains.append(tuple(chain))
        return tuple(chains)

    def csv(self, responses: Sequence[str] = ("response",)) -> str:
        """The runs in run order as CSV under the safety warning (D4): run id, run order, seed, one
        column per factor (present or absent), the replicate count, and an empty column for each
        response, one per replicate (`libido_1`, `libido_2`…) when there is more than one."""
        if self.replicates == 1:
            measured = list(responses)
        else:
            measured = [f"{r}_{i}" for r in responses for i in range(1, self.replicates + 1)]
        header = ["run_id", "run_order", "seed", *self.factors, "replicates", *measured]
        if len(set(header)) < len(header):
            raise ValueError("Each column needs its own name: " + ", ".join(header))
        out = io.StringIO()
        out.write(SAFETY_WARNING + "\n")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        for position, run_id in enumerate(self.run_order, start=1):
            levels = [LEVELS[x] for x in self.matrix[run_id - 1]]
            writer.writerow([run_id, position, self.seed, *levels, self.replicates, *[""] * len(measured)])
        return out.getvalue()

    @property
    def note(self) -> str:
        """What the design can and cannot separate, in the factors' names, to show beside it."""
        if not self.confounded:
            return "Main effects and two-factor interactions are all separable."
        name = dict(zip(LETTERS, self.factors, strict=False))
        chains = "; ".join(" = ".join(f"{name[p[0]]} × {name[p[1]]}" for p in c) for c in self.confounded)
        return (
            f"Resolution {self.resolution}: main effects are clear of two-factor interactions, but "
            f"two-factor interactions are confounded with each other: {chains}."
        )


def design(factors: Sequence[str], *, seed: int | None = None, replicates: int = 1) -> Design:
    """The design for these factors (2 to 8)."""
    names = tuple(f.strip() for f in factors)
    k = len(names)
    if not MIN_FACTORS <= k <= MAX_FACTORS:
        raise ValueError(f"A design takes {MIN_FACTORS} to {MAX_FACTORS} factors, not {k}.")
    if not all(names) or len(set(names)) < k:
        raise ValueError("Each factor needs its own name.")
    if replicates < 1:
        raise ValueError(f"A design needs at least one replicate, not {replicates}.")
    generators = GENERATORS.get(k, ())
    base = k - len(generators)
    letters = LETTERS[:k]
    matrix = []
    for levels in product((-1, 1), repeat=base):
        run = dict(zip(letters[:base], reversed(levels), strict=True))
        for g in generators:
            letter, word = g.split("=")
            x = 1
            for b in word:
                x *= run[b]
            run[letter] = x
        matrix.append(tuple(run[x] for x in letters))
    if seed is None:
        seed = randbelow(2**31)
    order = list(range(1, len(matrix) + 1))
    Random(seed).shuffle(order)
    return Design(
        factors=names,
        generators=generators,
        matrix=tuple(matrix),
        seed=seed,
        run_order=tuple(order),
        replicates=replicates,
    )
