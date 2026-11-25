// TOPSIS for the drug repurposing leaderboard (M6-07, spec R4): the page ranks with it as a weight
// slider moves, without asking the server. It is `_closeness` and the ordering of `rank` in
// hypothesis_engine/topsis.py, the reference, given each drug's criteria (stand-ins included) as the
// leaderboard sends them; test_web_topsis_parity.py checks the two rank alike.
"use strict";

const TOPSIS_CRITERIA = ["impact", "coverage", "safety", "bbb", "readiness"]; // topsis.CRITERIA
const TOPSIS_COST = new Set(["safety"]); // the lower, the better; the rest are benefits

/** Each row's closeness to the ideal best (0 to 1), in the rows' order: each criterion vector-normalised
 * and weighted by its share of the weights' sum. A row that cannot be told from the rest scores 0.5. */
function topsisScores(rows, weights) {
  const total = TOPSIS_CRITERIA.reduce((sum, c) => sum + weights[c], 0);
  const shares = TOPSIS_CRITERIA.map((c) => weights[c] / total);
  const matrix = rows.map((r) => TOPSIS_CRITERIA.map((c) => r.criteria[c]));
  const norms = TOPSIS_CRITERIA.map((_, j) => Math.sqrt(matrix.reduce((sum, row) => sum + row[j] ** 2, 0)));
  const weighted = matrix.map((row) => row.map((x, j) => (norms[j] ? (shares[j] * x) / norms[j] : 0)));
  const column = (j) => weighted.map((row) => row[j]);
  const best = TOPSIS_CRITERIA.map((c, j) => (TOPSIS_COST.has(c) ? Math.min : Math.max)(...column(j)));
  const worst = TOPSIS_CRITERIA.map((c, j) => (TOPSIS_COST.has(c) ? Math.max : Math.min)(...column(j)));
  const distance = (row, to) => Math.hypot(...row.map((x, j) => x - to[j]));
  return weighted.map((row) => {
    const toBest = distance(row, best), toWorst = distance(row, worst);
    return toBest + toWorst ? toWorst / (toBest + toWorst) : 0.5;
  });
}

/** The rows, each with its `score`, highest first and then by name. */
function topsisRank(rows, weights) {
  const scores = topsisScores(rows, weights);
  return rows
    .map((r, i) => ({ ...r, score: scores[i] }))
    .sort((a, b) => b.score - a.score || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

if (typeof module === "object") module.exports = { TOPSIS_CRITERIA, TOPSIS_COST, topsisScores, topsisRank };
