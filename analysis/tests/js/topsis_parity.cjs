// Ranks each case with the page's TOPSIS (asclepius/web/topsis.js), as test_web_topsis_parity.py
// asks: reads [{rows: [{name, criteria}], weights}] on stdin, writes [[{name, score}]] to stdout.
//   node topsis_parity.cjs < cases.json
const { topsisRank } = require("../../asclepius/web/topsis.js");

let input = "";
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => {
  const cases = JSON.parse(input);
  const out = cases.map((c) => topsisRank(c.rows, c.weights).map((r) => ({ name: r.name, score: r.score })));
  process.stdout.write(JSON.stringify(out));
});
