"""Keep only PSSD / PFS / PAS material from a raw subreddit download (free, no API calls).

For a subreddit that is mostly about something else (r/DrWillPowers: hormone therapy), run this on
the raw file before `load`. A post is kept when its title, text or flair names one of the three
conditions or a drug that causes them. A comment is kept when it does, or when it sits under a kept
post (replies in a PSSD thread often never name the condition).

    .venv/bin/python topic_filter.py --inp data/DrWillPowers_raw.jsonl --out data/DrWillPowers_topic_raw.jsonl

The raw input is only read. Wide on purpose: the extraction step labels each case's condition and
drops the ones that are not PSSD, PFS or PAS.
"""
import argparse, json, re
from pathlib import Path

from extract_cases import iter_ndjson

TOPIC = re.compile(
    r"\b(pssd|post[- ]?ssri|pfs|post[- ]?finasteride|finasteride|propecia|proscar|dutasteride|"
    r"post[- ]?accutane|accutane|isotretinoin|"
    r"ssris?|snris?|sertraline|zoloft|escitalopram|lexapro|cipralex|citalopram|celexa|fluoxetine|prozac|"
    r"paroxetine|paxil|seroxat|fluvoxamine|venlafaxine|effexor|desvenlafaxine|pristiq|duloxetine|cymbalta|"
    r"vortioxetine|trintellix|clomipramine)\b",
    re.I,
)


def text(o):
    return o.get("body") or f"{o.get('title', '')}\n{o.get('selftext', '')}\n{o.get('link_flair_text') or ''}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    if Path(a.out).resolve() == Path(a.inp).resolve():
        raise SystemExit("--out must differ from --inp: the raw file is never overwritten")
    posts = set()
    for o in iter_ndjson(Path(a.inp)):
        if "body" not in o and TOPIC.search(text(o)):
            posts.add(o.get("name") or f"t3_{o['id']}")
    kept = {"post": 0, "comment": 0, "comment_in_thread": 0}; total = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for o in iter_ndjson(Path(a.inp)):
            total += 1
            if "body" not in o:
                if (o.get("name") or f"t3_{o['id']}") in posts:
                    out.write(json.dumps(o) + "\n"); kept["post"] += 1
            elif TOPIC.search(text(o)):
                out.write(json.dumps(o) + "\n"); kept["comment"] += 1
            elif o.get("link_id") in posts:
                out.write(json.dumps(o) + "\n"); kept["comment_in_thread"] += 1
    print(f"kept {sum(kept.values())}/{total} ({kept}) -> {a.out}")


if __name__ == "__main__":
    main()
