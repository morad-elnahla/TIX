"""Talk to the REAL agent (real Groq model + real TIX API) from the terminal, with a trace of every tool call.

    python chat_cli.py                          # interactive
    python chat_cli.py --script scenarios.txt   # replay conversations (blank-line-separated "---"), one message per line
    python chat_cli.py --raw-media 1070         # print the raw image/video fields of a product (to confirm media field names)
    python chat_cli.py --catalog-check "Redmi A5"   # where does a product live: snapshot? live search? live category listing?
    python chat_cli.py --check-llm              # which configured Groq models exist for your key, and do they answer + call tools?

Needs GROQ_API_KEY (env or .env) and network access to the TIX API."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from agent.agent import SalesAgent
from agent.state import Session


def catalog_check(term: str) -> None:
    """Find out WHY a product the store has might not be found: compares the cached snapshot with the live API."""
    from agent.guards import normalize
    from tix.catalog import Catalog
    from tix.client import TixClient
    from tix.schemas import Product, extract_items, extract_last_page
    client = TixClient()
    cat = Catalog(client)
    cat.load()
    print("snapshot:", json.dumps(cat.status(), ensure_ascii=False))
    if cat.missing:
        print("  !! INCOMPLETE snapshot: pages", cat.missing, "could not be downloaded (products on them are invisible to the cache)")
    want = normalize(term)
    hits = [p for p in cat.products if want in normalize(p.name)]
    print(f"in snapshot: {len(hits)} product(s) matching {term!r}")
    for p in hits:
        print(f"   id={p.id} price={p.price} path={p.path!r} sub_id={p.subcategory_id}")
    live = [Product.from_raw(r) for r in extract_items(client.search(term))]
    print(f"live /api/search?q={term}: {len(live)} -> {[(p.id, p.name[:40]) for p in live][:6]}")
    for sid in sorted({p.subcategory_id for p in hits if p.subcategory_id is not None}):
        rows, first = [], client.subcategory_products(sid)
        rows += extract_items(first)
        for pg in range(2, extract_last_page(first) + 1):
            rows += extract_items(client.subcategory_products(sid, pg))
        names = [Product.from_raw(r) for r in rows]
        print(f"live subcategory {sid}: {len(names)} products; contains {term!r}: {any(want in normalize(p.name) for p in names)}")
    if not hits:
        print("-> not in the snapshot. If live search or a live category listing has it, the snapshot is stale/incomplete.")


def check_llm() -> None:
    import config
    from agent.llm import GroqLLM, LLMError
    from agent.tools import SCHEMAS
    try:
        g = GroqLLM()
    except LLMError as e:
        sys.exit(f"{e}: put GROQ_API_KEY in .env")
    try:
        available = sorted(m.id for m in g._client.models.list().data)
        print("Models Groq lists for this key:\n  " + "\n  ".join(available))
    except Exception as e:  # noqa: BLE001
        available = []
        print("Could not list models:", type(e).__name__, str(e)[:120])
    for m in config.GROQ_MODELS:
        listed = "listed" if m in available else ("NOT LISTED" if available else "unknown")
        try:
            kw = dict(model=m, messages=[{"role": "user", "content": "ابحث عن آيفون"}], max_tokens=200, tools=SCHEMAS[:1])
            if "gpt-oss" in m:
                kw["reasoning_effort"] = config.REASONING_EFFORT
            r = g._client.chat.completions.create(**kw).choices[0].message
            print(f"[{m}] {listed}: OK, tool call={'yes' if r.tool_calls else 'no'}, text={(r.content or '')[:60]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[{m}] {listed}: FAILED HTTP {getattr(e, 'status_code', '?')} {str(e)[:140]}")
    print("\nSet GROQ_MODEL=<comma separated ids> in .env to change the list.")


def traced_agent() -> SalesAgent:
    agent = SalesAgent()
    real = agent.tools.run

    def run(name, args, session):
        out = real(name, args, session)
        brief = {k: (f"[{len(v)} items]" if isinstance(v, list) else v) for k, v in out.items()}
        print(f"   ⚙ {name}({json.dumps(args, ensure_ascii=False)}) -> {json.dumps(brief, ensure_ascii=False)[:300]}")
        return out

    agent.tools.run = run
    return agent


def say(agent: SalesAgent, sess: Session, text: str) -> None:
    print(f"\n👤 {text}")
    reply = agent.run(text, sess)
    print(f"🤖 {reply}")
    for i, c in enumerate(sess.messages[-1].get("products", []), 1):
        extra = " 🎥" if c.get("video") else ""
        print(f"   [{i}] {c['name']} — {c['price']:g} EGP{extra}  {c['url']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script")
    ap.add_argument("--raw-media", type=int)
    ap.add_argument("--check-llm", action="store_true")
    ap.add_argument("--catalog-check", metavar="TEXT")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.check_llm:
        return check_llm()
    if args.catalog_check:
        return catalog_check(args.catalog_check)
    if args.raw_media:
        from tix.client import TixClient
        raw = TixClient().product(args.raw_media)
        raw = raw.get("data", raw) if isinstance(raw, dict) else raw
        print(json.dumps({k: v for k, v in raw.items() if "image" in k.lower() or "video" in k.lower() or "media" in k.lower()},
                         ensure_ascii=False, indent=2))
        return
    agent = traced_agent()
    agent.warmup(block=True)
    if args.script:
        blocks = [b for b in open(args.script, encoding="utf-8").read().split("---") if b.strip()]
        for n, block in enumerate(blocks, 1):
            print(f"\n{'=' * 20} conversation {n} {'=' * 20}")
            sess = Session()
            for line in (l.strip() for l in block.splitlines()):
                if line and not line.startswith("#"):
                    say(agent, sess, line)
        return
    sess = Session()
    print("TIX assistant (Ctrl+C to quit)")
    try:
        while True:
            text = input("\n👤 ").strip()
            if text:
                say(agent, sess, text)
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)


if __name__ == "__main__":
    main()
