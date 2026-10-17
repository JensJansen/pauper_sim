"""Operator CLI for a running hub (talks to hub_server's /admin/* endpoints).

    python -m distributed.admin --hub-url http://100.x.y.z:5000 \\
        --admin-credential-file ../distributed_configs/hub_admin.local status
    ... stop --deck elves        # graceful stop of one deck's node
    ... stop --all               # graceful stop of the whole fleet
    ... resume --deck elves      # cancel a stop request not yet acted on (or --all)

A stopped node finishes its in-flight call, does its final push, releases its
deck back to the hub and exits (see node_agent.run). The admin token is a
secret DIFFERENT from the registration credential; generate one with
`python -m distributed.auth` and pass the same file to the hub's
--admin-credential-file.
"""
import argparse

import requests


def call(hub_url, token, method, path, body=None):
    resp = requests.request(method, f"{hub_url}{path}", json=body, headers={"X-Admin-Token": token}, timeout=30)
    try:
        data = resp.json()
    except ValueError:
        data = {"error": resp.text}
    if resp.status_code >= 400:
        raise SystemExit(f"{resp.status_code}: {data.get('error')}")
    return data


def format_status(rows):
    header = f"{'deck':<20} {'state':<10} {'games':>8} {'last sync':>10} {'signal':<9} node"
    lines = [header]
    for r in rows:
        age = "-" if r["last_sync_age_s"] is None else f"{r['last_sync_age_s']}s"
        lines.append(f"{r['deck']:<20} {r['state']:<10} {r['cumulative_games']:>8} {age:>10} "
                     f"{r['signal'] or '-':<9} {r['node_id'] or '-'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hub-url", required=True)
    ap.add_argument("--admin-credential-file", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("stop", "resume"):
        p = sub.add_parser(name)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--deck")
        g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    with open(args.admin_credential_file) as f:
        token = f.read().strip()

    if args.cmd == "status":
        body = call(args.hub_url, token, "GET", "/admin/status")
        print(format_status(body["decks"]))
        for r in body.get("version_rejections", []):
            print(f"VERSION REJECTED {r['age_s']}s ago: node {r['node_id']} sha {r['sha']}: {r['reason']}")
    else:
        body = {"all": True} if args.all else {"deck": args.deck}
        print(call(args.hub_url, token, "POST", f"/admin/{args.cmd}", body))


if __name__ == "__main__":
    main()
