# Tailscale setup for distributed training nodes

Written 2026-10-17 for review issue 10 (nodes had no way in). Syntax below was
checked against Tailscale's docs the same day (policy-syntax, auth-keys,
tailscale-ssh), not written from memory. **Not yet applied to the real tailnet
or run against a real node** -- verify with the checks at the bottom on the
first Phase 3 node.

## What this sets up

- Nodes join your tailnet tagged `tag:pauper-node` (tag comes from the auth key).
- Your own devices (the hub PC and anything else of yours) can reach everything,
  including SSH into nodes.
- Nodes can reach **only the hub's training port** (100.113.155.84:5000). Not
  each other, not any other device of yours. Your tailnet was on the default
  allow-everything policy, under which any node could reach every device of
  yours -- replacing it is the point of step 1.
- Nodes still need ordinary outbound internet (apt, pip, git, Tailscale's
  control plane); that goes through the OVH gateway, not the tailnet, and is
  unaffected by this policy.

## 1. Replace the tailnet policy

Tailscale admin console -> Access controls. **This replaces the default
allow-everything rule**: anything not listed is denied. Keep a copy of the old
policy text before saving.

```json
{
  // Who may attach the tag. autogroup:admin = you (an Admin of the tailnet).
  "tagOwners": {
    "tag:pauper-node": ["autogroup:admin"]
  },

  "acls": [
    // Your own devices can reach everything, including the nodes.
    {"action": "accept", "src": ["autogroup:member"], "dst": ["*:*"]},

    // Nodes may reach ONLY the hub's training server. Nothing else.
    // (100.113.155.84 = the hub PC's tailnet address; update if it ever changes.)
    {"action": "accept", "src": ["tag:pauper-node"], "dst": ["100.113.155.84:5000"]}
  ],

  // Tailscale SSH needs BOTH an acls rule (network path; covered above) and an
  // ssh rule. Members may SSH into nodes as root. Docs suggest "check" instead of
  // "accept" for root (re-auth every 12h); "accept" is fine for a solo owner
  // with disposable nodes -- switch to "check" if you ever share the tailnet.
  "ssh": [
    {
      "action": "accept",
      "src": ["autogroup:member"],
      "dst": ["tag:pauper-node"],
      "users": ["root", "autogroup:nonroot"]
    }
  ]
}
```

The editor has a preview/test view: after saving, confirm that a node-tagged
source can reach `100.113.155.84:5000` and nothing else.

## 2. Create the node auth key

Admin console -> Settings -> Keys -> Generate auth key:

| Option | Value | Why |
|---|---|---|
| Reusable | **on** | one key is baked into the provisioning of every node |
| Ephemeral | **on** | a destroyed/recreated node drops off the tailnet by itself instead of leaving a stale machine entry |
| Pre-approved | on | harmless; only matters if device approval is enabled |
| Tags | **`tag:pauper-node`** | the node takes the tag from the key (no `--advertise-tags` needed) |
| Expiry | up to 90 days (the maximum) | the key is only needed when an instance is *created*; running nodes are unaffected when it expires |

Pass it to Terraform as `tailscale_auth_key` (kept out of git, same as the other
secrets; `*.tfvars` is gitignored). When it expires, generate a new one before
creating more nodes.

The bootstrap (`node_bootstrap.sh.tftpl`) runs `tailscale up --authkey=... --ssh`.
It also makes the systemd journal persistent so a node's service log survives
a reboot.

## 3. Getting into a node

From any of your own tailnet devices (the node's name is on the Machines page,
or `tailscale status`):

```
ssh root@<node-tailnet-name>
```

What to look at:

| Question | Command |
|---|---|
| Is the training service running / why did it die? | `systemctl status pauper-sim-node` and `journalctl -u pauper-sim-node -e` |
| Follow it live | `journalctl -u pauper-sim-node -f` |
| Did the first-boot bootstrap finish? | `tail -n 100 /var/log/cloud-init-output.log` and `cloud-init status` |
| Is the node on the tailnet / can it see the hub? | `tailscale status`, `curl -s -o /dev/null -w '%{http_code}' http://100.113.155.84:5000/admin/status` (401 or 404 = reachable) |
| RAM / CPU / disk | `free -h`, `top`, `df -h` |
| Node ID / which deck | `cat /var/lib/pauper-sim/node_id`; the hub's `python -m distributed.admin ... status` shows the deck |
| Restart it | `systemctl restart pauper-sim-node` -- but this is the SIGTERM path: the node finishes its in-flight call first (about 5-10 min), so it can block up to `TimeoutStopSec=1800`. For a planned stop use `python -m distributed.admin stop --deck <deck>` from the hub instead (lossless, no wait on systemd). |

## Caveats

- **If Tailscale itself fails to come up there is still no way in** (no public
  IP, by design -- decision #4). `cloud-init-output.log` is the only record;
  check whether OVH's console can show it for private-only instances (not
  verified). The fallback is destroy and recreate.
- Ephemeral nodes are removed from the tailnet after they go offline for a
  while (Tailscale's docs don't say how long). A node that is merely rebooted
  comes back normally; one that stays down long enough has to be recreated.
- The nodes hold the shared registration credential in
  `/opt/pauper-sim/distributed_configs/node.local.json` (mode 600), so root on a
  node can read it. Fine for your own nodes; it is one more reason third-party
  nodes would need their own credentials before joining this tailnet.
- Unverified on a real node: that `--ssh` works on Ubuntu 24.04 under cloud-init
  (Tailscale's docs only say Linux generally), and that the tag really arrives
  from the key alone -- `tailscale status` / the Machines page should show
  `tag:pauper-node` on the first node. If it doesn't, add
  `--advertise-tags=tag:pauper-node` to the `tailscale up` line.
