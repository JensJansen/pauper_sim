# OVHcloud Setup Notes: Personal RL Self-Play Worker Node

**Purpose:** practical notes on what it takes to deploy a lightweight, CPU-only, outbound-only
background worker node on OVHcloud Public Cloud, for the personal distributed-RL project described
in `DISTRIBUTED_TRAINING_PLAN.md`. This is a deployment how-to, not another provider comparison —
see `HOSTING_RESEARCH.md` for that broader survey (OVHcloud's b3-16 at ~$75/mo was one candidate
there). All findings below are sourced from OVHcloud's own pages (pricing, docs.ovhcloud.com,
support.us.ovhcloud.com, and the official `ovh/*` GitHub repos).

**Date checked: 2026-08-30.** OVHcloud pricing changed materially on **October 1, 2026** (local
storage and public IPv4 unbundled from Gen-3 instance prices) — re-verify live prices before
committing budget, especially anything quoted per-month.

---

## 1. Smallest reasonable instance tier(s)

Below the already-known b3-16 (4 vCPU / 16 GB, ~$75/mo), OVHcloud's cheapest CPU tier is the
**Discovery range (d2)** — shared-vCPU, explicitly marketed for "test, development and sandbox
environments," which is a reasonable fit for a small self-play worker:

| Tier | vCPU | RAM | Storage | Hourly | Monthly |
|---|---|---|---|---|---|
| **d2-2** | 1 (shared) | 2 GB | 25 GB | ~$0.0129 | ~$7.08 |
| **d2-4** | 2 (shared) | 4 GB | 50 GB | ~$0.0256 | ~$14.18 |
| d2-8 | 4 (shared) | 8 GB | 50 GB | ~$0.0461 | ~$25.53 |

Source: [us.ovhcloud.com/public-cloud/prices](https://us.ovhcloud.com/public-cloud/prices/).

For comparison, the next tier up with dedicated-feeling General Purpose resources is **b3-8** (2
vCPU / 8 GB NVMe) at ~$37.54/mo on a 12-month commitment — roughly half of b3-16 — per the same
pricing page. b3-8 is the better fit if 2-3 worker processes need real (not shared/burstable) vCPU
headroom; d2-4 is the better fit if "a few vCPUs, a few GB RAM" can tolerate shared/oversubscribed
CPU, which is plausible for a batch-run, not-latency-sensitive self-play worker.

**Caveat found while researching, worth flagging directly:** the Discovery range's listed regions do
**not** include either OVHcloud US region. Per
[us.ovhcloud.com/public-cloud/regions-availability](https://us.ovhcloud.com/public-cloud/regions-availability/)
and corroborating third-party benchmarking (spare cores' scrape of OVHcloud's own availability data),
d2 instances are offered in BHS5 (Canada), several French/German/UK/Polish zones, Singapore, and
Australia — not Vint Hill, VA or Hillsboro, OR, which is where the b3/c3 US General Purpose/Compute
Optimized lines live per
[us.ovhcloud.com/public-cloud/virtual-instances](https://us.ovhcloud.com/public-cloud/virtual-instances/)
("All of our compute instances are available in Vint Hill, VA, and Hillsboro, OR"). If a North
American region matters, the nearest Discovery-range option is Canada (BHS5), or fall back to
b3-8/b3-16 in the US regions. **Unclear / to be verified:** exact current Discovery-range region
list should be reconfirmed live (availability tables change) rather than trusting this snapshot.

**Unclear / to be verified:** whether the October 2026 unbundling of local storage + public IPv4
pricing (see §1 pricing table caveat and §6) applies to the Discovery range the same way it applies
to Gen-3 B3/C3/R3. OVHcloud's own pricing-change blog post
([blog.ovhcloud.com/en/posts/public-cloud-pricing-model-update-2026](https://blog.ovhcloud.com/en/posts/public-cloud-pricing-model-update-2026/))
names "B3, C3, and R3 families" explicitly and says older ranges (B2, C2, R2) keep storage bundled,
but does not mention Discovery/d2 by name either way — treat the d2 prices above as possibly
missing a small separately-billed IPv4 line, and check the live pricing page before budgeting.

---

## 2. Instance creation / provisioning

OVHcloud Public Cloud is OpenStack-based, and offers several real, non-mutually-exclusive
provisioning paths — all realistic for a hobbyist, not just enterprise:

- **Web console ("Horizon")** — the OpenStack dashboard, reached through the OVHcloud Control
  Panel. Manual click-through flow: name the instance, pick an availability zone (default "nova" is
  fine), pick a boot source (image), pick a flavor (instance size), pick a network (public `Ext-Net`
  vs. a private network you've created), and register an SSH key ("registering an SSH key is
  absolutely necessary to be able to connect to an instance"). Source:
  [docs.ovhcloud.com .../create-instance-in-horizon](https://docs.ovhcloud.com/en/guides/public-cloud/compute/create-instance-in-horizon).
- **Official `ovhcloud` CLI** — a real, actively maintained Go binary at
  [github.com/ovh/ovhcloud-cli](https://github.com/ovh/ovhcloud-cli), installable via a one-line
  install script, Homebrew (`brew install --cask ovh/tap/ovhcloud-cli`), or `go install`. It covers
  Public Cloud, Bare Metal, VPS, networking, databases, and AI products from one tool, and is
  explicitly positioned for scripting/automation. Documented instance-creation form (region as
  positional arg):
  `ovhcloud cloud instance create BHS5 --cloud-project <PROJECT_ID> --name my-instance --flavor-selector --image-selector --network.public --ssh-key.name <SSH_KEY_NAME>`.
  Source: [github.com/ovh/ovhcloud-cli](https://github.com/ovh/ovhcloud-cli),
  [labs.ovhcloud.com/en/cli](https://labs.ovhcloud.com/en/cli/). **Unclear / to be verified:** I
  could not pull the exact subcommand syntax for stop/start/delete from the CLI's own doc page in
  this pass (the specific `doc/ovhcloud.md` page fetched only showed a VPS example); run
  `ovhcloud cloud instance --help` locally to confirm exact flags before scripting a teardown.
- **REST API directly** — the underlying OVHcloud API (`eu.api.ovh.com` / `us` equivalent), reachable
  with official thin-wrapper SDKs (`python-ovh`, `node-ovh`, `go-ovh`, `php-ovh`, etc.) that handle
  request signing. Getting credentials is a few-minute self-serve flow: generate an Application Key,
  Application Secret, and Consumer Key from a token-creation page
  (`https://us.ovhcloud.com/auth/api/createToken` for US accounts), no approval workflow. Source:
  [docs.ovhcloud.com .../api/first-steps](https://docs.ovhcloud.com/en/guides/manage-and-operate/api/first-steps).
- **Terraform (`ovh/ovh` provider)** — official provider in the Terraform Registry, backed by the
  same API/credentials above (application_key/application_secret/consumer_key), with resources for
  Public Cloud projects, instances, networks, databases, etc. There's also an alternative path via
  the generic OpenStack Terraform provider, since Public Cloud is OpenStack underneath. Good fit if
  you want "several nodes together" to be declarative. Source:
  [docs.ovhcloud.com .../how-to-use-terraform](https://docs.ovhcloud.com/en/guides/public-cloud/cross-functional/how-to-use-terraform),
  [registry.terraform.io/providers/ovh/ovh](https://registry.terraform.io/providers/ovh/ovh/latest/docs).
- **OpenStack CLI directly** — since Public Cloud instances are OpenStack Nova instances under the
  hood, the standard `openstack` CLI also works once pointed at OVHcloud's OpenStack-compatible
  endpoint/credentials, e.g. `openstack server create --flavor b2-7 --image "Ubuntu 24.04" --key-name my-key --network Ext-Net my-instance`
  (per search-indexed OVHcloud support docs on Horizon/OpenStack instance creation).

**Bottom line for this project:** scripted, automatable provisioning is realistic for a hobbyist —
either the official `ovhcloud` CLI or the Terraform provider are both good fits for "bring up N
nodes together, tear them down together," and account setup for API access is a self-serve,
minutes-long flow, not an enterprise onboarding process.

---

## 3. Default networking behavior

- **Public IP by default depends on the creation mode, not a single universal default.** In what
  OVHcloud's own networking-concepts doc calls "Public Mode," "a public IP is attached directly to
  the instance" automatically — this is the default when you attach the instance to the public
  `Ext-Net` network (the default network shown in Horizon/CLI examples above). In "Private Mode" (no
  public network attached), "a newly created instance does not have a public IP attached to any
  port... the instance will remain fully private (no public connectivity)" unless you add a Floating
  IP, Load Balancer, or Gateway. Source:
  [docs.ovhcloud.com .../network-services/concepts](https://docs.ovhcloud.com/en/guides/public-cloud/network-services/concepts).
- **Outbound-only workloads don't need a public IP at all.** Because this node only ever *initiates*
  outbound HTTPS (to the coordinator) and never needs to be reached from outside, the
  privacy-preferred setup is: create the instance on a **private network only**, then add an
  **SNAT Gateway**. The Gateway "translates the private IP address of your resource into a public
  one" for outbound reachability while the instance itself "remains fully private... no kind of
  access initiated from the outside network is possible." This is a materially smaller attack
  surface than the default public-IP path, at the cost of one more resource to provision. Source:
  same concepts page, and
  [ovhcloud.com/en/public-cloud/gateway](https://www.ovhcloud.com/en/public-cloud/gateway/).
- **Gateway billing:** pay-as-you-go, billed hourly with a monthly invoice; five bandwidth tiers
  (200 Mbps up to 8 Gbps) exist, cheapest tier obviously sufficient for a 10-30-min sync cadence.
  The bundled public IPv4 that comes with a Gateway is billed at the same standalone-IPv4 rate noted
  below and "cannot be moved from one gateway to another." **Unclear / to be verified:** I could not
  pull an exact headline hourly/monthly price for the cheapest (200 Mbps) Gateway tier from
  OVHcloud's own pricing pages in this pass — the product page describes the billing model but not
  a rate table, and the machine-readable price catalog (`pricelist.ovh/public-cloud.html`) was
  unreachable during this research session (connection refused). Check the Gateway pricing tab in
  the Control Panel or `pricelist.ovh` directly before committing to this path over a plain public
  IP.
- **Plain public-IP cost, as a baseline comparison:** as of the October 1, 2026 pricing change, a
  standalone public IPv4 address on a Gen-3 (b3/c3/r3) instance is billed separately at
  **€0.0027/hour (excl. VAT), ~€1.97/month**. Source:
  [blog.ovhcloud.com/en/posts/public-cloud-pricing-update-october-2026](https://blog.ovhcloud.com/en/posts/public-cloud-pricing-update-october-2026/).
  Given this workload needs zero inbound exposure, the private-network + Gateway path is arguably
  the *more correct* choice on security grounds even if it isn't obviously cheaper than a plain
  public IP — but for a single lightweight node, the plain "Public Mode" default is simplest to
  stand up and is not unreasonable (outbound access is unrestricted by default either way; nothing
  extra is needed to make outbound HTTPS "just work" once a public IP or Gateway exists).
- **Bandwidth:** OVHcloud's own network product page states Public Cloud bandwidth is "unlimited and
  unmetered... transfer any amount of data at no extra cost" for ingress/egress (S3 object storage
  egress excluded) — so the periodic 10-30-min coordinator syncs incur no metered-bandwidth cost
  regardless of which networking mode is chosen. Source:
  [us.ovhcloud.com/public-cloud/network](https://us.ovhcloud.com/public-cloud/network/).
- **Anti-DDoS protection is included on all instances at no extra cost**, per the same network
  product page — a minor plus for the plain-public-IP path.

---

## 4. OS image options

- OVHcloud's Public Image Catalog offers standard Linux distributions including **Ubuntu, Debian,
  AlmaLinux, Rocky Linux, CentOS, and Fedora**, selected by "Distribution Type" and version at
  instance-creation time; available versions can vary by flavor/region. Source:
  [us.ovhcloud.com/public-cloud/public-image-catalog](https://us.ovhcloud.com/public-cloud/public-image-catalog/),
  corroborated by
  [docs.ovhcloud.com .../compute/key-concepts](https://docs.ovhcloud.com/en/guides/public-cloud/compute/key-concepts)
  ("Cloud images provided by OVHcloud include cloud-init scripts that allow you to customize
  instances when you launch them" — useful for scripting the venv/`git clone`/`pip install` setup
  as a cloud-init user-data script at boot). A standard Ubuntu LTS image is a plain, well-supported
  choice for the `git clone` → venv → `pip install -r requirements.txt` workflow this node needs.
  **Unclear / to be verified:** I could not pull the exact current list of specific Ubuntu LTS point
  releases (e.g. 22.04 vs 24.04) offered right now from OVHcloud's own catalog page in this research
  pass — confirm the live list in the Control Panel's image picker or via
  `ovhcloud cloud instance create --image-selector` at creation time.
- **A pre-built ML/PyTorch image does exist, but it belongs to a different, higher-level OVHcloud
  product family (AI Notebooks / AI Training), not to plain Public Cloud compute instances.**
  OVHcloud's **AI Notebooks** offering has preset images including "PyTorch... including JupyterLab
  notebook, Visual Studio Code IDE, and pytorch libraries," alongside similar presets for TensorFlow
  2, Hugging Face, MXNet, Fast.ai, and AutoGluon; **AI Training** similarly runs pre-built
  PyTorch/TensorFlow/HuggingFace Docker images as submitted jobs via the `ovhai` CLI, API, or
  Control Panel. Sources:
  [us.ovhcloud.com/public-cloud/ai-notebooks](https://us.ovhcloud.com/public-cloud/ai-notebooks/),
  [support.us.ovhcloud.com AI Training - Start a job with a notebook Docker image](https://support.us.ovhcloud.com/hc/en-us/articles/35370369024147-AI-Training-Start-a-job-with-a-notebook-Docker-image).
  As anticipated in the research brief, this is almost certainly **not** the right fit here: those
  products are built around a managed Jupyter/job-submission model (and typically GPU-oriented
  pricing/quotas), not a long-running background process doing its own `git pull`/venv management on
  a plain always-addressable VM. A bare Ubuntu image on a normal Public Cloud instance is the
  simpler, more direct match for this workload's stated setup (plain venv + pip, CPU-only PyTorch
  wheel).

---

## 5. Start/stop/destroy mechanics and billing

OVHcloud Public Cloud instances have three distinct "not fully running" states, each with a
**different billing outcome** — this distinction matters a lot for a batch-run node:

| Action | What happens | Billing while in this state | Source |
|---|---|---|---|
| **Turn off / Stop** | VM state stored on disk; compute resources remain reserved for the instance | **Billed at the same rate as a running instance** — stopping does *not* save money | [docs.ovhcloud.com .../suspend-or-pause-an-instance](https://docs.ovhcloud.com/en/guides/public-cloud/compute/suspend-or-pause-an-instance) |
| **Pause** | VM state stored in RAM, instance frozen (Horizon/OpenStack-Nova API only, not exposed everywhere) | **Billed at the same rate as running** | same source |
| **Shelve** | Resources released; disk data preserved via an automatically-created snapshot | **Only billed for the snapshot storage** — the cost-effective option for "off but not gone" | same source |
| **Delete/destroy** | Instance and its local storage are removed | No further compute billing; any *separately provisioned* resources (a detached Volume, a reserved public IP, a Gateway) can keep billing independently until they too are deleted | inferred from the same doc plus the general storage/IP unbundling described in §1/§3 — see caveat below |

Practical implication for "spin up and tear down as a batch": plain **Stop** does **not** avoid
cost — for this workload's "not necessarily always-on" pattern, either **Shelve** between batches
(cheap, keeps the IP and a disk snapshot, per the same doc: "you can shelve, pause or suspend a
Public Cloud instance to temporarily free resources while keeping your IP address") or fully
**delete and recreate** from a cloud-init/Terraform/CLI script each batch are the two options that
actually save money; a "stopped" instance sitting idle is just as expensive as a running one.

Scriptability: start/stop/shelve/delete are all standard OpenStack Nova operations and are exposed
through every provisioning path in §2 (Horizon buttons, `ovhcloud` CLI, raw API, Terraform
apply/destroy, `openstack` CLI) — bringing up or tearing down several nodes together is realistic
with a short script or a Terraform module that loops over node definitions.

**Unclear / to be verified:** I found no OVHcloud-authored page in this pass that explicitly states
whether **local NVMe storage** billing (the storage line unbundled from Gen-3 instance price as of
Oct 2026, ~€0.11/GB-month per §1) is destroyed and stops billing immediately on instance *deletion*,
versus persisting on some separate schedule — treat "delete the instance" as the intended terminal
action and confirm zero further line items appear on the next invoice cycle after a real test
delete, rather than assuming it from documentation alone.

---

## 6. Account / billing practicalities

- **Account types:** OVHcloud explicitly distinguishes an **"Individual" account** ("natural persons
  purchasing products for private needs outside the scope of professional activity") from a
  **"Company" account** (legal persons/sole traders making professional purchases) and a
  **"Non-Profit" account**. A personal hobbyist project is squarely the "Individual" account type —
  no business registration is required. Source: OVHcloud account-creation documentation (per
  [help.ovhcloud.com Creating an OVHcloud account](https://help.ovhcloud.com/csm/en-account-create-ovhcloud-account?id=kb_article_view&sysparm_article=KB0043022)).
- **Payment method is required before any resource can be created.** A brand-new Public Cloud
  project without a saved payment method is placed in **"discovery mode"**: you can browse the
  console but cannot create instances. Adding a payment method and clicking "Activate project" lifts
  this restriction. Source:
  [docs.ovhcloud.com .../create-a-public-cloud-project](https://docs.ovhcloud.com/en/guides/public-cloud/cross-functional/create-a-public-cloud-project).
- **No hard minimum spend**, but there is a one-time **$0.99 anti-fraud verification charge** on the
  first project added to an account, which is credited back against the first bill. New eligible
  accounts also get a **free trial credit** on project activation (documentation for the US pricing
  page separately advertises **$200 USD in free Public Cloud credit** for a first project, plus
  **$100** toward Managed Database; a UK-region doc page instead cites "£175" — these are likely
  region/currency-specific promotions rather than the same figure, so **check the live signup flow
  for the current US offer** rather than trusting either number as fixed). Sources:
  [docs.ovhcloud.com .../create-a-public-cloud-project](https://docs.ovhcloud.com/en/guides/public-cloud/cross-functional/create-a-public-cloud-project),
  [us.ovhcloud.com/public-cloud/network](https://us.ovhcloud.com/public-cloud/network/).
- **Billing granularity:** usage is billed **per second** (beyond an initial short minimum), not
  rounded up to the hour — prices are *displayed* as hourly/monthly for readability, but actual
  metering is fine-grained, and creating/deleting a resource mid-month bills only for hours
  actually used. Source: OVHcloud billing documentation content indexed at
  [docs.ovhcloud.com .../analyze-billing](https://docs.ovhcloud.com/en/guides/public-cloud/cross-functional/analyze-billing)
  and [support.us.ovhcloud.com Public Cloud billing](https://support.us.ovhcloud.com/hc/en-us/articles/22179651508115-Public-Cloud-billing)
  (this specific support-guide URL returned HTTP 403 when fetched directly in this session — content
  above is corroborated via search-indexed excerpts of it, not a direct fetch; worth opening
  manually while logged in to confirm).
- **Region selection:** chosen per-resource at creation time (Horizon's region switcher, or the
  region argument in the CLI/API/Terraform), not fixed at account or project creation. For this
  project, note the region-availability mismatch flagged in §1: US regions (Vint Hill, VA /
  Hillsboro, OR) carry the b3/c3 General Purpose/Compute Optimized lines but **not** the cheaper
  Discovery range, which is only offered in Canada/Europe/Singapore/Australia per
  [us.ovhcloud.com/public-cloud/regions-availability](https://us.ovhcloud.com/public-cloud/regions-availability/).

---

## Practical steps to deploy the first node

1. **Create/confirm an OVHcloud account** as an **Individual** account type (no business
   registration needed for a personal project).
2. **Create a Public Cloud project** in the Control Panel, add a payment method (required to exit
   "discovery mode" — no instances can be created without this), and check whether the current sign-up
   flow still offers a free trial credit (was ~$200 for a first US project per the pricing/network
   page, but confirm live — figures found varied by region/currency during this research pass).
3. **Choose the instance tier and region together**, since they're linked: if minimizing cost is the
   priority and a non-US region (Canada/EU/Singapore/Australia) is acceptable, use **Discovery d2-4**
   (2 shared vCPU / 4 GB, ~$14/mo) or **d2-8** (4 shared vCPU / 8 GB, ~$25/mo) for 2-4 self-play
   workers. If a US region matters, the cheapest fit is **b3-8** (2 vCPU / 8 GB NVMe, ~$37.54/mo,
   12-month rate) in Vint Hill, VA or Hillsboro, OR. Re-check both regions' live availability and
   prices before committing, per the §1 caveats.
4. **Generate API credentials** at the token-creation page (`us.ovhcloud.com/auth/api/createToken`
   for US accounts) if scripting via API/CLI/Terraform — a self-serve, few-minutes step producing an
   Application Key, Application Secret, and Consumer Key.
5. **Pick a provisioning method** matching how many nodes you're standing up at once: the
   `ovhcloud` CLI or the `ovh/ovh` Terraform provider for scripted, repeatable multi-node
   creation/teardown; Horizon web console is fine for a single manual first test.
6. **Decide on networking before creating the instance**, since this workload never needs inbound
   access: either (a) accept the simpler default — attach to the public `Ext-Net` network, get a
   public IP automatically (Public Mode) — or (b) the more locked-down option — create on a private
   network only and add an SNAT Gateway for outbound-only reachability. Confirm outbound HTTPS to
   the coordinator actually works after boot (`curl` a known HTTPS endpoint) before trusting either
   path; no extra firewall/security-group change should be needed for outbound in either mode.
7. **Pick a plain Ubuntu LTS image** from the Public Image Catalog at creation time (skip
   OVHcloud's AI Notebooks/AI Training PyTorch presets — those are a different, managed-Jupyter/job
   product, not a fit for a self-managed `git clone` + venv + `pip install` background process).
   Optionally supply a cloud-init user-data script to automate the clone/venv/pip-install/start
   sequence at first boot.
8. **Register/attach an SSH key** at creation time — required to reach the instance at all per
   OVHcloud's own Horizon-creation doc.
9. **Verify the CPU-only PyTorch install and worker processes** come up as expected, then confirm
   the coordinator sees the node's periodic sync check-ins.
10. **Plan teardown correctly**: for a genuine pause between training batches, use **Shelve** (billed
    only for a disk snapshot, keeps the IP) rather than **Stop/Turn off** (which keeps billing at the
    full running rate — a common trap). For a real end-of-batch teardown across several nodes, script
    a full **delete** (via CLI/Terraform destroy) and recreate from the same image/cloud-init script
    next time, confirming on the next invoice that no orphaned volume/IP/Gateway charges remain.
