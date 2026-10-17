# Hosting Research: Distributed RL Training Nodes

**Purpose:** This document evaluates hosting options for WAN-distributed reinforcement-learning
training nodes for this project. It feeds the "hosting platform: not yet chosen" open decision in
`DISTRIBUTED_TRAINING_PLAN.md`.

**Date checked: 2026-09-01.** Cloud pricing changes frequently (Hetzner alone raised prices twice
in 2026). Treat every number below as approximate and re-verify on the linked official page before
committing to a provider — a live pricing calculator or console is the only authoritative source.

**Workload profile driving this research:** a CPU-bound Python/PyTorch self-play process (small
set-transformer nets, d_model=64, low millions of parameters), a few GB RAM, periodic PPO updates,
outbound HTTPS to a coordinator with a sync interval of 10-30 minutes, run in batches rather than
strictly 24/7. GPU is a "nice to have later," not a current requirement, and if used would be at
the cheap/entry tier, not flagship (A100/H100-class) cards.

---

## 1. CPU-only hosting options

Table below targets a **4-8 vCPU / 8-16 GB RAM** instance as the reference point for each provider.

| Provider | Instance type/tier | Approx price | Billing model | Caveats | Source URL |
|---|---|---|---|---|---|
| **Hetzner** | CPX31 (4 vCPU / 8 GB, shared AMD, US region) | ~$25.59/mo (~$0.035/hr) | Hourly (capped at monthly) or monthly | Prices jumped sharply in 2026 (April + June increases, some tiers +100-200%); EU-region (EUR) pricing differs from US (USD); generous included traffic (typically 20TB+), cheap overage. Pricing page is JS-rendered — figures reconstructed from Hetzner's own price-adjustment notice and third-party trackers; reconfirm live. | [hetzner.com/cloud/regular-performance](https://www.hetzner.com/cloud/regular-performance/), price-change details: [docs.hetzner.com price-adjustment](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/) |
| **Hetzner** | CPX41 (8 vCPU / 16 GB, shared AMD, US region) | ~$47.09/mo | Hourly or monthly | Same caveats as above. Dedicated-vCPU CCX23 (4 vCPU/16GB) ~$40.59/mo and CCX33 (8 vCPU/32GB) ~$77.59/mo are the "always-busy CPU" alternative tier. | [hetzner.com/cloud/general-purpose](https://www.hetzner.com/cloud/general-purpose/) |
| **DigitalOcean** | Basic Droplet, 4 vCPU / 8 GB | $48.00/mo ($0.07143/hr) | Hourly (per-second billing, 60s min) or monthly | 5,000 GiB/month bandwidth included on this tier; per-second billing means stop/start batching is cheap and clean. | [digitalocean.com/pricing/droplets](https://www.digitalocean.com/pricing/droplets) |
| **DigitalOcean** | Basic Droplet, 8 vCPU / 16 GB | $96.00/mo ($0.14286/hr) | Hourly or monthly | 6,000 GiB/month bandwidth included. General Purpose (balanced) line costs more for the same vCPU count at higher RAM ratios. | [digitalocean.com/pricing/droplets](https://www.digitalocean.com/pricing/droplets) |
| **Vultr** | Cloud Compute High Performance, 4 vCPU / 8 GB | $48.00/mo | Hourly or monthly | 6 TB bandwidth included, NVMe storage. "Regular Performance" (older/cheaper CPU) line is cheaper still but slower per-core — matters for a branchy Python sim. | [vultr.com/pricing](https://www.vultr.com/pricing/) |
| **Vultr** | Cloud Compute High Performance, 8 vCPU / 16 GB | $96.00/mo | Hourly or monthly | 8 TB bandwidth included. | [vultr.com/pricing](https://www.vultr.com/pricing/) |
| **Linode (Akamai)** | Dedicated CPU, ~4 vCPU / 8 GB | ~$72/mo (est., scales from 2vCPU/4GB @ $36/mo) | Hourly or monthly | Akamai's pricing page is JS-rendered and could not be scraped directly for this report; figure is extrapolated from Linode's historical linear Dedicated-CPU scaling and should be reconfirmed live. Dedicated CPU tier guarantees full cores (good fit for a CPU-bound sim) vs. cheaper Shared tier. `linode.com/pricing` now redirects to Akamai's site. | [akamai.com/cloud/pricing](https://www.akamai.com/cloud/pricing) |
| **Linode (Akamai)** | Dedicated CPU, ~8 vCPU / 16 GB | ~$144/mo (est.) | Hourly or monthly | Same caveat — reconfirm live before committing budget. | [akamai.com/cloud/pricing](https://www.akamai.com/cloud/pricing) |
| **Oracle Cloud** | **Always Free** Ampere A1 (ARM) | **$0** | N/A (free tier) | As of the June 2026 change, new **Always Free** tenancies get 2 OCPUs / 12 GB RAM total (shared across all A1 instances in the account); pre-existing/PAYG accounts may retain the older 4 OCPU / 24 GB allowance. Oracle made this change to the free-tier docs without a public announcement — treat the free allotment as unstable and re-check before relying on it. ARM (aarch64) requires your PyTorch/deps to have ARM wheels. 10 TB/month egress included even on free tier. | [docs.oracle.com Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), [oracle.com/cloud/free](https://www.oracle.com/cloud/free/) |
| **Oracle Cloud** | Paid Ampere A1, ~4 OCPU / 16 GB (beyond free allotment) | ~$29/mo (4 OCPU × $0.01/OCPU-hr × 730 + 16GB × $0.0015/GB-hr × 730 ≈ $29.2+$17.5) | Hourly (metered) | Per-OCPU-hour ($0.01) and per-GB-hour ($0.0015) metering, billed only for what's provisioned above the free tier; very cheap per-core if you outgrow Always Free. | [oracle.com/cloud/compute/pricing](https://www.oracle.com/cloud/compute/pricing/) |
| **OVHcloud** | Public Cloud b3-16 (4 vCPU / 16 GB, Gen3 NVMe) | $74.96/mo ($0.1208/hr) | Hourly or monthly (12-mo savings plan: 15% off) | From Oct 1, 2026, local storage and IPv4 are billed separately from b3/c3 instances (~$2.33/mo each). This was the one official pricing page that rendered cleanly for this report. | [us.ovhcloud.com/public-cloud/prices](https://us.ovhcloud.com/public-cloud/prices/) |
| **OVHcloud** | Public Cloud b3-32 (8 vCPU / 32 GB) | $149.85/mo ($0.2415/hr) | Hourly or monthly | Same caveats as b3-16. C3 "Compute Optimized" line (e.g. c3-16: 8 vCPU/16GB @ $133.66/mo) trades RAM for more CPU-per-dollar if RAM headroom isn't needed. | [us.ovhcloud.com/public-cloud/prices](https://us.ovhcloud.com/public-cloud/prices/) |
| **Scaleway** | GP1-XS (4 vCPU / 16 GB) | ~€65.52/mo (~€0.091/hr) | Hourly (with savings plans up to 25% off) or monthly | Listed prices include egress and IPv6; storage and public IPv4 cost extra. EU-based (French/Dutch/Polish DCs) — expect ~100-150ms+ latency from US. COMPUTE3 "Compute Optimized" line is cheaper per-core if RAM isn't needed. | [scaleway.com/en/pricing/virtual-instances](https://www.scaleway.com/en/pricing/virtual-instances/) |
| **AWS EC2** *(reference)* | m6i.xlarge (4 vCPU / 16 GB) | ~$0.192/hr (~$140/mo on-demand, us-east-1) | Hourly (on-demand), or Spot/Savings Plans for less | Data egress billed separately (~$0.09/GB after free tier, first 100GB/mo free account-wide) — matters for a coordinator-sync workload. Pricing page is an interactive JS calculator; figure corroborated via AWS's own published on-demand rate as mirrored by third-party pricing trackers. | [aws.amazon.com/ec2/pricing/on-demand](https://aws.amazon.com/ec2/pricing/on-demand/) |
| **GCP Compute Engine** *(reference)* | e2-standard-4 (4 vCPU / 16 GB) | ~$0.134/hr (~$98/mo on-demand, us-central1) | Hourly (on-demand), sustained-use discounts apply automatically | n2-standard-4 (same specs, newer CPU) runs ~$0.194/hr (~$142/mo) — pick e2 unless per-core speed matters. Network egress ~$0.12/GB after free tier. Pricing page is JS-rendered; figures corroborated via GCP's own published list price as mirrored by third-party pricing trackers. | [cloud.google.com/compute/all-pricing](https://cloud.google.com/compute/all-pricing) |

**Notable omission worth flagging:** none of the budget providers above publish a *guaranteed*
hourly-billed "start/stop freely, pay only for on hours" model as cleanly as DigitalOcean and
Vultr do (true per-second billing with a monthly cap) — those two are the best fit if the plan is
genuinely "batch start/stop," while Hetzner/OVH/Scaleway are more monthly-commitment-shaped even
though they technically also offer hourly rates.

---

## 2. GPU hosting options

Table targets the cheapest/entry-level GPU tier at each provider (not flagship A100/H100-class
cards), consistent with the stated need: small nets that would run fine on a low-end GPU if one is
used later.

| Provider | GPU tier | Approx price | Billing model | Caveats | Source URL |
|---|---|---|---|---|---|
| **Vast.ai** | RTX 3060 (12 GB) | **$0.02/hr** | Per-second, marketplace (peer-hosted) | Cheapest option found anywhere in this survey. Marketplace = individual hosts, not a data center SLA — reliability, host trust score, and interruption risk vary per listing; "interruptible" listings are cheaper but can be reclaimed by the host at will. Storage/bandwidth sometimes billed by the specific host, not uniformly. No real minimum session length. | [vast.ai/pricing/gpu/RTX-3060](https://vast.ai/pricing/gpu/RTX-3060), live rates: [vast.ai/pricing](https://vast.ai/pricing) |
| **Vast.ai** | RTX 3060 Ti | ~$0.03/hr | Per-second, marketplace | Same marketplace caveats as above. | [vast.ai/pricing](https://vast.ai/pricing) |
| **RunPod** | RTX A5000 (24 GB), Community Cloud | $0.16/hr | Per-second | "Community Cloud" = third-party-hosted nodes (cheaper, less guaranteed) vs. "Secure Cloud" (RunPod's own DCs, pricier: same card $0.27/hr). Network volume storage billed separately (~$0.05/GB-mo over 1TB). No published minimum session. | [runpod.io/pricing](https://www.runpod.io/pricing) |
| **RunPod** | RTX 3090 (24 GB), Community Cloud | $0.22/hr | Per-second | Secure Cloud equivalent: $0.50/hr. | [runpod.io/pricing](https://www.runpod.io/pricing) |
| **TensorDock** | RTX 4090 (24 GB) | from $0.35/hr | Per-second, marketplace (hosts set own price) | Marketplace model like Vast.ai — multiple listings/prices for the same card depending on host; reliability varies by host. $5 minimum deposit to start. | [tensordock.com](https://www.tensordock.com/) |
| **Lambda Labs (lambda.ai)** | Quadro RTX 6000 (24 GB), 1x GPU on-demand | $0.69/hr | Pay-by-the-minute, on-demand (data-center hosted, not marketplace) | No egress fees (explicitly advertised), no minimum commitment for on-demand. Not a marketplace — Lambda's own infrastructure, so more predictable than Vast.ai/TensorDock. Capacity for the cheapest tier can sell out; no interruptible/spot discount tier advertised for the cheap cards. | [lambda.ai/service/gpu-cloud](https://lambda.ai/service/gpu-cloud) |
| **Lambda Labs (lambda.ai)** | A10 (24 GB) | ~$0.60/hr | Pay-by-the-minute | Same no-egress-fee, no-minimum terms as above. | [lambda.ai/service/gpu-cloud](https://lambda.ai/service/gpu-cloud) |
| **DigitalOcean / Paperspace** | M4000 (8 GB) | $0.45/hr | Hourly, on-demand | Paperspace was absorbed into DigitalOcean; `paperspace.com/pricing` now redirects into DO's GPU droplet pricing. Free-tier GPU instances exist but auto-shutdown after 12 hours. Storage overage billed at $0.29/GB beyond plan allowance. | [digitalocean.com/products/gpu-droplets](https://www.digitalocean.com/products/gpu-droplets) |
| **AWS EC2** *(reference)* | g4dn.xlarge — 1x NVIDIA T4 (16 GB) + 4 vCPU/16GB host | ~$0.526/hr on-demand (us-east-1) | Hourly on-demand, or Spot (typically 50-70% off, interruptible with 2-min warning) | Spot pricing is meaningfully cheaper but AWS can reclaim the instance with only ~2 minutes' notice — needs checkpoint/resume logic to be safe. EBS storage and data egress billed separately from compute. | [aws.amazon.com/ec2/pricing/on-demand](https://aws.amazon.com/ec2/pricing/on-demand/), instance specs: [aws.amazon.com/ec2/instance-types/g4](https://aws.amazon.com/ec2/instance-types/g4/) |
| **GCP Compute Engine** *(reference)* | 1x NVIDIA T4 (16 GB), attached to n1-standard host | ~$0.35/hr for the GPU alone (~$0.73/hr bundled with an 8 vCPU/30GB n1 host) | Hourly on-demand, or Spot/Preemptible (~60-91% off, can be reclaimed anytime, max 24h runtime) | GPU and host VM billed separately in GCP's model — cheaper to pair the T4 with a small custom VM shape than the default n1-standard-8 bundle shown by aggregators. Preemptible/Spot GPUs are the cheapest path but unsuitable without frequent checkpointing given the training loop's stated 10-30 min sync cadence. | [cloud.google.com/compute/gpus-pricing](https://cloud.google.com/compute/gpus-pricing) |
| **Azure** *(reference)* | NC4as T4 v3 — 1x NVIDIA T4 (16 GB) + 4 vCPU/28GB host | ~$0.526/hr pay-as-you-go (East US) | Hourly on-demand, or Spot (variable discount, reclaimable) | Microsoft has signaled the classic NC-series (non-T4) is being retired/decommissioned; NCas_T4_v3 is Azure's budget GPU line specifically. Storage and egress billed separately. | [azure.microsoft.com/pricing/details/virtual-machines/linux](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/), calculator: [azure.microsoft.com/pricing/calculator](https://azure.microsoft.com/en-us/pricing/calculator/) |
| **CoreWeave** *(reference, enterprise-leaning)* | L40S | reported ~$18/hr (or ~$1-2.25/GPU-hr on 8-GPU nodes, pricing page is interactive and did not resolve cleanly for this report) | Hourly on-demand, or committed-use for up to 60% off; Spot available | CoreWeave targets serious/enterprise AI infra, not hobbyist single-GPU rental — its pricing page bundles GPUs into multi-GPU node configurations rather than exposing a clean single-card hourly rate the way Vast.ai/RunPod do. Treat this as the "if you outgrow marketplace GPUs" reference point, not a starting option. | [coreweave.com/pricing](https://www.coreweave.com/pricing) |

**A note on trust/reliability for marketplace GPUs (Vast.ai, TensorDock):** these are peer-hosted —
individual operators rent out spare capacity, prices are set per-listing, and quality/uptime vary.
Vast.ai publishes host reliability scores and uptime stats per listing; picking a host with a strong
track record mitigates but doesn't eliminate the risk relative to a traditional cloud DC. Given this
project's "nice to have later" GPU posture and tolerance for interruption (batch-run nodes, periodic
sync), marketplace GPUs are a reasonable fit as long as checkpointing already exists for the CPU-only
path.

---

## Synthesis

**Cheapest-that's-still-reasonable, CPU:**
1. **Oracle Cloud Always Free (Ampere A1 ARM)** is the standout if the resource need genuinely fits
   within 2-4 OCPU / 12-24 GB — it's free, indefinitely, for exactly the "moderate RAM, CPU-bound,
   started/stopped in batches" profile described. The catch is the free allotment was just cut in
   half (June 2026) without notice and ARM wheels are required for the Python/PyTorch stack.
2. **DigitalOcean** and **Vultr** are the best "true hourly, batch on/off" fit among paid options —
   clean per-second billing, predictable ~$48/mo for 4 vCPU/8GB, generous bandwidth, no surprise
   minimum commitments.
3. **OVHcloud b3-16** (4 vCPU/16GB at $74.96/mo, $0.12/hr) is a reasonable middle ground if more RAM
   per core matters, and was the one provider whose official pricing page rendered fully for this
   report — worth spot-checking there first when re-verifying numbers.

**If budget matters less (CPU):** **AWS EC2 m6i.xlarge** or **GCP e2-standard-4** (~$0.13-0.19/hr,
$98-140/mo) — the well-known reference points, at roughly 2-3x the budget-VPS price for broadly
comparable specs, with mature tooling, Spot/committed-use discounts, and enterprise-grade reliability
in exchange.

**Cheapest-that's-still-reasonable, GPU:**
1. **Vast.ai** is the floor of the market (RTX 3060 at $0.02/hr) — appropriate given this project's
   small nets and tolerance for interruption, provided a host with a good reliability score is picked.
2. **RunPod Community Cloud** (RTX A5000 at $0.16/hr) is a good step up in predictability while
   staying cheap, still marketplace-adjacent but with RunPod's own trust/rating layer around it.
3. **Lambda Labs** (RTX 6000/A10 at $0.60-0.69/hr) is the cheapest *non-marketplace* option — worth
   it if peer-hosted reliability risk is unwanted even at this small a scale.

**If budget matters less (GPU):** **AWS EC2 g4dn.xlarge** (T4, ~$0.53/hr on-demand) — mature,
well-documented, Spot-discountable, and consistent with the "entry-level card is plenty" framing
already established for this project.
