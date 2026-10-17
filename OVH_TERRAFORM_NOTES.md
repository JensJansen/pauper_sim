# OVHcloud `ovh` Terraform Provider — Research Notes

Researched: 2026-09-01, for a module that provisions Ubuntu, CPU-only, no-public-IP
RL-training worker nodes (cloud-init installs Tailscale, clones a repo, starts a
systemd service).

**Source note on how this was fetched.** `registry.terraform.io/providers/ovh/ovh/latest/docs/...`
pages are JS-rendered — `WebFetch` returned an empty shell for them (confirmed
by directly fetching `https://registry.terraform.io/providers/ovh/ovh/latest/docs`
and getting no body content). The Terraform Registry renders those pages
directly from the `docs/` markdown files in the provider's own GitHub repo
(`ovh/terraform-provider-ovh`), so every schema claim below was instead fetched
from the **raw markdown source of the same docs**, at commit-tip `master`
(current release tag `v2.19.0` per `gh api repos/ovh/terraform-provider-ovh/tags`).
This is the identical content the registry page renders — not a blog post or a
cached summary — with the exact `raw.githubusercontent.com` URL cited per
section. Two OVHcloud KB pages were fetched directly via `WebFetch` (URLs
noted where used).

---

## 1. Provider configuration (`provider "ovh" {}`)

Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/index.md

Three auth modes are supported: OAuth2 (`client_id`/`client_secret`), a
short-lived `access_token`, or the classic **application key / application
secret / consumer key** triple. All arguments are individually `(Optional)`
in the schema because each falls back to an environment variable or an
`ovh.conf` file — but functionally you must supply one complete auth mode.

| Argument | Env var fallback | Notes |
|---|---|---|
| `endpoint` | `OVH_ENDPOINT` | Which OVH API region to target |
| `application_key` | `OVH_APPLICATION_KEY` | |
| `application_secret` | `OVH_APPLICATION_SECRET` | |
| `consumer_key` | `OVH_CONSUMER_KEY` | |
| `client_id` | `OVH_CLIENT_ID` | OAuth2 mode |
| `client_secret` | `OVH_CLIENT_SECRET` | OAuth2 mode |
| `access_token` | `OVH_ACCESS_TOKEN` | Access-token mode |
| `user_agent_extra` | — | appended to the UA string |
| `http_headers` | `OVH_HTTP_HEADERS_0`, `_1`, ... | map of extra headers |
| `api_rate_limit` | — | ops/sec; unlimited if omitted |
| `ignore_init_error` | `OVH_IGNORE_INIT_ERROR` | skips the `/auth/details` validation call at init — docs flag "use with caution" |

`endpoint` valid values (doc-listed): `ovh-eu`, `ovh-us`, `ovh-ca`,
`soyoustart-eu`, `soyoustart-ca`, `kimsufi-eu`, `kimsufi-ca`, or an arbitrary
URL for testing. (The acceptance-test env-var list separately also shows
`runabove-ca` as a valid endpoint value.)

Example from the docs (app key/secret/consumer key mode):

```hcl
terraform {
  required_providers {
    ovh = { source = "ovh/ovh" }
  }
}

provider "ovh" {
  endpoint           = "ovh-eu"
  application_key    = "xxxxxxxxx"
  application_secret = "yyyyyyyyy"
  consumer_key       = "zzzzzzzzzzzzzz"
}
```

---

## 2. Compute instance resource — which one, and its schema

### 2a. Two different resources exist; the picture is genuinely split

- **`ovh_cloud_project_instance`** (in the `ovh` provider) is a real, currently
  documented, actively-shipped resource (present in `docs/resources/cloud_project_instance.md`
  in the current `master`/v2.19.0 provider). Its own doc header says: **"This
  resource uses a Beta API."**
  Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_project_instance.md

- OVHcloud's own **official control-panel → Terraform mapping guide**
  (fetched live via WebFetch, https://docs.ovhcloud.com/en/guides/manage-and-operate/terraform/at-ovhcloud
  — this is the redirect target of the KB0054612 link cited from the provider's
  own `docs/index.md`) maps **"Instances (VM & Metal)"** to the **`openstack`**
  provider's **`openstack_compute_instance_v2`** resource, not to
  `ovh_cloud_project_instance`. Fetched twice (different prompts) with
  consistent results both times; no mention of `ovh_cloud_project_instance`
  appeared on that page at all.

**Conclusion:** `ovh_cloud_project_instance` is confirmed to exist, is current,
and is not deprecated in the provider's own docs — but it is explicitly
labeled Beta, and OVHcloud's own cross-provider mapping guide points users at
`openstack_compute_instance_v2` (via the separate `openstack` provider,
authenticated against OVH's OpenStack-compatible Horizon/Nova endpoint) as
*the* documented way to manage Public Cloud instances. Both are legitimate,
OVH-maintained paths; they are not the same underlying API. Since the task
specifically asked for the native `ovh` provider resource and its schema
(question 1 explicitly names `ovh_cloud_project_instance`), the rest of this
document proceeds with `ovh_cloud_project_instance`, but **this split is worth
flagging to the repo owner before locking in the module design** — see
"Could not verify" below for the one thing this leaves unresolved.

### 2b. `ovh_cloud_project_instance` full argument schema

Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_project_instance.md

Required (all `Forces new resource` except where noted):

- `service_name` — public cloud project ID. Falls back to `OVH_CLOUD_PROJECT_SERVICE` env var if omitted.
- `region` — instance region.
- `billing_period` — `hourly` or `monthly`.
- `name` — instance name.
- `flavor` block, required nested arg `flavor_id` (UUID — see §6 data sources).
- `boot_from` block: `image_id` **or** `volume_id` (mutually exclusive, one required).
- `network` block — see §2c below.
- `ssh_key` block (`name`) **or** `ssh_key_create` block (`name`, `public_key`) — one required.

Optional:

- `availability_zone` — the only argument in this resource that is **not** `Forces new resource`.
- `user_data` — cloud-init / launch script content. Plain string argument, `Forces new resource`.
- `group` block (`group_id`) — start instance in an instance group.
- `auto_backup` block (`cron`, `rotation`).

Exported attributes: `addresses` (list of `ip`/`version`), `attached_volumes`,
`flavor_id`, `flavor_name`, `id`, `name`, `image_id`, `task_state`, `status`.

`timeouts { create = "10min" }` supported, default `60m`.

### 2c. The `network` block — public vs private-only, exact shape

This is the doc-verified nested schema, straight from the same page:

```
network (Required, Forces new resource)
  public          (Optional) Set the new instance as public
  private         (Optional) Private network information
    floating_ip          (Optional)
      id                    Floating IP ID
    floating_ip_create    (Optional)
      description           Floating IP description
    gateway               (Optional) Existing gateway
      id                    Gateway ID
    gateway_create        (Optional) Info to create a new gateway
      model                 Gateway model (s | m | l)
      name                  Gateway name
    ip                    (Optional) Instance IP in the private network
    network               (Optional) Existing private network
      id                    Network ID
      subnet_id             Existing subnet ID
    network_create        (Optional) Info to create a new private network
      name
      vlan_id
      subnet
        cidr
        enable_dhcp
        ip_version
```

There is **no separate `network_ids` / `private_network_id` top-level
argument** — everything routes through the single `network` block with its
`public` and `private` sub-blocks, exactly as shown above. No literal argument
named `Ext-Net` appears anywhere in this schema; `public = true` is how the
public network interface is requested.

**For no-public-IP, private-network-only:** the doc's only worked example sets
`network { public = true }` and does not show a private-only example. Reading
the schema directly: `public` is an independent optional boolean on the same
`network` block as `private`, so a private-only instance is `network { private
{ network { id = <private_network_id>; subnet_id = <subnet_id> } } }` with
`public` simply omitted. **This specific omit-for-no-public-IP behavior is an
inference from the schema shape, not a sentence stated in the docs** — flagged
explicitly in "Could not verify" below.

You can either point `private.network.id`/`subnet_id` at an **existing**
private network + subnet (the §3/§4 resources below), or use the inline
`network_create` block to have the instance provision its own private network
+ subnet in the same apply. For a shared multi-worker setup, using the
standalone network/subnet/gateway resources (so all workers join one network)
is the sane choice — `network_create` is per-instance.

The `private.gateway_create` sub-block is also worth noting: it's an
**alternative** to the standalone `ovh_cloud_project_gateway` resource (§5) —
you can have the instance itself provision its outbound gateway inline instead
of managing a separate resource. For multiple worker instances sharing one
gateway, the standalone gateway resource (§5) referenced by ID via
`private.gateway.id` is the correct shape, not `gateway_create` per instance.

---

## 3. SSH key

Two resources exist in the `ovh` provider; the newer one is explicitly
recommended over the older:

- **`ovh_cloud_project_ssh_key`** — doc header: *"Prefer using the new
  `ovh_cloud_ssh_key` resource instead."*
  Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_project_ssh_key.md
  Args: `name` (required), `public_key` (required), `region` (optional),
  `service_name` (optional, env fallback `OVH_CLOUD_PROJECT_SERVICE`).

- **`ovh_cloud_ssh_key`** — current recommended resource. Doc header: *"This
  resource uses a Beta API."* Both `name` and `public_key` are **immutable**
  (changing either forces replacement).
  Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_ssh_key.md
  Args: `name` (required), `public_key` (required), `service_name` (optional,
  same env fallback). Exports `finger_print`, `id`, `regions`, `created_at`, `updated_at`.

The instance resource's `ssh_key { name = ... }` block references an existing
key by **name** (not by resource ID) — so either SSH-key resource works to
create it, or you can reference a key name that already exists in the project
without managing it in Terraform at all.

---

## 4. Private network resource

Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_project_network_private.md

Resource: **`ovh_cloud_project_network_private`**

- `service_name` (Required) — falls back to `OVH_CLOUD_PROJECT_SERVICE`.
- `name` (Required).
- `vlan_id` (Optional) — defaults to `0`; changing recreates the resource.
- `regions` (Optional) — array of region IDs (e.g. `"GRA1"`) the network is
  available in; defaults to all public cloud regions.

Exports (useful for downstream resources): `id`, `regions_attributes` (map
with `region`, `status`, `openstackid` per region — `openstackid` is the
per-region OpenStack network ID used by the gateway resource in the docs'
own worked example), `regions_openstack_ids`, `status`, `type`.

Import format: `service_name/network_id` where network_id is `pn-xxxx`.

## 5. Private network subnet resource

Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_project_network_private_subnet.md

Resource: **`ovh_cloud_project_network_private_subnet`**

- `service_name` (Required).
- `network_id` (Required) — the parent private network's ID; changing recreates.
- `network` (Required) — global CIDR, e.g. `192.168.168.0/24`; changing recreates.
- `start` (Required) — first usable IP for this region; changing recreates.
- `end` (Required) — last usable IP for this region; changing recreates.
- `region` (Optional per schema listing, but required in practice for a
  region-scoped network) — e.g. `"GRA1"`; changing recreates.
- `dhcp` (Optional) — defaults `false`; changing recreates.
- `no_gateway` (Optional) — defaults `false`; skip assigning a default gateway IP.

Exports: `dhcp_id`, `gateway_ip`, `cidr`, `ip_pools` (list of per-region pool info).

Import format: `service_name/network_id/subnet_id`.

There is also a `cloud_project_network_private_subnet_v2` resource file
present in the repo's `docs/resources/` directory — **not fetched/verified in
this pass**; flagged in "Could not verify."

---

## 6. SNAT Gateway resource

Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/resources/cloud_project_gateway.md

Resource: **`ovh_cloud_project_gateway`**

All arguments required:

- `service_name` — doc literally labels this "ID of the private network" in
  the argument table (inconsistent wording vs. every other resource, which
  calls it "the id of the public cloud project" — but the worked example sets
  it to `ovh_cloud_project_network_private.mypriv.service_name`, i.e. the same
  public-cloud-project service name used everywhere else. Treat it as the
  project service name; the doc's own field label is just worded oddly).
- `name` — gateway name.
- `region` — must match the subnet's region.
- `model` — gateway size/model (worked example uses `"s"`; valid values not
  enumerated on this page beyond the instance resource's `gateway_create.model`
  note of `s | m | l`).
- `network_id` — ID of the private network. In the doc's own example this is
  **not** the private-network resource's own `.id`, but rather the
  **region-specific OpenStack network ID**, pulled via
  `tolist(ovh_cloud_project_network_private.mypriv.regions_attributes[*].openstackid)[0]`.
- `subnet_id` — the subnet resource's `.id`.

Doc's own worked example (verbatim):

```hcl
resource "ovh_cloud_project_network_private" "mypriv" {
  service_name = "xxxxxxxxxx"
  vlan_id      = "0"
  name         = "mypriv"
  regions      = ["GRA9"]
}

resource "ovh_cloud_project_network_private_subnet" "myprivsub" {
  service_name = ovh_cloud_project_network_private.mypriv.service_name
  network_id   = ovh_cloud_project_network_private.mypriv.id
  region       = "GRA9"
  start        = "10.0.0.2"
  end          = "10.0.255.254"
  network      = "10.0.0.0/16"
  dhcp         = true
}

resource "ovh_cloud_project_gateway" "gateway" {
  service_name = ovh_cloud_project_network_private.mypriv.service_name
  name         = "my-gateway"
  model        = "s"
  region       = ovh_cloud_project_network_private_subnet.myprivsub.region
  network_id   = tolist(ovh_cloud_project_network_private.mypriv.regions_attributes[*].openstackid)[0]
  subnet_id    = ovh_cloud_project_network_private_subnet.myprivsub.id
}
```

Note the subnet resource's `network_id` argument uses the *private network's*
plain `.id` (`pn-xxxx`), while the *gateway's* `network_id` argument uses the
region-specific `openstackid` from `regions_attributes` — these are two
different ID namespaces for "the same" network, and the doc's own example is
the only evidence for this distinction (it is not called out in prose on
either page, only demonstrated in the code).

Exports: `id`, `external_information` (external network id + IPs), `interfaces`
(per-interface id/ip/network_id/subnet_id), `status`.

---

## 7. Flavor / image ID lookups (needed because the instance resource takes raw UUIDs)

`ovh_cloud_project_instance`'s `flavor.flavor_id` and `boot_from.image_id`
are opaque IDs, not names like `"b3-8"` or `"Ubuntu 24.04"` — the resource
schema has no name-based shortcut. Two data sources resolve this:

- **`ovh_cloud_project_flavors`** (plural, list/filter) — Optional
  `name_filter` (doc's own example: *"Filter flavors using the given name
  (strict equality, e.g. b3-8)"*), Optional `region`. Returns a `flavors` set
  with `.id`, `.name`, `.region`, `.vcpus`, `.ram`, `.disk`, etc.
  Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/data-sources/cloud_project_flavors.md

- **`ovh_cloud_project_images`** (plural) — Optional `region`, `os_type`
  (allowed: `baremetal-linux`, `bsd`, `linux`, `windows`), `flavor_type`. **No
  name filter argument** — returns an `images` set with `.id`/`.name`, and you
  filter by name in HCL yourself (e.g. a `for` expression matching on
  `"Ubuntu 24.04"` in `.name`).
  Source: https://raw.githubusercontent.com/ovh/terraform-provider-ovh/master/docs/data-sources/cloud_project_images.md

(Singular `ovh_cloud_project_flavor` / `ovh_cloud_project_image` also exist
but require the ID as input already — useless for name-based lookup; only
listed as read-one-by-id lookups.)

---

## Could not verify

- **Whether `ovh_cloud_project_instance` or `openstack_compute_instance_v2`
  is the "recommended" resource for new code is not settled by primary docs**
  — the provider's own resource docs show `ovh_cloud_project_instance` as
  live and current (Beta API, not deprecated), while OVHcloud's official
  cross-provider mapping guide (docs.ovhcloud.com) maps Public Cloud instances
  to `openstack_compute_instance_v2` instead and doesn't mention
  `ovh_cloud_project_instance` at all. No page found that reconciles these
  two facts or states a preference between them. **Recommend the repo owner
  make an explicit call here before the module is finalized** — this
  materially affects which resource the rest of the module is built around.
- **Whether omitting `network.public` on `ovh_cloud_project_instance` actually
  yields no public IP** — the schema shows `public` as an independent
  Optional boolean sibling of `private`, and the only worked example sets
  `public = true`; no private-only example or explicit prose ("omit this for
  no public IP") was found on the page to confirm the omitted/false behavior.
- **`ovh_cloud_project_network_private_subnet_v2`** — a doc file for this
  resource exists in the repo (`docs/resources/cloud_project_network_private_subnet_v2.md`)
  but was not fetched or read in this research pass; unknown whether it
  supersedes the `v1` resource used throughout this document.
- **Valid values for `gateway.model`** beyond `s`/`m`/`l` (that enumeration
  comes from the *instance* resource's `gateway_create.model` field
  description, not from the standalone gateway resource's own page, which
  gives no enumeration) — not independently confirmed on the gateway resource's own page.
- **`registry.terraform.io` page content itself** was not directly readable
  via WebFetch (JS-rendered, empty on fetch) for any of the pages above — all
  claims come from the GitHub-hosted markdown source that page renders from,
  not from the rendered HTML itself, so trivial rendering differences
  (e.g. auto-generated "Beta" badges, added sidebar notes) between the
  markdown source and the live rendered page cannot be ruled out.

---

## Example `.tf` — assembled only from verified schema above

```hcl
terraform {
  required_providers {
    ovh = {
      source = "ovh/ovh"
    }
  }
}

provider "ovh" {
  endpoint           = "ovh-eu"
  application_key    = var.ovh_application_key
  application_secret = var.ovh_application_secret
  consumer_key       = var.ovh_consumer_key
}

variable "ovh_application_key" {}
variable "ovh_application_secret" {}
variable "ovh_consumer_key" {}
variable "service_name" {
  description = "OVHcloud Public Cloud project ID"
}

# --- Private network + subnet ---------------------------------------------

resource "ovh_cloud_project_network_private" "workers" {
  service_name = var.service_name
  name         = "rl-workers-net"
  regions      = ["GRA9"]
}

resource "ovh_cloud_project_network_private_subnet" "workers" {
  service_name = ovh_cloud_project_network_private.workers.service_name
  network_id   = ovh_cloud_project_network_private.workers.id
  region       = "GRA9"
  start        = "10.0.0.2"
  end          = "10.0.255.254"
  network      = "10.0.0.0/16"
  dhcp         = true
}

# --- SNAT gateway for outbound internet from the private-only workers -----

resource "ovh_cloud_project_gateway" "workers" {
  service_name = ovh_cloud_project_network_private.workers.service_name
  name         = "rl-workers-gw"
  model        = "s"
  region       = ovh_cloud_project_network_private_subnet.workers.region
  network_id   = tolist(ovh_cloud_project_network_private.workers.regions_attributes[*].openstackid)[0]
  subnet_id    = ovh_cloud_project_network_private_subnet.workers.id
}

# --- SSH key (current recommended resource) --------------------------------

resource "ovh_cloud_ssh_key" "worker_key" {
  service_name = var.service_name
  name         = "rl-worker-key"
  public_key   = file("${path.module}/worker_key.pub")
}

# --- Flavor / image lookups (instance resource needs raw IDs) -------------

data "ovh_cloud_project_flavors" "b3_8" {
  service_name = var.service_name
  region       = "GRA9"
  name_filter  = "b3-8"
}

data "ovh_cloud_project_images" "ubuntu" {
  service_name = var.service_name
  region       = "GRA9"
  os_type      = "linux"
}

locals {
  flavor_id = [for f in data.ovh_cloud_project_flavors.b3_8.flavors : f.id][0]
  image_id = [
    for img in data.ovh_cloud_project_images.ubuntu.images :
    img.id if strcontains(img.name, "Ubuntu 24.04")
  ][0]
}

# --- Compute instance: private-network-only, no public IP -----------------

resource "ovh_cloud_project_instance" "worker" {
  service_name   = var.service_name
  region         = "GRA9"
  billing_period = "hourly"
  name           = "rl-worker-0"

  flavor {
    flavor_id = local.flavor_id
  }

  boot_from {
    image_id = local.image_id
  }

  network {
    # public omitted -> no public interface (see "Could not verify": this
    # omitted-means-off behavior is inferred from the schema, not doc prose)
    private {
      network {
        id        = ovh_cloud_project_network_private.workers.id
        subnet_id = ovh_cloud_project_network_private_subnet.workers.id
      }
      gateway {
        id = ovh_cloud_project_gateway.workers.id
      }
    }
  }

  ssh_key {
    name = ovh_cloud_ssh_key.worker_key.name
  }

  user_data = <<-EOT
    #cloud-config
    package_update: true
    packages:
      - curl
    runcmd:
      - curl -fsSL https://tailscale.com/install.sh | sh
      - tailscale up --authkey=${var.tailscale_authkey} --ssh
      - git clone https://github.com/example/repo.git /opt/repo
      - systemctl enable --now rl-worker.service
  EOT
}

variable "tailscale_authkey" {}
```

Argument names in this snippet are all taken directly from the verified
schemas in sections 1–7 above; the `locals` filtering block and the
`user_data` cloud-init body are original composition (the cloud-init content
itself is not something the provider docs specify — `user_data` is just a
plain string argument per §2b).
