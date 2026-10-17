# Provisions one OVHcloud Public Cloud instance per deck in var.decks --
# distributed training nodes, private-network-only (no public IP; Tailscale
# is the only routing path to the hub, per DISTRIBUTED_TRAINING_PLAN.md's
# networking decision). See OVH_TERRAFORM_NOTES.md for the verified
# resource schemas this is built from, and its "Could not verify" section
# for the two things NOT confirmed by primary docs (noted inline below,
# not glossed over).
#
# Resource choice: ovh_cloud_project_instance (the native `ovh` provider),
# not openstack_compute_instance_v2 -- OVHcloud's own official Terraform
# guide points at the latter, but this keeps every resource in this module
# on one provider/credential set. Owner's explicit call (2026-09-06), not
# a default -- see DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md.
#
# NOT YET RUN AGAINST A REAL ACCOUNT. `terraform validate` only checks
# syntax/internal consistency, not that these arguments are actually
# correct against OVH's live API -- that's Phase 3's job, once real
# credentials exist. Treat this module as "as close as the plan can get
# without them," not "proven."

terraform {
  required_version = ">= 1.5"
  required_providers {
    ovh = {
      source = "ovh/ovh"
    }
  }
}

provider "ovh" {
  endpoint           = var.ovh_endpoint
  application_key    = var.ovh_application_key
  application_secret = var.ovh_application_secret
  consumer_key       = var.ovh_consumer_key
}

# --- OVH account / project ---------------------------------------------

variable "ovh_endpoint" {
  description = "Which OVH API region to target -- e.g. \"ovh-eu\", \"ovh-us\", \"ovh-ca\". Must match where var.service_name's project actually lives."
  type        = string
}

variable "ovh_application_key" {
  description = "From the OVH API token-creation page (see OVHCLOUD_SETUP_NOTES.md)."
  type        = string
  sensitive   = true
}

variable "ovh_application_secret" {
  type      = string
  sensitive = true
}

variable "ovh_consumer_key" {
  type      = string
  sensitive = true
}

variable "service_name" {
  description = "OVHcloud Public Cloud project ID."
  type        = string
}

variable "region" {
  description = "e.g. \"GRA9\". Determines flavor/image availability -- OVHCLOUD_SETUP_NOTES.md found the cheaper Discovery range isn't offered in either US region, only Canada/EU/Singapore/Australia; confirm live for whichever region you pick."
  type        = string
}

# --- Instance sizing ------------------------------------------------------

variable "flavor_name" {
  description = "e.g. \"b3-8\" -- matched via ovh_cloud_project_flavors' name_filter (strict equality). Re-check current pricing/availability live before committing (HOSTING_RESEARCH.md/OVHCLOUD_SETUP_NOTES.md both flagged several numbers as needing live reconfirmation)."
  type        = string
  default     = "b3-8"
}

variable "image_name_contains" {
  description = "Substring match against ovh_cloud_project_images' name field (that data source has no name filter argument, unlike flavors -- see OVH_TERRAFORM_NOTES.md section 7)."
  type        = string
  default     = "Ubuntu 24.04"
}

# --- Fleet ------------------------------------------------------------

variable "decks" {
  description = "One instance is provisioned per entry -- this is purely a fleet-sizing knob, not a deck ASSIGNMENT: which deck a node actually ends up training is decided dynamically at registration time by the hub (see DISTRIBUTED_TRAINING_PLAN.md's \"Node lifecycle\"), not by this list or by the instance's own name."
  type        = list(string)
}

variable "ssh_public_key_path" {
  type = string
}

# --- Node bootstrap (rendered into node_bootstrap.sh.tftpl per instance) --

variable "hub_url" {
  description = "e.g. http://100.113.155.84:5000 -- the hub's stable Tailscale address. http, not https: the hub serves plain HTTP by default (--tls none), safe because it is only reachable inside the tailnet (WireGuard-encrypted)."
  type        = string
}

variable "registration_credential" {
  description = "From `python -m distributed.auth`, run once by the operator (see DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md decision #5's sibling note on the two distinct secrets)."
  type        = string
  sensitive   = true
}

variable "tailscale_auth_key" {
  description = "A REUSABLE, TAGGED (tag:pauper-node) Tailscale auth key -- see TAILSCALE_SETUP_NOTES.md. Max 90-day expiry; it is only needed when an instance is created."
  type      = string
  sensitive = true
}

variable "git_ref" {
  description = "A pinned commit SHA or tag -- never a branch name, see node_bootstrap.sh.tftpl's own comment on why. The commit it resolves to must be listed in the hub's accepted-versions file, or the node is rejected (426) and never trains."
  type        = string
}

variable "git_repo_url" {
  type = string
}

variable "league_config_name" {
  description = "e.g. main_league_node.json (CPU, 2 workers -- NOT main_league.json, whose device=cuda/n_workers=10 crash a b3-8 node) -- which training_configs/*.json every provisioned node reads."
  type        = string
  default     = "main_league_node.json"
}
