# SSH key + flavor/image lookups + the actual worker instances --
# private-network-only, no public IP anywhere.

resource "ovh_cloud_ssh_key" "worker_key" {
  service_name = var.service_name
  name         = "rl-worker-key"
  public_key   = file(var.ssh_public_key_path)
}

data "ovh_cloud_project_flavors" "worker" {
  service_name = var.service_name
  region       = var.region
  name_filter  = var.flavor_name
}

data "ovh_cloud_project_images" "ubuntu" {
  service_name = var.service_name
  region       = var.region
  os_type      = "linux"
}

locals {
  flavor_id = [for f in data.ovh_cloud_project_flavors.worker.flavors : f.id][0]
  image_id = [
    for img in data.ovh_cloud_project_images.ubuntu.images :
    img.id if strcontains(img.name, var.image_name_contains)
  ][0]
}

resource "ovh_cloud_project_instance" "worker" {
  for_each = toset(var.decks) # fleet size only -- see var.decks' own description; not a deck assignment

  service_name   = var.service_name
  region         = var.region
  billing_period = "hourly"
  name           = "rl-worker-${each.key}"

  flavor {
    flavor_id = local.flavor_id
  }

  boot_from {
    image_id = local.image_id
  }

  network {
    # `public` intentionally omitted -- private-only, no public IP.
    # UNVERIFIED BY DOCS (OVH_TERRAFORM_NOTES.md's "Could not verify"):
    # omitting this block is inferred from the schema shape (public is an
    # independent optional sibling of private), not confirmed in prose by
    # any worked example. Confirm for real on the first Phase 3 apply --
    # check the instance's own `addresses` export has no public entry --
    # before trusting this for the rest of the fleet.
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

  user_data = templatefile("${path.module}/node_bootstrap.sh.tftpl", {
    hub_url                 = var.hub_url
    registration_credential = var.registration_credential
    tailscale_auth_key      = var.tailscale_auth_key
    git_ref                 = var.git_ref
    git_repo_url            = var.git_repo_url
    league_config_name      = var.league_config_name
  })
}
