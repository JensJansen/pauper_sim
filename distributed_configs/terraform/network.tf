# Private network + subnet + SNAT gateway, shared by every worker instance
# -- one gateway for the whole fleet, not one per instance (deliberately
# NOT using ovh_cloud_project_instance's inline network.private.gateway_create,
# which is per-instance -- see OVH_TERRAFORM_NOTES.md section 2c). Schema
# verified against the provider's own docs; the worked example there is
# reproduced closely since it's the only evidence found for the gateway
# resource's network_id argument using the region-specific "openstackid"
# rather than the network resource's own plain .id (see the comment below).

resource "ovh_cloud_project_network_private" "workers" {
  service_name = var.service_name
  name         = "rl-workers-net"
  regions      = [var.region]
}

resource "ovh_cloud_project_network_private_subnet" "workers" {
  service_name = ovh_cloud_project_network_private.workers.service_name
  network_id   = ovh_cloud_project_network_private.workers.id
  region       = var.region
  network      = "10.42.0.0/16"
  start        = "10.42.0.10"
  end          = "10.42.255.254"
  dhcp         = true
}

resource "ovh_cloud_project_gateway" "workers" {
  service_name = ovh_cloud_project_network_private.workers.service_name
  name         = "rl-workers-gw"
  model        = "s" # smallest model -- outbound-only traffic for periodic sync, no reason to size up
  region       = ovh_cloud_project_network_private_subnet.workers.region
  # NOT ovh_cloud_project_network_private.workers.id (that's the pn-xxxx
  # network ID) -- the gateway resource's own network_id argument takes the
  # region-specific OpenStack network ID instead, per the provider docs'
  # own worked example (OVH_TERRAFORM_NOTES.md section 6). Two different ID
  # namespaces for "the same" network; this distinction is demonstrated in
  # the docs' example code, never stated in prose on either resource's page.
  network_id = tolist(ovh_cloud_project_network_private.workers.regions_attributes[*].openstackid)[0]
  subnet_id  = ovh_cloud_project_network_private_subnet.workers.id
}
