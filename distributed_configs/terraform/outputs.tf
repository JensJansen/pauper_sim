output "worker_instance_ids" {
  value = { for deck, inst in ovh_cloud_project_instance.worker : deck => inst.id }
}

output "worker_addresses" {
  description = "Private-network addresses only -- no public IP is expected (see instances.tf's own note on this being unverified until a real apply confirms it)."
  value       = { for deck, inst in ovh_cloud_project_instance.worker : deck => inst.addresses }
}

output "gateway_id" {
  value = ovh_cloud_project_gateway.workers.id
}
