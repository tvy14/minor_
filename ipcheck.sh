#!/bin/bash

# Print out section header
print_section_header() {
  echo "==============================="
  echo " $1"
  echo "==============================="
}


# check network hw
check_net_hw(){
  print_section_header "Network Hardware Info"
  sudo lshw -class network
}

# check Ethernet Interface Info
check_eth_interface(){
  print_section_header "Ethernet Interface Info"
  sudo ethtool eno2
}

# check ip
check_ip_address() {
  print_section_header "Check IP Address"
  ifconfig
}

# check IP Status (LAN or WLAN)
check_ip_status_config() {
  print_section_header "IP Address Status and Config"
  ip addr show
}

# check Connection Status (LAN / WLAN / Ip-Static/Dynamic)
check_connection_status() {
  print_section_header "Connection Status"
  nmcli con show
}

# check Routing Table
check_routing_table() {
  print_section_header "Routing Table"
  ip route list
}

# Scan Neighbors IP
check_neighbors_ip() {
  print_section_header "Scan Neighbors IP"
  sudo ip neighbour
}


#other functions to be added


# Main function to execute all checks
main() {
  check_net_hw
  check_eth_interface
  check_ip_address
  check_ip_status_config
  check_connection_status
  check_routing_table
  check_neighbors_ip()
}

main
