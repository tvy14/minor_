#!/bin/bash

#Section header to be print out
section_header_print() {
   echo "============"
   echo " $1"
   echo "============"
}


#Scan IP Neighbor
ip_scan_neighbor() {
section_header_print "Scan Neighbors IP" 
   ip neighbour
}   

#List the used IP
list_used_ip() {   
section_header_print "List scanned neighbors IP that is used under the same domain" 
   nmap 192.168.101-110-123 #list 110-123 on target
}   
#getting error on this used IP, need debug



#other functions to be added
#will throw this under the server-ICCL repo


#Main Function
main() {
    ip_scan_neighbor
    list_used_ip
}

main

#updated 22nd Jan '24 - xard40
