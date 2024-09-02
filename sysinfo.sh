#!/bin/bash

# Print out section header
print_section_header() {
  echo "==============================="
  echo " $1"
  echo "==============================="
}

# check Ubuntu version
check_ubuntu_version() {
  print_section_header "Ubuntu Version"
  lsb_release -a
}

# check Py version
check_python_version() {
  print_section_header "Python Version"
  python3 --version
}

# check GCC Version
check_gcc_version() {
  print_section_header "GCC Compiler Version"
  gcc --version
}

# Check kernel headers
check_kernel_headers() {
  print_section_header "Kernel Headers"
  uname -r && cat etc/*release
  dpkg -l | grep linux-headers | awk '{print $2, $3}'
}

# Check GPU in our System
check_gpu(){
  print_section_header "GPU Type"
  lspci | grep -i nvidia
}



# Check NV driver
check_nvidia_driver() {
  print_section_header "NVIDIA Driver"
  nvidia-smi
}

# Check CUDA LLVM
check_cuda_compiler() {
    print_section_header "CUDA Compiler"
    nvcc -V
}



# Main function to execute all checks
main() {
  check_ubuntu_version
  check_python_version
  check_gcc_version
  check_kernel_headers
  check_gpu
  check_nvidia_driver
  check_cuda_compiler
}

main
