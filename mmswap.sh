sudo swapon --show
free -h
sudo dd if=/dev/zero of=/swap_file bs=1GB count=64
sudo chmod 600 /swap_file
sudo mkswap /swap_file
sudo swapon /swap_file
free -h
