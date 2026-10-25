#!/bin/bash

serial_number=$(udevadm info -a -n /dev/ttyACM* | grep serial | awk 'NR==1 {print substr($0, 21,12)}')
echo -e "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"16d0\", ATTRS{idProduct}==\"117e\", ATTRS{serial}==\"$serial_number\", SYMLINK+=\"arxcan1\"" > arx_can.rules
sudo cp arx_can.rules /etc/udev/rules.d/
sleep 1
sudo chmod +x /etc/udev/rules.d/arx_can.rules
sleep 1
sudo udevadm control --reload-rules && sudo udevadm trigger
./arx_can/arx_can1.sh
