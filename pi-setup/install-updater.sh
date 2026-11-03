#!/bin/bash
# Run once on each Pi, from the folder containing these files:
#   bash install-updater.sh
set -e

U=dorkmaster

sudo install -m 755 -o $U -g $U update-lab-code.sh /home/$U/update-lab-code.sh
sudo install -m 644 update-lab-code.service /etc/systemd/system/update-lab-code.service

# Fix ownership in case anything was ever created with sudo
[ -d /home/$U/Desktop/PHY1035 ] && sudo chown -R $U:$U /home/$U/Desktop/PHY1035

# Make network-online.target actually mean "online"
sudo systemctl enable NetworkManager-wait-online.service 2>/dev/null || true

sudo systemctl daemon-reload
sudo systemctl enable update-lab-code.service
sudo systemctl restart update-lab-code.service

echo
echo "Installed. Now check for and remove any OLD update mechanism:"
echo "  systemctl list-unit-files | grep -i -e update -e lab"
echo "  crontab -l ; sudo crontab -l          (look for @reboot lines)"
echo "  ls ~/.config/autostart /etc/xdg/autostart"
echo
echo "Watch this run with:  journalctl -u update-lab-code -f"
