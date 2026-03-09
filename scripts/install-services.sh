#!/bin/bash

# Install systemd services for Gamified Hockey Shot app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_DIR/services"

echo "Installing Gamified Hockey Shot systemd services..."

# Copy service files to systemd
sudo cp "$SERVICES_DIR/gamified-hs-frontend.service" /etc/systemd/system/
sudo cp "$SERVICES_DIR/gamified-hs-backend.service" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable gamified-hs-frontend
sudo systemctl enable gamified-hs-backend

echo "Services installed and enabled!"
echo ""
echo "To start services now:"
echo "  sudo systemctl start gamified-hs-backend"
echo "  sudo systemctl start gamified-hs-frontend"
echo ""
echo "To check status:"
echo "  sudo systemctl status gamified-hs-frontend"
echo "  sudo systemctl status gamified-hs-backend"
echo ""
echo "To view logs:"
echo "  journalctl -u gamified-hs-frontend -f"
echo "  journalctl -u gamified-hs-backend -f"
