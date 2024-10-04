#!/bin/bash

# Update package list and install required libraries
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0

# Start Gunicorn
gunicorn --bind=0.0.0.0 --timeout 600 app:app
