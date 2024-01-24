#!/bin/bash

# used only if the AMI misses to attach the image correctly

disk_mount_path=/mnt/neuron
device=/dev/sda1

# Mount the disk if it is not already mounted
if ! mountpoint -q $disk_mount_path; then
    sudo mkdir -p $disk_mount_path
    # Format the disk, create a file system 
    sudo mkfs -t xfs /dev/sda1
    # mount the disk
    sudo mount $device $disk_mount_path
    # adjust user permissions
    sudo chown -R $USER:$USER $disk_mount_path
fi

# Add the HF_HOME environment variable to point to the disk
mkdir -p $disk_mount_path/hf_cache
echo "export HF_HOME=$disk_mount_path/hf_cache" >> ~/.bashrc
export HF_HOME=$disk_mount_path/hf_cache
export XDG_CACHE_HOME=$disk_mount_path