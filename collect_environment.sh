#! /bin/sh

# Linux data-gathering commands; adjust as necessary for your platform.
#
# Be sure to remove any information from the output that would violate
# SC's double-blind review policies.

env | sed "s/$USER/USER/g"
set -x
cat /etc/redhat-release
uname -a
lscpu || cat /proc/cpuinfo
cat /proc/meminfo
inxi -F -c0
lsblk -a
(lshw -short -quiet -sanitize || lspci) | cat
pip list
