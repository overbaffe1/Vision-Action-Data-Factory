#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cat part_0000 part_0001 part_0002 part_0003 part_0004 part_0005 part_0006 part_0007 part_0008 part_0009 part_0010 part_0011 part_0012 part_0013 part_0014 part_0015 part_0016 part_0017 part_0018 part_0019 part_0020 part_0021 part_0022 part_0023 part_0024 part_0025 part_0026 part_0027 part_0028 part_0029 part_0030 part_0031 part_0032 part_0033 part_0034 > ../submission.zip
echo "MD5: $(md5sum ../submission.zip)"
echo "Size: $(stat -c%s ../submission.zip) bytes"
