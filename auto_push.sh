#!/bin/bash

# 修改为你的 Vault 路径
cd ~/ObsidianVault

echo "------------------------------------"
echo " Starting Obsidian Git Backup..."
echo "------------------------------------"

current_time=$(date "+%Y-%m-%d %H:%M:%S")
git add .
git commit -m "auto backup at $current_time"
git push

echo "------------------------------------"
echo " Backup completed at $current_time"
