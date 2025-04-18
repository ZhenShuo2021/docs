---
title: 安裝與設定
slug: /installation
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2024-09-10T03:07:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T03:07:33+08:00
---

## 安裝

Git 可以安裝命令行 (CLI) 和圖形介面 (GUI) 兩種工具，圖形介面就是以往我們熟悉的圖形應用程式，用滑鼠點擊選擇，命令行則是直接輸入指令。不建議初學者使用圖形介面操作，因為會搞不清楚確切在執行的指令是什麼，而且圖形介面不會包含所有指令。

<!-- truncate -->

命令行介面在[官網](https://git-scm.com/downloads)選擇自己的系統安裝，就算一路 next 都可以安裝完。

圖形介面筆者只用 VSCode 裡面的 [Git Graph](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph) 和 [Git History](https://marketplace.visualstudio.com/items?itemName=donjayamanne.githistory) 方便查看歷史記錄，其餘可選項目有 [Git Blame](https://marketplace.visualstudio.com/items?itemName=waderyan.gitblame) 可以看程式是誰寫的，[GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens) 介面雜亂臃腫而且還會要你付費升級。

## 設定

安裝後先進行基礎設定，首先開啟終端機設定使用者名稱和 email，macOS 就叫做終端機，Windows 是 Powershell 或 CMD

```sh
# 基礎必要設定
git config --global user.name "example"
git config --global user.email example@example.com

# 可選：忽略系統檔案
git config --global core.excludesfile ~/.gitignore
echo -e ".DS_Store\ndesktop.ini" >> ~/.gitignore

# 可選：rebase 時自動 stash 檔案
git config --global rebase.autoStash true

# 可選：pull 時自動 rebase
git config --global pull.rebase true

# 可選：設定別名
git config --global alias.ll "log --graph --pretty='%Cred%h%Creset -%C(auto)%d%Creset %s %Cgreen(%ar) %C(bold blue)<%an>%Creset' --all"
```

alias 是設定別名，非必要但是是最有感的，之後就可以使用 `git ll` 指令印出漂亮且易讀的歷史紀錄。

每個儲存庫可以新增 `.gitignore` 文件設定忽略追蹤的清單，例如安裝的套件 (node_modules, .venv) 或是敏感文件 (.env)。根據你的專案到 [github/gitignore](https://github.com/github/gitignore) 直接複製模板，或者使用 [.gitignore Generator](https://www.git-tower.com/free-tools/gitignore/) 輸入程式語言等關鍵字自動生成，不要浪費時間自己寫。
