---
title: Git 進階設定
sidebar_label: Git 進階設定
slug: /git-configurations
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-19T02:06:00+08:00
  author: zsl0621
first_publish:
  date: 2025-04-19T02:06:00+08:00
---

Git 其實有很多設定可以客製化，連 git branch 顏色、git diff 工具都可以自訂，但是最常用且最實用的應該是別名系統。

指令超長根本記不起來也打的很痛苦該怎麼辦呢？設定別名 (alias) 可以把你從落落長的指令拯救出來。當然你也可以使用 [lazygit](https://github.com/jesseduffield/lazygit/) 完成，但是我自己還是習慣用別名來操作。

你可以把別名設定成 Git 本身的別名系統，或者是整個 shell 的別名系統，差在會不會全局衝突和前面需不需要打 git，以我個人的設定還是全局偏多。

## Git 設定

使用 git config --global 預設會修改 `~/.gitconfig`，在 Unix 系統上可以在 `~/.zshrc` 或者 `~/.bashrc` 裡面設定 `export GIT_CONFIG_GLOBAL="/path/to/gitconfig"` 自定義路徑，廢話不多說直接上我的設定，除了從 Github 上面抄了一波以外，主要就是設定 delta、rebase autostash 以及一些顏色設定

> 裡面出現的 <code v-pre>\{\{ .xxx \}\}</code> 是 go template，因為我的 dotfiles 是使用支援 go template 的 chezmoi 管理，如果你是 Mac 或者 Linux 用戶可以一鍵安裝[我的 dotfiles](https://github.com/ZhenShuo2021/dotfiles)，特別優化過速度，啟動超快。

```ini
[user]
  name = {{ .name }}
  email = {{ .email }}
  signingkey = 123456789
[core]
  excludesfile = {{ .chezmoi.destDir }}/.config/git/gitignore_global
  pager = delta
  editor = nvim
  quotepath = false
[include]
  path = ~/.gitconfig.local
[alias]
  # Credit: https://github.com/mathiasbynens/dotfiles/blob/main/.gitconfig
  # === 列出資訊類型的指令 ===
  # List aliases.
  aliases = config --get-regexp alias

  # List contributors with number of commits.
  contributors = shortlog --summary --numbered

  # `git ft <commit-ish>` 找到該提交的下一個標籤，並且顯示距離
  ft = "!f() { git describe --always --contains $1; }; f"

  # `git fc <any>` 從原始碼找到標籤
  fc = "!f() { git log --pretty=format:'%C(yellow)%h  %Cblue%ad  %Creset%s%Cgreen  [%cn] %Cred%d' --decorate --date=short -S$1; }; f"

  # `git fm <msg>` 從提交訊息找到標籤
  fm = "!f() { git log --pretty=format:'%C(yellow)%h  %Cblue%ad  %Creset%s%Cgreen  [%cn] %Cred%d' --decorate --date=short --grep=$1; }; f"

  # `git d` 和上一次的提交比較差異
  d = !"git diff-index --quiet HEAD -- || clear; git --no-pager diff --patch-with-stat"

  # `git di <number>` 和前 <number> 次的提交比較差異
  di = !"d() { git diff --patch-with-stat HEAD~$1; }; git diff-index --quiet HEAD -- || clear; d"

  # === 修改歷史的指令 ===
  # 移除本地和遠端分支
  nuke = !{{ .chezmoi.destDir -}}/.config/zsh/bin/git-nuke-branch
  
  # 自動追蹤遠端分支
  track = !{{ .chezmoi.destDir -}}/.config/zsh/bin/git-track

  # 預存除了子模組的所有修改並且提交
  ca = !git add ':(exclude,attr:builtin_objectmode=160000)' && git commit -av

  # `git credit <name> [<email>]` 修改前一個提交的作者
  credit = "!f() { git commit --amend --author \"$1 <$2>\" -C HEAD; }; f"

  # `git reb <number>` 對前 <number> 次提交進行互動式變基
  rbi = "!r() { git rebase -i HEAD~$1; }; r"

  # `git dm` 移除所有已經被合併到 main 的分支
  # a.k.a. ‘delete merged’
  dm = "!git branch --merged | grep -v '\\*' | xargs -n 1 git branch -d"


[init]
  defaultBranch = main
[commit]
  gpgsign = true
[delta]
  navigate = true  # use n and N to move between diff sections
  line-numbers = true
[merge]
  conflictstyle = diff3
[diff]
  colorMoved = default
[gui]
  encoding = utf-8
[i18n "commit"]
  encoding = utf-8
[i18n]
  logoutputencoding = utf-8
[rebase]
  autoStash = true
[rerere]
  enabled = 1
[pull]
  rebase = true


[color]
  diff = auto
  status = auto
  branch = auto
  ui = auto
[color "branch"]
  current = green reverse
  remote = blue
[color "diff"]
  meta = yellow bold
  frag = magenta bold # line info
  old = red # deletions
  new = green # additions
[color "status"]
  added = green
  changed = blue


[apply]
  whitespace = nowarn
[mergetool]
  keepBackup = false
[help]
  autocorrect = 1
[push]
  # See `git help config` (search for push.default)
  # for more information on different options of the below setting.
  #
  # Setting to git 2.0 default to suppress warning message
  default = simple
  followTags = true
[ghq]
  root = ~/Code
[interactive]
  diffFilter = delta --color-only
[filter "lfs"]
  required = true
  clean = git-lfs clean -- %f
  smudge = git-lfs smudge -- %f
  process = git-lfs filter-process


[credential "https://github.com"]
  helper =
  helper = !gh auth git-credential
[credential "https://gist.github.com"]
  helper =
  helper = !gh auth git-credential

[credential]
  helper = osxkeychain
[diff "spaceman-diff"]
  command = /opt/homebrew/bin/spaceman-diff
```

## Shell 設定

這部份 Windows 要改過才能用，，Unix 系統則是可以直接複製到 `~/.zshrc` 或者 `~/.bashrc` 就可以直接啟用：

```sh
alias gpg_test='echo test | gpg --clear-sign'
alias gpg_reload='gpgconf --kill gpg-agent; gpgconf --reload gpg-agent'
alias gpg_list_keys='gpg --list-keys'
alias gpg_list_config='gpgconf --list-options gpg-agent'
alias gpg_delete_key='gpg --delete-secret-and-public-keys'

alias g=git
alias gc='git commit'
alias 'gcn'='git commit --no-verify'
alias 'gcn!'='git commit --no-verify --amend'
alias 'gcnn!'='git commit --no-verify --amend --no-edit'

# change both committer date and author date
#   usage: 
#     gcd "2025-02-24 15:30:00"
#     gcd now
alias gcd='gcd(){ DATE=${1:-"now"}; [ "$DATE" = "now" ] && DATE=$(date "+%Y-%m-%d %H:%M:%S"); GIT_COMMITTER_DATE="$DATE" git commit --amend --date="$DATE" --allow-empty --no-edit; }; gcd'

alias gtl='gtl(){ git tag --sort=-v:refname -n --list "${1}*" }; noglob gtl'
alias 'gtll'='gtll(){ git tag --sort=-v:refname -n10 --format="[%(refname:short)] %(contents:lines=10)%0a" --list "${1}*" }; noglob gtll'
alias 'gtlll'='gtlll(){ git tag --sort=-v:refname -n999 --format="[%(objectname:short) %(refname:short)] %(contents:lines=999)%0a" --list "${1}*" }; noglob gtlll'

alias gca='git commit -a'
alias gco='git checkout'
alias gcp='git cherry-pick'
alias grb='git rebase'
alias grbi='git rbi'
alias grba='git rebase --abort'
alias grbc='git rebase --continue'
alias grbo='git rebase --onto'
alias grbs='git rebase --skip'
alias ga='git add'
alias gaa='git add --all'
alias gb='git branch'
alias gs='git status -sb'

alias glog="git log --graph --pretty='%Cred%h%Creset -%C(auto)%d%Creset %s %Cgreen(%ar) %C(bold blue)<%an>%Creset'"
alias gloga="git log --graph --pretty='%Cred%h%Creset -%C(auto)%d%Creset %s %Cgreen(%ar) %C(bold blue)<%an>%Creset' --all"
alias gp='git push'
alias 'gp!'='git push --force-with-lease --force-if-includes'
```

其中 `gcd` `gtl` 這類語法要 Zsh 才能啟用，Bash 需要寫函式完成。
