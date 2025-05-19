---
title: Linux 指令完全制霸
id: about-linux-command
slug: /intro
sidebar_label: 簡介
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-13T23:50:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-13T23:50:00+08:00
---

指令會用是會用，但是細節參數永遠都記不起來，每次都要上網查很煩，網路上也沒有什麼系統性或是實務的教學，甚至連讓人方便查找的文檔都沒有（一股腦全部寫的不算），那就乾脆自己寫好了。

這些指令參數數量多到誇張，相信大家都有看過這個表格

| command | 1979 | 1996 | 2015 | 2017 |
| --- | --- | --- | --- | --- |
| ls | 11 | 42 | 58 | 58 |
| rm | 3 | 7 | 11 | 12 |
| mkdir | 0 | 4 | 6 | 7 |
| mv | 0 | 9 | 13 | 14 |
| cp | 0 | 18 | 30 | 32 |
| cat | 1 | 12 | 12 | 12 |
| pwd | 0 | 2 | 4 | 4 |
| chmod | 0 | 6 | 9 | 9 |
| echo | 1 | 4 | 5 | 5 |
| man | 5 | 16 | 39 | 40 |
| which |  | 0 | 1 | 1 |
| sudo |  | 0 | 23 | 25 |
| tar | 12 | 53 | 134 | 139 |
| touch | 1 | 9 | 11 | 11 |
| clear |  | 0 | 0 | 0 |
| find | 14 | 57 | 82 | 82 |
| ln | 0 | 11 | 15 | 16 |
| ps | 4 | 22 | 85 | 85 |
| ping |  | 12 | 12 | 29 |
| kill | 1 | 3 | 3 | 3 |
| ifconfig |  | 16 | 25 | 25 |
| chown | 0 | 6 | 15 | 15 |
| grep | 11 | 22 | 45 | 45 |
| tail | 1 | 7 | 12 | 13 |
| df | 0 | 10 | 17 | 18 |
| top |  | 6 | 12 | 14 |

> [The growth of command line options, 1979-Present](https://danluu.com/cli-complexity/)

這些參數數量已經多到我沒有耐心翻 man page，尤其是在任務明明很簡單卻要找老半天的情況下，尤其是 find, grep, rg, sed, awk, tr, cut, xargs 這些常用的文字處理。

只寫 bash 版本，因為有問題可以回退到 bash，但是 bash 不見得會有 zsh 可以用。
