---
title: 使用 Github Pages 架設網站
sidebar_label: 免費架設網站
slug: /github-pages
tags:
  - Git
  - Github
  - 教學
keywords:
  - Git
  - Github
  - 教學
last_update:
  date: 2025-04-06T23:48:30+08:00
  author: zsl0621
first_publish:
  date: 2025-04-06T23:48:30+08:00
---

榨乾免費工具的功能是一定要的，Github 除了提供免費的 Actions 功能，也提供 Pages 功能免費架設靜態網站，一個帳號只能架設一個 `帳號站點`，但是 `專案站點` 可以有多個，簡單來說就是 `YourName.github.io` 只能有一個 repo，但是你可以開多個 repo 建立 pages 功能，他們的網站路徑必續在 `YourName.github.io/path/` 之下。

> 以 Blowfish 為例，可以看到兩個專案站點 [Red Blowfish](https://nunocoracao.github.io/blowfish_artist/) 和 [Gray Blowfish](https://nunocoracao.github.io/blowfish_lite/) 不衝突。兩者的設定在此：[Red Blowfish](https://github.com/nunocoracao/blowfish_artist/blob/eef8f4cab0ed3bd7a07518570931f1c735f59b67/.github/workflows/pages.yml#L58) [Gray Blowfish](https://github.com/nunocoracao/blowfish_lite/blob/5a6ac3c331667d0a25aa1fbe6aba3aa18180dd6b/.github/workflows/pages.yml#L58)。

## Github Pages

設定方式網路上太多教學了，而且這偏向 GUI 操作我就不重複寫，請直接從 `1:23` 開始看 [Github王炸功能Pages，免费免服务器上线网站，详细教程](https://www.youtube.com/watch?v=YVj3JKMH9p8&t=83s)，看到 `2:45` 就結束了，就是這麼簡單又方便的功能。

## Cloudflare Pages

除了 Github 以外還有很多廠商也提供免費的靜態網站架設服務，比如說全世界最大的網路傷 Cloudflare 也提供 Pages 功能，支援直接連動 Github 倉庫，如果 Github Pages 強度王炸 Cloudflare Pages 就是王炸平方超級王炸，因為他支援無限數量的 pages，還可以無縫整合 `redirect` `workers` 等功能，甚至還支援隨時替換過去部署過的網頁，除錯或是回退都是一鍵搞定，除此之外你還可以享受到全球最大 CDN 服務商的速度、DNS 管理如 DNSSEC 等功能、DDoS 防護等功能這些其他雲端服務商都沒有。

### Cloudflare D1

靜態網頁的缺點就是無後端能力，所有內容都是靜態的，有沒有辦法解決這個問題呢？Cloudflare 可以。

Cloudflare D1 提供無伺服器的 SQL 資料庫服務，這代表可以在終端搭配 Workers 存取資料庫，不需自己架設一個後端伺服器，比如說你可以架一個流量統計，或者文章按讚、留言系統，又或者是把自己的 tcx 數據直接丟雲端搞一個進化版的 [running page](https://github.com/yihong0618/running_page)。

不過這當然是要有技術能力才弄的出來就是了。

### Cloudflare R2

把全部內容都放在儲存庫會造成 repo 容量迅速增加，該怎麼解決呢？

一般來說使用 Git LFS 功能可解決，但是對於網站架設我推薦 Cloudflare R2。R2 是物件儲存服務，最大的優勢在於零出口流量費（Zero egress fee），可以無痛地將靜態資源、圖片、備份資料儲存在 R2，並與 Cloudflare CDN 無縫串連，關於這問題請見下一篇文章[善用 Git LFS 功能減少儲存庫容量](reduce-size-using-git-lfs)。
