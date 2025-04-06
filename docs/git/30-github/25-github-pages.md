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

身為客家超人，榨乾任何一個免費工具的功能是必要的，Github 除了 Actions 自動化執行之外，另外一個免費功能就是 Pages，允許我們免費架設靜態網站，一個帳號只能架設一個 `帳號站點`，但是 `專案站點` 可以有多個，意思就是 `YourName.github.io` 只能有一個 repo，但是你可以開多個 repo 建立 pages 功能，他們的路徑必續在 `YourName.github.io/path/` 之下。

> 以 Blowfish 為例，可以看到兩個專案站點 [Red Blowfish](https://nunocoracao.github.io/blowfish_artist/) 和 [Gray Blowfish](https://nunocoracao.github.io/blowfish_lite/) 不衝突。兩者的設定在此：[Red Blowfish](https://github.com/nunocoracao/blowfish_artist/blob/eef8f4cab0ed3bd7a07518570931f1c735f59b67/.github/workflows/pages.yml#L58) [Gray Blowfish](https://github.com/nunocoracao/blowfish_lite/blob/5a6ac3c331667d0a25aa1fbe6aba3aa18180dd6b/.github/workflows/pages.yml#L58)。

## Github Pages

設定方式網路上太多教學了，而且這偏向 GUI 操作我就不重複寫，請直接從 `1:23` 開始看 [Github王炸功能Pages，免费免服务器上线网站，详细教程](https://www.youtube.com/watch?v=YVj3JKMH9p8&t=83s)，看到 `2:45` 就結束了，就是這麼簡單又方便的功能。

也可以用[我的網站](https://github.com/ZhenShuo2021/ZhenShuo2021.github.io)測試，一樣是 fork 後進行設定，但是已經有 workflow file 不用設定，fork 後點擊 `Actions`，選擇左側的 `Github Pages`，再點擊右側的 `Run workflow` 應該就可以了。平常不會手動觸發，每次 push 都會觸發網站部署，文字教學網路上超多就不再重複撰寫。

## Cloudflare Pages

Cloudflare 也提供 Pages 功能，支援直接連動 Github 倉庫，簡單來說就是王炸平方超級王炸，因為他支援高達 100 個 pages，你還可以在上面設定 `redirect` `workers` 等功能，甚至還支援隨時替換過去部署過的網頁，除錯或是回退都是一鍵搞定，除此之外你還可以享受到全球最大 CDN 服務商的速度、DNS 管理如 DNSSEC 等功能、DDoS 防護等功能這些其他雲端服務商都沒有，唯一的缺點是中國大陸連 Cloudflare 速度很慢。

## Cloudflare D1

Cloudflare D1 提供無伺服器的 SQL 資料庫服務，這代表可以在 Edge 端搭配 Workers 直接存取資料庫，不需自己架設一個後端伺服器。比如說你可以架一個流量統計，或者文章按讚、留言系統，又或者是把自己的 [tcx/gpx](https://support.strava.com/hc/zh-tw/articles/216918437-%E5%8C%AF%E5%87%BA%E8%B3%87%E6%96%99%E8%88%87%E5%A4%A7%E9%87%8F%E5%8C%AF%E5%87%BA) 數據直接丟雲端，就可以免費搞一個進化版的 [running page](https://github.com/yihong0618/running_page)。

不過這當然是要有技術能力才弄的出來就是了。

## Cloudflare R2

Cloudflare R2 是物件儲存服務，最大的優勢在於零出口流量費（Zero egress fee），可以無痛地將靜態資源、圖片、備份資料儲存在 R2，並與 Cloudflare CDN 無縫串連，[R2 Pricing Calculator](https://r2-calculator.cloudflare.com/) 測試計費如下

- 10 GB 容量，每月操作次數寫一百萬，讀一千萬次**免費**
- 20 GB 容量，每月操作次數寫一百萬，讀一千萬次**0.15 美金**
- 50 GB 容量，每月操作次數寫一百萬，讀一千萬次**0.60 美金**
- 1000 GB 容量，每月操作次數寫一百萬，讀一千萬次**15 美金**

使用方式則是在 [架設Cloudflare R2免費圖床，給Hugo靜態網站託管圖片](https://ivonblog.com/posts/cloudflare-r2-image-hosting/) 有說明。
