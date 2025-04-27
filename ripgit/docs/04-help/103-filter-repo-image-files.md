---
title: 使用 filter-repo 移除大型檔案
slug: /filter-repo-image-files
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-27T17:34:07+08:00
  author: zsl0621
first_publish:
  date: 2025-04-27T17:34:07+08:00
---

# {{ $frontmatter.title }}

最近把網站圖片改為 CDN 部署以減小儲存庫容量，因此順便用了一下 filter-repo 功能來清除大型檔案，廢話不多說直接上步驟

1. 請使用全新克隆的儲存庫操作保證安全
2. 使用你喜歡的方式[安裝 filter-repo](https://github.com/newren/git-filter-repo/blob/main/INSTALL.md)
3. 輸入指令就完成了

```sh
git filter-repo --path-glob '*.jpg' --path-glob '*.jpeg' --path-glob '*.png' --path-glob '*.webp' --path-glob '*.gif' --path-glob '*.bmp' --path-glob '*.tiff' --path-glob '*.tif' --path-glob '*.svg' --path-glob '*.ico' --path-glob '*.heic' --path-glob '*.heif' --path-glob '*.mp4' --path-glob '*.mp4' --invert-paths --refs bak --force
```

也就是找出圖片檔移除，使用 `--refs` 限制分支，`--force` 強制執行。也可以依照檔案容量過濾，請見[文檔](https://htmlpreview.github.io/?https://github.com/newren/git-filter-repo/blob/docs/html/git-filter-repo.html)裡面的 `--strip-blobs-bigger-than`。
