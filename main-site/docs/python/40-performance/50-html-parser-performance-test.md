---
title: HTML 解析器效能測試：BeautifulSoup4、LXML 和 Selectolax
sidebar_label: HTML 解析器效能測試
slug: /html-parser-performance-test
tags:
  - Python
  - HTML parser
keywords:
  - Python
  - HTML parser
last_update:
  date: 2025-03-27T17:52:00+08:00
  author: zsl0621
first_publish:
  date: 2024-12-24T21:53:30+08:00
---

這篇評測比較了 Python 中流行的 HTML 解析器庫的性能。測試內容包括從 Shopee 頁面提取產品名稱、評分、價格和月銷量。為了模擬處理超大 HTML 文件的情境，`duplicate_div_main` 函數將 `<div id="main">` 區塊重複了 100 次，從而創建了一個顯著更大的文件。

測試程式碼可以在我的 [repo](https://github.com/ZhenShuo2021/blog-script/tree/main/python/html-parser) 中找到。

### 結果

測試環境：

- 硬體：M1 MacBook Pro，8GB RAM
- 測試案例：Shopee 產品頁面解析
- Package version:
  - beautifulsoup4==4.12.3
  - cssselect==1.2.0
  - lxml==5.3.0
  - selectolax==0.3.27

```none
開始解析原始 HTML...
原始 HTML 執行時間（秒）：
--------------------------------------------------
selectolax     : 0.005336 (1.00x)
lxml           : 0.023221 (4.35x)
lxml_css       : 0.045544 (8.53x)
bs4            : 0.127979 (23.98x)

開始解析大 HTML...
大 HTML 執行時間（秒）：
--------------------------------------------------
selectolax     : 1.160970 (1.00x)
lxml           : 2.236277 (1.93x)
lxml_css       : 4.248061 (3.66x)
bs4            : 71.444133 (61.54x)
```

### 結論

lxml 表現出色，對大多數使用情境來說已經足夠。

Selectolax 的速度大約是 lxml 的 4 倍，但它也有一些限制。缺少對 XPath 的支持是其主要缺點，並且在某些 CSS 語法上也存在問題，例如不支持包含括號或斜線的類名，這在現代網站中越來越常見。這些限制可能會影響它在複雜場景中的使用。

> 注意：[html5-parser](https://github.com/kovidgoyal/html5-parser) 被排除在外，因為它的受歡迎程度和維護狀況較差。
