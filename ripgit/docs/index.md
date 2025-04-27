---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Git 零到一百"
  text: ""
  tagline: 完整、詳細且正確，唯一一個三者兼備的教學
  actions:
    - theme: brand
      text: 開始閱讀
      link: /basic/intro
    - theme: alt
      text: 遠端推送
      link: /core/remote-concept#remote-debug
    - theme: alt
      text: 客製化終端機
      link: /core/git-bash-setup-in-windows
    - theme: alt
      text: Github 技巧
      link: /pro/github-search
    - theme: alt
      text: 疑難排解
      link: /help/removing-sensitive-data
  image:
    src: https://cdn.zsl0621.cc/2025/docs/gemini-imagen-3-git-cover---2025-04-27T17-47-47.png
    alt: VitePress

features:
  - title: 內容完整
    details: 是否常常發現很多文章連基礎內容都不完整？本站從頭到尾完整教學，不再需要學到一半跑到其他網站查詢。
  - title: 詳細說明
    details: 文章內容詳細且分段安排妥當，一看就知道自己要到哪個段落找到資料，同時用字精煉，沒有廢話連篇也不和你閒話家常。
  - title: 資訊正確
    details: 不同於網路上不負責任的文章自創名詞甚至提供錯誤用法，此教學用語遵循文檔翻譯，指令用法絕對正確。
---

<br/>
<br/>

<div style="max-width: 960px; margin: 0 auto; padding: 0 1.5rem;">

# 網站介紹

自學 Git 最大的阻礙就是錯誤的網路資訊，於是本站基於 [Boost Your Git DX](https://adamj.eu/tech/2023/10/04/boost-your-git-dx-out-now/) 還有 [Pro Git](https://iissnan.com/progit/index.zh-tw.html) 完成，正確且完整是最大賣點，一站解決所有問題不用再去其他網站查資料，我認為不同程度的讀者都可以在本站找到有用的資訊：

1. 初學者請從[前言](/basic/intro)開始看。
2. 遇到遠端推送問題，請見我整理的[懶人包](/core/remote-concept#remote-debug)。
3. 遇到高階指令不會用，如 `rebase --onto` `force-if-includes` `sparse-checkout` `bisect` 等等，請見[超過一百分](/pro/intro)。
4. 想要客製化終端機體驗酷酷的終端機功能，請見[客製化教學](/core/git-bash-setup-in-windows)
5. 想玩轉 Github 榨乾他的所有功能，請見 [Github 技巧](/pro/github-search)。
6. 各式各樣的疑難雜症處理請見[日常問題](/help/daily-issues-local)。

</div>
