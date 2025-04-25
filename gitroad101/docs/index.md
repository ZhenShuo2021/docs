---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Git 零到一百"
  text: ""
  tagline: 完整、詳細且正確，全網唯一一個三者兼備的教學
  actions:
    - theme: brand
      text: 開始閱讀
      link: /beginner/intro
    - theme: alt
      text: 遠端推送
      link: /intermediate/remote-concept#remote-debug
    - theme: alt
      text: 客製化終端機
      link: /intermediate/git-bash-setup-in-windows
    - theme: alt
      text: Github 技巧
      link: /advance/github-search
    - theme: alt
      text: 疑難排解
      link: /troubleshooting/removing-sensitive-data
  image:
    src: /gemini-imagen-3-git-cover.png
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

網路上到處都是錯誤資訊，甚至連已經出書賣網路課程的都寫錯，本站基於 [Boost Your Git DX](https://adamj.eu/tech/2023/10/04/boost-your-git-dx-out-now/) 還有 [Pro Git](https://iissnan.com/progit/index.zh-tw.html) 完成，正確是本教學最大賣點，絕非網路上不負責任傳達錯誤資訊的文章。

我認為不同程度的讀者都可以在本站找到有用的資訊：

1. 初學者請從[前言](/beginner/intro)開始看。
2. 遇到遠端推送問題，請見我整理的[懶人包](/intermediate/remote-concept#remote-debug)。
3. 遇到高階指令不會用，如 `rebase --onto` `force-if-includes` `sparse-checkout` `bisect` 等等，請見[超過一百分](/advance/intro)。
4. 想要客製化終端機，體驗網路上那些酷酷的終端機功能，請見[客製化教學](/intermediate/git-bash-setup-in-windows)
5. 想要玩轉 Github 榨乾他的所有功能，請見 [Github 技巧](/advance/github-search)。
6. 各式各樣的疑難雜症處理請見[日常問題](/troubleshooting/daily-issues-local)。

</div>
