我的文檔網站，主站是由 [ouch1978.github.io](https://github.com/Ouch1978/ouch1978.github.io) 修改的 Docusaurus 網站，子站是以 Vitepress 搭建的 Git 教學。

## 安裝與初始化

```bash
git clone https://github.com/ZhenShuo2021/docs
cd docs
```

安裝依賴：

- 主站：`pnpm run install:main`
- 子站：`pnpm run install:git`

## 開發

- 主站：`pnpm run dev:main`
- 子站：`pnpm run dev:git`
- 或使用主入口：`pnpm start`（啟動主站）

## 建構網站

- 主站：`pnpm run build:main`
- 子站：`pnpm run build:git`
- 全站建構整合：`pnpm run build`（建構子站 → 執行整合腳本 → 建構主站）

## 預覽建構結果

- 主站：`pnpm run preview:main`
- 子站：`pnpm run preview:git`
- 整體預覽（主站 build 結果）：`pnpm run preview`

記得把個人訊息改成自己的，包含 git repo name, baseurl, Giscus, algolia 等等。修改時應該進入各自網站目錄執行比較不會搞混。

## 部屬到 Cloudflare Pages 方式

1. 進入設定頁面
   1. 登入 Cloudflare
   2. 點選左側 Workers and Pages
   3. 選擇建立
   4. 選擇 Pages
2. 設定部署網站
   1. 選擇儲存庫
   2. Framework Docusaurus
   3. 組建命令 `pnpm build`
   4. 組建輸出目錄 `main-site/build`
   5. 環境變數 `PNPM_VERSION` `NODE_VERSION` 選擇和本地一樣的版本
3. （可選）設定 custom domain，正常設定約兩分鐘內完成部屬

## 插入影片

使用 [react-player](https://github.com/cookpete/react-player) 完成，支援的影片來源和他一樣，或者放在 /static 資料夾中的影片。

```md
import VideoPlayer from '@site/src/components/VideoPlayer';

<VideoPlayer url="https://www.youtube.com/watch?v=<VIDEO_ID>" />
<VideoPlayer url="https://www.facebook.com/facebook/videos/<VIDEO_ID>/" />
<VideoPlayer url="./data/<FILE_NAME>" />
```

在同一 repo 的影片則使用此方式插入：

```md
import MyVideo from './data/my-video.mp4';

<figure>
  <video controls width="100%">
    <source src={MyVideo} type="video/mp4" />
    抱歉，您的瀏覽器不支援內嵌影片。
    您可以 <a href={MyVideo}>點此下載影片</a>。
  </video>
  <figcaption>影片說明，可選</figcaption>
</figure>
```

## 插入輪播圖片

使用 Embla 完成，範例如下，假設圖片放在 md 文件同層級的 data/img-n.webp 中

```md
import EmblaCarousel from '@site/src/components/EmblaCarousel';

import image1 from './data/img-1.webp';
import image2 from './data/img-2.webp';

<EmblaCarousel
  images={[image1, image2]}
  options={{ loop: true }}
/>
```
