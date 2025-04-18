我的文檔網站，主站是由 [ouch1978.github.io](https://github.com/Ouch1978/ouch1978.github.io) 修改的 Docusaurus 網站，子站是以 Vitepress 搭建的 Git 教學。

## 安裝與初始化

```bash
git clone https://github.com/ZhenShuo2021/docs
cd docs
```

|            | 主站（Docusaurus）      | 子站（Vitepress）          | 全站整合     |
|------------|------------------------|--------------------------|--------|
| 安裝依賴    | `pnpm run install:main` | `pnpm run install:git`   | –  |
| 開發       | `pnpm run dev:main`     | `pnpm run dev:git`       | `pnpm start`（等同 dev:main）   |
| 建構網站    | `pnpm run build:main`   | `pnpm run build:git`     | `pnpm run build`（子站 → 整合 → 主站） |
| 預覽建構結果 | `pnpm run preview:main` | `pnpm run preview:git`   | `pnpm run preview`（預覽主站輸出） |
| 建立文章列表 | `pnpm run new`          | –  | –  |
| 清除快取    | `pnpm run clear:main`   | `pnpm run clear:git`      | - |

> ⚠️ 如果使用 `pnpm run new` 建立首頁文章列表的內容異常，請使用 `clear:main` 清除快取並重新操作。

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

接下來是介紹也是備忘錄，因為我自己也記不起來怎麼用所以寫在這。

## Docusaurus 相關操作

### 插入影片

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

### 插入輪播圖片

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

## 掃描 Broken Links

除了 Docusaurus 和 Vitepress 內建的偵測以外還有兩種工具可以掃描：node 的 [linkinator](https://github.com/JustinBeckwith/linkinator) 和 Python 的 [linkchecker](https://github.com/linkchecker/linkchecker)。

### linkinator

安裝

```sh
npm install -g linkinator
```

掃描

```sh
npx linkinator http://127.0.0.1:8080 -r -s assets -s github -s wiki --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36' --verbosity error
```

### linkchecker

使用 uv 不需安裝，掃描指令

```sh
uvx linkchecker http://127.0.0.1:8080 --ignore-url=/tags/ --user-agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36' -t 50
```
