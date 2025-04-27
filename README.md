結合兩種靜態網站產生器的文檔網站。主站是基於 [ouch1978.github.io](https://github.com/Ouch1978/ouch1978.github.io) 修改而來的 Docusaurus 網站，用於主要內容，子站則是以 Vitepress 搭建的 Git 教學文檔。

## 使用說明

```bash
git clone https://github.com/ZhenShuo2021/docs --single-branch
cd docs
```

### 主站

1. 安裝依賴 `pnpm run install:main`
2. 開發 `pnpm run dev:main`，同 `pnpm start`
3. 建構網站 `pnpm run build:main`
4. 預覽建構 `pnpm run preview:main`
5. 清除快取 `pnpm run clear:main`
6. 建立首頁文章列表: `pnpm run new`

> ⚠️ **注意：** 如果使用 `pnpm run new` 建立的首頁文章列表內容出現錯誤，則需要用 `pnpm run clear:main` 清除主站快取。

### 子站

1. 安裝依賴 `pnpm run install:git`
2. 開發 `pnpm run dev:git`
3. 建構網站 `pnpm run build:git`
4. 預覽建構 `pnpm run preview:git`
5. 清除快取 `pnpm run clear:git`

### 全站

1. 建構 (先建構子站，再整合至主站，最後建構主站): `pnpm run build`
2. 預覽建構 `pnpm run preview`

記得把個人訊息改成自己的，包含 git repo name, baseurl, Giscus, algolia 等等。

## 部屬到 Cloudflare Pages 方式

1. 進入設定頁面
   1. 登入 Cloudflare
   2. 點選左側 Workers and Pages
   3. 選擇建立
   4. 選擇 Pages
2. 設定部署網站
   1. 選擇儲存庫
   2. 組建命令 `pnpm build`
   3. 組建輸出目錄 `main-site/build`
   4. 環境變數 `PNPM_VERSION` `NODE_VERSION` 選擇和本地一樣的版本
3. （可選）設定 custom domain，正常設定約兩分鐘內完成部屬

接下來是介紹也是備忘錄，因為我自己也記不起來怎麼用所以寫在這。

## Docusaurus 相關操作

### 插入影片

使用 [react-player](https://github.com/cookpete/react-player) 完成，支援的影片來源和他一樣，或者放在 /static 資料夾中的影片。

```md
import VideoPlayer from '@site/src/components/VideoPlayer';

<VideoPlayer url="https://www.youtube.com/watch?v=<VIDEO_ID>" />
<VideoPlayer url="https://www.facebook.com/facebook/videos/<VIDEO_ID>/" />
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

使用 Embla 完成，接受 URL、static 和同層級目錄的檔案，並且支援一些自訂參數。

```md
import circleLogo from './logo_circle.webp';

<EmblaCarousel 
  images={[
    'https://picsum.photos/600/400?random=99',
    'https://picsum.photos/600/400?random=100',
    '/img/logo.png',
    circleLogo
  ]}
  width="70%"
  captions={[
    'foo',
    'bar',
    ''
  ]}
  options={{
    loop: false,
    watchDrag: false,
    duration: 18,
    containScroll: false
  }}
/>
```

## 掃描 Broken Links

除了 Docusaurus 和 Vitepress 內建的連結檢查功能，還可以用 node 的 [linkinator](https://github.com/JustinBeckwith/linkinator) 和 Python 的 [linkchecker](https://github.com/linkchecker/linkchecker) 掃描。

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
