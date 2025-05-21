---
title: 在 Mac 上以 VS Code 編輯 LaTeX
tags:
  - LaTeX
  - VS Code
keywords:
  - LaTeX
  - VS Code
last_update:
  date: 2025-05-21T15:50:33+08:00
  author: zsl0621
first_publish:
  date: 2025-05-21T15:50:33+08:00
---

本文基於以下幾篇文章完成，由於我只需要 Paper 和 Book 兩種版型所以省略了很多步驟，如果需要完整版教學的請看原文：

- [LaTeX in Visual Studio Code (VSCode) on MacOS](https://daangeijs.nl/posts/LaTeX-vscode)
- [xeLaTeX 以及 LaTeXmk 命令行编译](https://zhuanlan.zhihu.com/p/256370737)
- [科研必会——在Mac上配置LaTeX写作环境](https://zhuanlan.zhihu.com/p/560361957)

以及 Windows 系統的用法，由於 Docker 模擬 Linux 環境，所以應該是比較不容易出問題的方案

- [LaTeX Workshop – 在VSCode中編輯及編譯LaTeX](https://shaynechen.gitlab.io/vscode-LaTeX/)
- [使用VSCode上撰寫中文LaTeX文件](https://kaibaoom.tw/posts/notes/vscode-LaTeX/)
- [Building LaTeX projects on Windows easily with Docker](https://andrewlock.net/building-LaTeX-projects-on-windows-easily-with-docker/)

## 應該本地編輯還是使用 Overleaf？

是否該在本地編輯 LaTeX？我分為以下幾點考量

在編譯時間上：

- 包含 tikz 並且文章嵌入 PDF 的文件
- 在 M1 Mac 上編譯 70 頁編譯耗時約 7.5 秒
- 在 Overleaf 上編譯時間為 10 秒
- 在 Overleaf 上會不斷被警告編譯超時
- 如果電腦比 M1 爛可以考慮直接在 Overleaf 上編輯

在容量問題上：

- 在容量寸土寸金的 Mac 上 MacTeX 需要 6GB 的儲存空間

在錯誤處理上：

- Overleaf 幫你處理了常見問題，如 build command, 字體等等
- 在本地需要自行解決

這三點是在 Overleaf 還有本地編輯的主要考量。

## 安裝

需要安裝以下幾項：

1. brew 套件管理器
2. MacTeX，LaTeX 本身
3. VS Code，文字編輯器，因為 MacTeX 編輯器醜到不行
4. VS Code 的 LaTeX 插件 LaTeX Workshop
5. 安裝標楷體

### 安裝 Brew

先安裝套件管理器，後續所有安裝都使用此管理器。

```sh
# 安裝
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 刷新 shell
exec zsh

# 檢查
brew --version

# 如果沒有找到 brew，需要把 brew 加入系統路徑
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc

# 再次刷新 shell
exec zsh
```

網路教學會把加入系統路徑的指令改為 `echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> ~/.zprofile`，兩者差別為前者只會加入系統路徑，後者除了系統路徑還會加上補全系統，這會導致終端機啟動速度變慢，依照個人需求選擇。

### 安裝 LaTeX

MacTeX 容量 6GB，他的伺服器網速又很慢，安裝時可以先去滑手機。

```sh
# 安裝 MacTeX
brew install --cask MacTeX-no-gui

# 安裝完成後重新開啟終端機

# 更新 LaTeX packages
sudo tlmgr update --self
sudo tlmgr update --all
```

### 安裝 VS Code

只需要一行就可完成。

```sh
brew install --cask visual-studio-code
```

### 安裝 VS Code LaTeX 插件

點擊此連結安裝 [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.LaTeX-workshop)。

### 安裝標楷體

Windows 和 Mac 的標楷體雖然名稱一樣但是顯示效果不一樣，這大家應該都知道，如果沒有和 Windows 系統一樣的標楷體，換到 Windows 開就一定會跑版，解決方法是**請找到符合著作權法的 ttf 檔案替換**。

## 設定

設定包含 LaTeX 本身的設定以及在 Mac 一定會遇到的字體問題。

### 設定 LaTeX 編譯

1. 複製以下文本

```txt
Preferences: Open User Settings (JSON)
```

2. 打開 VS Code，按下 `command + shift + p`，貼上剛才複製的文本，enter 進入。
3. 貼上以下設定檔

```json
{
    "LaTeX-workshop.LaTeX.tools": [
    {
      "name": "xeLaTeX",
      "command": "/Library/TeX/TeXbin/xeLaTeX",
      "args": [
        "-syncTeX=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ]
    },
    {
      "name": "bibTeX",
      "command": "/Library/TeX/TeXbin/bibTeX",
      "args": [
        "%DOCFILE%"
      ]
    },
    {
      "name": "LaTeXmk-xeLaTeX",
      "command": "/Library/TeX/TeXbin/LaTeXmk",
      "args": [
        "-xeLaTeX",
        "-syncTeX=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-outdir=%OUTDIR%",
        "%DOC%"
      ]
    }
  ],
  "LaTeX-workshop.LaTeX.recipes": [
    {
      "name": "XeLaTeX ➞ bibTeX ➞ XeLaTeX×2",
      "tools": [
        "xeLaTeX",
        "bibTeX",
        "xeLaTeX",
        "xeLaTeX"
      ]
    },
    {
      "name": "LaTeXmk 🔃",
      "tools": [
        "LaTeXmk-xeLaTeX"
      ]
    }
  ],
  "LaTeX-workshop.view.pdf.viewer": "tab",
  "LaTeX-workshop.LaTeX.autoBuild.run": "onSave",
  "LaTeX-workshop.LaTeX.autoClean.run": "onFailed",
  "LaTeX-workshop.LaTeX.clean.fileTypes": [
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.idx",
    "*.ind",
    "*.lof",
    "*.lot",
    "*.out",
    "*.toc",
    "*.acn",
    "*.acr",
    "*.alg",
    "*.glg",
    "*.glo",
    "*.gls",
    "*.fls",
    "*.log",
    "*.fdb_LaTeXmk",
    "*.snm",
    "*.syncTeX(busy)",
    "*.syncTeX.gz(busy)",
    "*.nav"
  ],
  "LaTeX-workshop.LaTeX.recipe.default": "lastUsed",
  "LaTeX-workshop.view.pdf.internal.syncTeX.keybinding": "double-click",
  "editor.unicodeHighlight.allowedLocales": {
    "zh-hans": true,
    "zh-hant": true
  },
  "[LaTeX]": {
    "editor.defaultFormatter": "James-Yu.LaTeX-workshop"
  }
}
```

其中 `LaTeX-workshop.LaTeX.recipes` 是目錄，平常使用就是選擇以哪個 recipe 進行編譯，而 `LaTeX-workshop.LaTeX.tools` 代表的是該 recipe 執行的具體方式。

完整版本請見 [科研必会——在Mac上配置LaTeX写作环境](https://zhuanlan.zhihu.com/p/560361957)，我用不到那麼多所以只留下必要的，兩個分別代表

1. **XeLaTeX ➞ bibTeX ➞ XeLaTeX×2**: 傳統編譯方式，需要先編譯 XeLaTeX，再回來編 bibTeX，最後又要整合，耗時約 20 秒。
2. **LaTeXmk**: 整合上述步驟，執行時間只需要 7.5 秒。

### 設定字體

以我的 LaTeX 文件來說，這是我原始在 Overleaf 的設定，需要直接把 BiauKai.ttf 丟上去

```TeX
\setCJKmainfont[AutoFakeBold=6,AutoFakeSlant=.4]{[BiauKai.ttf]}
\defaultCJKfontfeatures{AutoFakeBold=6,AutoFakeSlant=.4}
\newCJKfontfamily\Kai{[BiauKai.ttf]}       	%定義指令\Kai則切換成標楷體
```

現在改為本地編譯，直接使用系統 ttf 檔案，改為以下

```TeX
\setCJKmainfont[AutoFakeBold=6,AutoFakeSlant=.4]{DFKai-SB}
\defaultCJKfontfeatures{AutoFakeBold=6,AutoFakeSlant=.4}
\newCJKfontfamily\Kai{DFKai-SB}
```

其中 `DFKai-SB` 這個字串是 ttf 檔案裡面設定的字體名稱，使用以下指令檢查你的名稱是否也是 `DFKai-SB`：

```sh
# 安裝字體套件
brew install fontconfig

# 找到 kaiu 檔案
fc-list | grep "/Library/Fonts/kaiu.ttf"

>>> /Library/Fonts/kaiu.ttf: DFKai\-SB,標楷體:style=Regular
```

此輸出代表 `DFKai-SB` 和 `標楷體` 都是字體名稱。

### 設定 VS Code 換行

由於 LaTeX 通常一行很長，所以要設定自動換行 (wrap)，如果目錄中沒有 `.vscode/settings.json`，直接在終端機貼上

```sh
mkdir -p .vscode
cat > .vscode/settings.json <<EOF
{
  "editor.wordWrap": "on"
}
EOF
```

或者手動開啟文件加入

```txt
"editor.wordWrap": "on"
```

## 版本控制

強烈建議使用 Git 進行版本控制，這是非常優秀的版本控制系統，即使你的 Git 只聽過沒用過，只會最基本的新增版本記錄都比不用來的好。

這是 LaTeX 專案使用的 `.gitignore` 設定檔：

<details>

<summary>.gitignore</summary>

來自 https://github.com/github/gitignore/blob/main/TeX.gitignore

```ini
## Core latex/pdflatex auxiliary files:
*.aux
*.lof
*.log
*.lot
*.fls
*.out
*.toc
*.fmt
*.fot
*.cb
*.cb2
.*.lb

## Intermediate documents:
*.dvi
*.xdv
*-converted-to.*
# these rules might exclude image files for figures etc.
# *.ps
# *.eps
# *.pdf

## Generated if empty string is given at "Please type another file name for output:"
.pdf

## Bibliography auxiliary files (bibtex/biblatex/biber):
*.bbl
*.bbl-SAVE-ERROR
*.bcf
*.bcf-SAVE-ERROR
*.blg
*-blx.aux
*-blx.bib
*.run.xml

## Build tool auxiliary files:
*.fdb_latexmk
*.synctex
*.synctex(busy)
*.synctex.gz
*.synctex.gz(busy)
*.pdfsync
*.rubbercache
rubber.cache

## Build tool directories for auxiliary files
# latexrun
latex.out/

## Auxiliary and intermediate files from other packages:
# algorithms
*.alg
*.loa

# achemso
acs-*.bib

# amsthm
*.thm

# attachfile2
*.atfi

# beamer
*.nav
*.pre
*.snm
*.vrb

# changes
*.soc
*.loc

# comment
*.cut

# cprotect
*.cpt

# elsarticle (documentclass of Elsevier journals)
*.spl

# endnotes
*.ent

# fixme
*.lox

# feynmf/feynmp
*.mf
*.mp
*.t[1-9]
*.t[1-9][0-9]
*.tfm

#(r)(e)ledmac/(r)(e)ledpar
*.end
*.?end
*.[1-9]
*.[1-9][0-9]
*.[1-9][0-9][0-9]
*.[1-9]R
*.[1-9][0-9]R
*.[1-9][0-9][0-9]R
*.eledsec[1-9]
*.eledsec[1-9]R
*.eledsec[1-9][0-9]
*.eledsec[1-9][0-9]R
*.eledsec[1-9][0-9][0-9]
*.eledsec[1-9][0-9][0-9]R

# glossaries
*.acn
*.acr
*.glg
*.glo
*.gls
*.glsdefs
*.lzo
*.lzs
*.slg
*.slo
*.sls

# uncomment this for glossaries-extra (will ignore makeindex's style files!)
# *.ist

# gnuplot
*.gnuplot
*.table

# gnuplottex
*-gnuplottex-*

# gregoriotex
*.gaux
*.glog
*.gtex

# htlatex
*.4ct
*.4tc
*.idv
*.lg
*.trc
*.xref

# hypdoc
*.hd

# hyperref
*.brf

# knitr
*-concordance.tex
# TODO Uncomment the next line if you use knitr and want to ignore its generated tikz files
# *.tikz
*-tikzDictionary

# listings
*.lol

# luatexja-ruby
*.ltjruby

# makeidx
*.idx
*.ilg
*.ind

# minitoc
*.maf
*.mlf
*.mlt
*.mtc[0-9]*
*.slf[0-9]*
*.slt[0-9]*
*.stc[0-9]*

# minted
_minted*
*.data.minted
*.pyg

# morewrites
*.mw

# newpax
*.newpax

# nomencl
*.nlg
*.nlo
*.nls

# pax
*.pax

# pdfpcnotes
*.pdfpc

# sagetex
*.sagetex.sage
*.sagetex.py
*.sagetex.scmd

# scrwfile
*.wrt

# svg
svg-inkscape/

# sympy
*.sout
*.sympy
sympy-plots-for-*.tex/

# pdfcomment
*.upa
*.upb

# pythontex
*.pytxcode
pythontex-files-*/

# tcolorbox
*.listing

# thmtools
*.loe

# TikZ & PGF
*.dpth
*.md5
*.auxlock

# titletoc
*.ptc

# todonotes
*.tdo

# vhistory
*.hst
*.ver

# easy-todo
*.lod

# xcolor
*.xcp

# xmpincl
*.xmpi

# xindy
*.xdy

# xypic precompiled matrices and outlines
*.xyc
*.xyd

# endfloat
*.ttt
*.fff

# Latexian
TSWLatexianTemp*

## Editors:
# WinEdt
*.bak
*.sav

# Texpad
.texpadtmp

# LyX
*.lyx~

# Kile
*.backup

# gummi
.*.swp

# KBibTeX
*~[0-9]*

# TeXnicCenter
*.tps

# auto folder when using emacs and auctex
./auto/*
*.el

# expex forward references with \gathertags
*-tags.tex

# standalone packages
*.sta

# Makeindex log files
*.lpz

# xwatermark package
*.xwm

# REVTeX puts footnotes in the bibliography by default, unless the nofootinbib
# option is specified. Footnotes are the stored in a file with suffix Notes.bib.
# Uncomment the next line to have this generated file ignored.
#*Notes.bib
```

</details>

## 協作

由於我不需要協作所以沒有用這個方案，請看原文 [Syncing VSCode with Overleaf for Collaboration and Reviews](https://daangeijs.nl/posts/latex-vscode/#step-6-syncing-vscode-with-overleaf-for-collaboration-and-reviews)。

## FAQ

寫給我自己的常見問題，都是問 AI 的

1. TeX LaTeX kaTeX差在哪，繁體中文回答
   - TeX 是底層語言，我們不會直接用他來寫文件，輸出格式為 DVI
   - LaTeX 是基於 TeX 的巨集包，輸出格式還是 DVI
   - KaTeX 是一種專為網頁設計的數學排版程式庫
2. LaTeX XeTeX XeLaTeX pdfTeX LuaTeX 差在哪
   - LaTeX 與原始 TeX 一樣，對中文等非拉丁文字的支援有限
   - pdfTeX 是 TeX 的改進版本，可直接輸出 PDF 檔案，也改進了字距調整和連字功能，但是字元支援仍然有限
   - XeTeX 和 XeLaTeX
     - XeTeX 是現代 TeX 引擎
     - XeLaTeX 是使用 XeTeX 引擎的 LaTeX 格式
     - 原生支援 Unicode 和 UTF-8 編碼，非常適合中文等多語言排版
   - LuaTeX 最新的 TeX 引擎，整合了 Lua 腳本語言，完全支援 Unicode 和 UTF-8 編碼
3. 加速編譯  
  問 AI 只會給廢話，看起來比較有用的方式是
   1. 指定字型，不要每次都搜尋整個系統
   2. 分塊編譯，比如每次都只編譯特定 section 而不是整本書
