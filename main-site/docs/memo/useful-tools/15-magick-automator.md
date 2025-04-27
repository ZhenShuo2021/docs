---
title: 使用 automator 簡化日常工作流程
sidebar_label: 使用 automator 簡化流程
tags:
  - 實用工具
  - 教學
keywords:
  - 實用工具
  - 教學
last_update:
  date: 2025-04-06T23:33:30+08:00
  author: zsl0621
first_publish:
  date: 2025-04-06T23:33:30+08:00
---

import VideoPlayer from '@site/src/components/VideoPlayer';

寫腳本的目的就是簡化重複的操作，大家都是宅宅工程師不需要再說廢話，這裡介紹我怎麼使用 automator 的。

每次寫文章時要把圖片壓縮避免上傳大檔案到 Git 裡面，以宅宅工程師的方式就是

1. 打開終端
2. 輸入 magick
3. 上鍵找到之前用的指令
4. 把路徑改成這次的檔案路徑
5. 最後再把圖片移動到專案目錄裡面

要打一堆字就算了還要在滑鼠跟鍵盤之間來回移動麻煩死了！於是看向 automator，用了他之後原本五個步驟的流程現在只要這樣

<VideoPlayer url="https://cdn.zsl0621.cc/2025/docs/automator-magick---2025-04-27T16-50-05.mp4" />

讚讚讚，而且支援轉換一整個文件夾，這篇既是教學也是記錄，免的連自己都忘記要怎麼用。

## 設定

1. 打開 Automator
2. 螢幕上方 bar 選擇「檔案」>「新增」
3. 選擇「應用程式」
4. 左側程式庫中選擇 finder 圖示的「檔案和檔案夾」
5. 貼上「取得指定的」，雙擊「取得指定的Finder項目」
6. 左側程式庫中選擇「工具程式」
7. 貼上「Shell」，雙擊「執行Shell工序指令」
8. 左上角 shell 選擇「/bin/bash」，**右上角傳遞輸入選擇「作為引數使用」，這個步驟超級重要！！！**
9. 文字框中貼上腳本

```sh
#!/bin/bash

LOGFILE="/tmp/magick_conversion.log"
echo "=== 開始轉換 `date` ===" >> "$LOGFILE"

for input in "$@"; do
    if [[ -f "$input" ]]; then
        # 處理單個檔案
        dir=$(dirname "$input")
        filename=$(basename "$input")
        name="${filename%.*}"
        output="$dir/$name.webp"

        echo "轉換 $input → $output" >> "$LOGFILE"
        /opt/homebrew/bin/magick "$input" -sampling-factor 4:2:0 -strip -quality 80 "$output" >> "$LOGFILE" 2>&1

        if [[ $? -eq 0 ]]; then
            echo "成功: $output" >> "$LOGFILE"
        else
            echo "失敗: $input" >> "$LOGFILE"
        fi

    elif [[ -d "$input" ]]; then
        # 處理資料夾
        parent_dir=$(dirname "$input")
        folder_name=$(basename "$input")
        output_dir="$parent_dir/${folder_name}_webp"

        mkdir -p "$output_dir"
        echo "處理資料夾 $input → $output_dir" >> "$LOGFILE"

        find "$input" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" \) | while read file; do
            rel_path=${file#"$input/"}
            output_file="$output_dir/${rel_path%.*}.webp"
            output_subdir=$(dirname "$output_file")

            mkdir -p "$output_subdir"
            echo "轉換 $file → $output_file" >> "$LOGFILE"
            /opt/homebrew/bin/magick "$file" -sampling-factor 4:2:0 -strip -quality 80 "$output_file" >> "$LOGFILE" 2>&1

            if [[ $? -eq 0 ]]; then
                echo "成功: $output_file" >> "$LOGFILE"
            else
                echo "失敗: $file" >> "$LOGFILE"
            fi
        done
    fi
done

echo "=== 轉換結束 `date` ===" >> "$LOGFILE"
```

10. 儲存檔案
11. 按住 Command 拖曳程式到 Finder 上方
12. （可選）到 [macosicons](https://macosicons.com/#/) 挑一個喜歡的圖示下載
13. （可選）回到剛才製作的應用程式中，右鍵取得資訊，把下載的檔案拖曳到機器人圖示替換
14. 大功告成！
