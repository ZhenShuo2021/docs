---
title: GitHub Actions 複雜工作流：從自動同步到建立預覽、發佈版本
sidebar_label: Github Actions 自動化部署發佈
slug: /complex-github-actions
tags:
  - Git
  - Github
  - 教學
keywords:
  - Git
  - Github
  - 教學
last_update:
  date: 2025-05-11T18:32:30+08:00
  author: zsl0621
first_publish:
  date: 2025-05-11T18:32:30+08:00
---

我們在 [Github Actions 自動化 CI/CD](github-actions) 展示了多種使用 Github Actions 的範例，不過都侷限在單一的簡單任務，本文則提供一個複雜範例，長達 250 行的自動化腳本，目的是移除 [blowfish](https://github.com/nunocoracao/blowfish) 專案的非必要檔案並且自動發佈 core 版本，具體目標如下：

1. 以 crontab 定時檢查更新
2. 確認遠端更新後自動執行腳本更新 repo，如果這兩項任務都成功就繼續完成以下任務
   1. 自動提交，建立新分支，並且發佈 PR
   3. 建立構建預覽
   4. 發佈預覽到網路 (Cloudflare Pages)
   5. 自動建立 release

本文教你如何成為乞丐超人，自動化完成這些任務而且不花一毛錢，專案在 [blowfish-core](https://github.com/ZhenShuo2021/blowfish-core)，PR 執行範例在 [#1](https://github.com/ZhenShuo2021/blowfish-core/pull/1)。

> 1. 本文完成於 2025/5，請注意文章期限。
>
> 2. 由於這是鏡像儲存庫所以我不想要放太多我的程式在上面，而且希望對程式有**完全掌控權**，並且可以**動態設定環境變數**，所以沒有用 [GitHub Integration](https://developers.cloudflare.com/pages/configuration/git-integration/github-integration/) 而是採用 [Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/) 方式部署到 Cloudflare Pages，如果沒有特殊要求，使用 integration 方式應該更好設定。
>
> 3. 250 行的腳本還是使用了多個現有 Actions 的結果，可以想見整個 workflow 有多長，手動完成有多浪費時間，現在這些繁瑣的工作流完全不需手動設定，放著就會自己跑，而且不花一毛錢。

## TL;DR

完整 workflow file 在[這裡](https://github.com/ZhenShuo2021/blowfish-core/blob/6383c98a77a809f92387f04e7023f2fc64823dab/.github/workflows/update_theme.yaml)。

<details>

<summary>update_theme.yaml</summary>

```yaml
name: Sync Theme Release

env:
  REPO_NAME: ${{ github.repository }}
  SOURCE_REPO: nunocoracao/blowfish
  SOURCE_REPO_NAME: blowfish
  SOURCE_REPO_AUTHOR: github-actions[bot]
  SOURCE_REPO_AUTHOR_EMAIL: 41898282+github-actions[bot]@users.noreply.github.com
  DEPLOY_PROJECT_NAME: blowfish-core  # Cloudflare project name (pages)
  DEFAULT_HUGO_VERSION: 0.147.1

on:
  schedule:
    - cron: '0 */8 * * *'
  workflow_dispatch:
  push:
    branches:
      - main

jobs:
  check-version:
    runs-on: ubuntu-latest
    outputs:
      SHOULD_CONTINUE: ${{ steps.determine_run.outputs.SHOULD_CONTINUE }}
      LATEST_RELEASE: ${{ steps.determine_run.outputs.LATEST_RELEASE }}
      BRANCH_NAME: ${{ steps.create_branch_name.outputs.BRANCH_NAME }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Determine whether to run
        id: determine_run
        run: |
          message="${{ github.event.head_commit.message }}"

          if [ -z "$message" ] || echo "$message" | grep -iq "merge pull request"; then
            echo "SHOULD_CONTINUE=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          latest_release=$(curl -s https://api.github.com/repos/${SOURCE_REPO}/releases/latest | jq -r .tag_name)
          current_release=$(cat .github/.theme_version 2>/dev/null || echo none)
          echo "CURRENT_RELEASE=$current_release"
          echo "LATEST_RELEASE=$latest_release" >> $GITHUB_OUTPUT

          if [ "$latest_release" == "$current_release" ]; then
            echo "SHOULD_CONTINUE=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          echo "SHOULD_CONTINUE=true" >> $GITHUB_OUTPUT

      - name: Create branch name
        id: create_branch_name
        if: steps.determine_run.outputs.SHOULD_CONTINUE == 'true'
        run: |
          branch_name="theme-update-$(echo ${{ steps.determine_run.outputs.LATEST_RELEASE }} | sed 's/[^a-zA-Z0-9]/-/g')"
          echo "BRANCH_NAME=$branch_name" >> $GITHUB_OUTPUT

  sync-release:
    if: needs.check-version.outputs.SHOULD_CONTINUE == 'true'
    needs: check-version
    runs-on: ubuntu-latest
    outputs:
      PR_NUMBER: ${{ steps.create_pr.outputs.pull-request-number }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run update script
        id: run_update_script
        run: |
          if .github/update.sh; then
            echo "UPDATE_SUCCESS=true" >> $GITHUB_OUTPUT
          else
            echo "UPDATE_SUCCESS=false" >> $GITHUB_OUTPUT
            exit 1
          fi

      - name: Commit changes
        if: steps.run_update_script.outputs.UPDATE_SUCCESS == 'true'
        run: |
          git config user.name "${SOURCE_REPO_AUTHOR}"
          git config user.email "${SOURCE_REPO_AUTHOR_EMAIL}"
          branch_name="${{ needs.check-version.outputs.BRANCH_NAME }}"
          if git ls-remote --heads origin "$branch_name" | grep -q "$branch_name"; then
            git push origin --delete "$branch_name"
          fi

          git add .
          git commit -m "⬆️ Sync Blowfish release ${{ needs.check-version.outputs.LATEST_RELEASE }}"

          echo "${{ needs.check-version.outputs.LATEST_RELEASE }}" > .github/.theme_version
          git add .github/.theme_version
          git commit -m "⬆️ Update .theme_version to ${{ needs.check-version.outputs.LATEST_RELEASE }}"

      - name: Create Pull Request
        if: steps.run_update_script.outputs.UPDATE_SUCCESS == 'true'
        id: create_pr
        uses: peter-evans/create-pull-request@v7
        with:
          title: "Sync with Blowfish ${{ needs.check-version.outputs.LATEST_RELEASE }}"
          author: "${SOURCE_REPO_AUTHOR} <${SOURCE_REPO_AUTHOR_EMAIL}>"
          body: |
            This PR synchronizes the theme with Blowfish release ${{ needs.check-version.outputs.LATEST_RELEASE }}.

            A Cloudflare preview deployment will be available for review.

            Please verify the changes before merging.
          base: main
          branch: ${{ needs.check-version.outputs.BRANCH_NAME }}
          delete-branch: true
          labels: theme-update

  deploy-preview:
    needs: [check-version, sync-release]
    if: needs.sync-release.outputs.PR_NUMBER != ''
    runs-on: ubuntu-latest
    permissions:
      contents: read
      deployments: write
      pull-requests: write
    steps:
      - name: Checkout PR branch
        uses: actions/checkout@v4
        with:
          ref: ${{ needs.check-version.outputs.BRANCH_NAME }}

      - name: Read Hugo version
        id: hugo-version
        run: |
          if [ -f "release-versions/hugo-latest.txt" ]; then
            HUGO_VERSION=$(cat release-versions/hugo-latest.txt | sed 's/^v//')
          else
            HUGO_VERSION="${DEFAULT_HUGO_VERSION}"
          fi
          echo "HUGO_VERSION=${HUGO_VERSION}" >> $GITHUB_OUTPUT

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: ${{ steps.hugo-version.outputs.HUGO_VERSION }}
          extended: true

      - name: Build site for preview
        run: |
          # Get our repo name
          IFS='/' read -r _ repo_dir <<< "${REPO_NAME}"

          # Clone the exampleSite and assets
          git clone --sparse --filter=blob:none --no-checkout --depth=1 https://github.com/${SOURCE_REPO}.git theme
          cd theme
          git sparse-checkout set exampleSite assets static
          git checkout
          cp -r exampleSite ../
          cp -r assets ../
          cp -r static ../
          cd ..

          # Move the whole 'repo_dir' directory to theme directory temporarily
          # Hugo requires the themesDir to have the same name as the config file
          cd ..
          mv "$repo_dir" "${SOURCE_REPO_NAME}"

          # Work inside the theme directory
          cd "${SOURCE_REPO_NAME}"
          hugo -E -F --minify --source exampleSite --themesDir ../.. --buildDrafts -b "https://preview-${{ needs.check-version.outputs.BRANCH_NAME }}.pages.dev/"

          # Rename 'theme_name' back to 'repo_dir'
          cd .. && mv ${SOURCE_REPO_NAME} "$repo_dir" && cd "$repo_dir"

      - name: Deploy to Cloudflare Pages
        id: deploy
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          command: pages deploy exampleSite/public --project-name=${{ env.DEPLOY_PROJECT_NAME }} --commit-dirty=true --branch=${{ needs.check-version.outputs.BRANCH_NAME }}
          gitHubToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Comment on PR with Preview URL
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const deploymentUrl = '${{ steps.deploy.outputs.deployment-url }}';
            const branchAliasUrl = '${{ steps.deploy.outputs.pages-deployment-alias-url }}';
            const previewUrl = deploymentUrl || branchAliasUrl;

            github.rest.issues.createComment({
              issue_number: ${{ needs.sync-release.outputs.PR_NUMBER }},
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `📋 Preview deployment available at: [${previewUrl}](${previewUrl})\n\nPlease review the changes before merging the PR.`
            });

  create-draft-release:
    needs: deploy-preview
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Get latest release info
        id: fetch_release
        run: |
          VERSION=$(gh release view --repo ${SOURCE_REPO} --json tagName -q '.tagName')
          BODY=$(gh release view --repo ${SOURCE_REPO} --json body -q .body)
          echo "VERSION=$VERSION" >> $GITHUB_OUTPUT
          echo "BODY<<EOF" >> $GITHUB_OUTPUT
          echo "$BODY" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Create a new tag
        run: |
          git config user.name "${SOURCE_REPO_AUTHOR}"
          git config user.email "${SOURCE_REPO_AUTHOR_EMAIL}"
          git tag -a "${{ steps.fetch_release.outputs.VERSION }}" -m "Release ${{ steps.fetch_release.outputs.VERSION }}"
          git push origin "${{ steps.fetch_release.outputs.VERSION }}"

      - name: Create release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ steps.fetch_release.outputs.VERSION }}
          name: Sync with ${{ env.SOURCE_REPO_NAME }} ${{ steps.fetch_release.outputs.VERSION }}
          body: ${{ steps.fetch_release.outputs.BODY }}
          draft: true
          prerelease: false
```

</details>

## 權限

Actions 需要權限，記得到 repo 設定這兩項才有寫入權限

- Read and write access
- Permission to create and approve pull requests

## 變數

接下來的文章介紹時只會重點解釋 highlight 的行。

```yaml {3,12}
name: Sync Theme Release

env:
  REPO_NAME: ${{ github.repository }}
  SOURCE_REPO: nunocoracao/blowfish
  SOURCE_REPO_NAME: blowfish
  SOURCE_REPO_AUTHOR: github-actions[bot]
  SOURCE_REPO_AUTHOR_EMAIL: 41898282+github-actions[bot]@users.noreply.github.com
  DEPLOY_PROJECT_NAME: blowfish-core  # Cloudflare project name (pages)
  DEFAULT_HUGO_VERSION: 0.147.1

on:
  schedule:
    - cron: '0 */8 * * *'
  workflow_dispatch:
  push:
    branches:
      - main
```

在 jobs 之前設定了這些欄位

- `env` 設定常用變數
- `on` 設定何時觸發，推送觸發限制只有 main 免的到處觸發

如果你本身就是原始 repo 的擁有者，應該使用 web hook ([repository_dispatch](https://docs.github.com/en/webhooks/webhook-events-and-payloads#repository_dispatch)) 而不是 crontab，這樣就不用重複檢查且可以立即更新。

## check-version

```yaml {1,3,12,17,21}
  check-version:  # job name "check-version"
    runs-on: ubuntu-latest
    outputs:
      SHOULD_CONTINUE: ${{ steps.determine_run.outputs.SHOULD_CONTINUE }}
      LATEST_RELEASE: ${{ steps.determine_run.outputs.LATEST_RELEASE }}
      BRANCH_NAME: ${{ steps.create_branch_name.outputs.BRANCH_NAME }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Determine whether to run
        id: determine_run
        run: |
          message="${{ github.event.head_commit.message }}"

          if [ -z "$message" ] || echo "$message" | grep -iq "merge pull request"; then
            echo "SHOULD_CONTINUE=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          latest_release=$(curl -s https://api.github.com/repos/${SOURCE_REPO}/releases/latest | jq -r .tag_name)
          current_release=$(cat .github/.theme_version 2>/dev/null || echo none)
          echo "CURRENT_RELEASE=$current_release"
          echo "LATEST_RELEASE=$latest_release" >> $GITHUB_OUTPUT

          if [ "$latest_release" == "$current_release" ]; then
            echo "SHOULD_CONTINUE=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          echo "SHOULD_CONTINUE=true" >> $GITHUB_OUTPUT
```

所有任務只在 blowfish 發佈 release 才要執行，因此將任務分成多個 jobs 以方便提前退出，這是受限於 Github Actions 沒有 [early exit 機制](https://github.com/orgs/community/discussions/82744)。

設定 outputs 欄位才可以在 jobs 之間傳遞變數，並且需要在每個 step 設定 id（例如 `determine_run`）才可以指定變數來源。如果一個 step 需要多個變數，請使用 `key=val >> $GITHUB_OUTPUT` 方式輸出。

Github 有 API 可以讀取 repo metadata，這裡我使用 curl 取得資訊並且以 jq 解析：

```sh
curl -s https://api.github.com/repos/${SOURCE_REPO}/releases/latest | jq -r .tag_name
```

也可以用 gh 指令完成，Github Actions 原生支援此指令不需設定：

```sh
gh release view --repo ${SOURCE_REPO} --json tagName -q .tagName
```

不過要注意使用 `gh` 指令時，該 step 要加上 env 才能正確執行

```yaml {4-5}
      - name: foo
        run: |
          bar=$(gh ...)
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## sync-release

```yaml {2-3,26-27,43}
  sync-release:
    if: needs.check-version.outputs.SHOULD_CONTINUE == 'true'
    needs: check-version
    runs-on: ubuntu-latest
    outputs:
      PR_NUMBER: ${{ steps.create_pr.outputs.pull-request-number }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run update script
        id: run_update_script
        run: |
          if .github/update.sh; then
            echo "UPDATE_SUCCESS=true" >> $GITHUB_OUTPUT
          else
            echo "UPDATE_SUCCESS=false" >> $GITHUB_OUTPUT
            exit 1
          fi

      - name: Commit changes
        if: steps.run_update_script.outputs.UPDATE_SUCCESS == 'true'
        run: |
          git config user.name "${SOURCE_REPO_AUTHOR}"
          git config user.email "${SOURCE_REPO_AUTHOR_EMAIL}"
          branch_name="${{ needs.check-version.outputs.BRANCH_NAME }}"
          if git ls-remote --heads origin "$branch_name" | grep -q "$branch_name"; then
            git push origin --delete "$branch_name"
          fi

          git add .
          git commit -m "⬆️ Sync Blowfish release ${{ needs.check-version.outputs.LATEST_RELEASE }}"

          echo "${{ needs.check-version.outputs.LATEST_RELEASE }}" > .github/.theme_version
          git add .github/.theme_version
          git commit -m "⬆️ Update .theme_version to ${{ needs.check-version.outputs.LATEST_RELEASE }}"

      - name: Create Pull Request
        if: steps.run_update_script.outputs.UPDATE_SUCCESS == 'true'
        id: create_pr
        uses: peter-evans/create-pull-request@v7
        with:
          title: "Sync with Blowfish ${{ needs.check-version.outputs.LATEST_RELEASE }}"
          author: "${SOURCE_REPO_AUTHOR} <${SOURCE_REPO_AUTHOR_EMAIL}>"
          body: |
            This PR synchronizes the theme with Blowfish release ${{ needs.check-version.outputs.LATEST_RELEASE }}.

            A Cloudflare preview deployment will be available for review.

            Please verify the changes before merging.
          base: main
          branch: ${{ needs.check-version.outputs.BRANCH_NAME }}
          delete-branch: true
          labels: theme-update
```

執行腳本同步 blowfish 版本。

而且因為不同的 jobs 是同時執行，所以之後的每個 jobs 都要加上 `needs` 才可以順序執行，我們也可以設定 `if` 欄位設定條件觸發。

這裡設定自動提交機器人，Github bot 的資訊應該用以下

```txt
SOURCE_REPO_AUTHOR: github-actions[bot]
SOURCE_REPO_AUTHOR_EMAIL: 41898282+github-actions[bot]@users.noreply.github.com
```

最後一步驟是自動發 PR，使用 [peter-evans/create-pull-request](https://github.com/peter-evans/create-pull-request/tree/v7/) 發佈要注意**不可推送提交**，推了就發不了 PR，因為他只找到這些目標發 PR：

1. untracked (new) files
2. tracked (modified) files
3. commits made during the workflow that have not been pushed

## deploy-preview

```yaml {5-8,31,62-63,67}
  deploy-preview:
    needs: [check-version, sync-release]
    if: needs.sync-release.outputs.PR_NUMBER != ''
    runs-on: ubuntu-latest
    permissions:
      contents: read
      deployments: write
      pull-requests: write
    steps:
      - name: Checkout PR branch
        uses: actions/checkout@v4
        with:
          ref: ${{ needs.check-version.outputs.BRANCH_NAME }}

      - name: Read Hugo version
        id: hugo-version
        run: |
          if [ -f "release-versions/hugo-latest.txt" ]; then
            HUGO_VERSION=$(cat release-versions/hugo-latest.txt | sed 's/^v//')
          else
            HUGO_VERSION="${DEFAULT_HUGO_VERSION}"
          fi
          echo "HUGO_VERSION=${HUGO_VERSION}" >> $GITHUB_OUTPUT

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: ${{ steps.hugo-version.outputs.HUGO_VERSION }}
          extended: true

      - name: Build site for preview
        run: |
          # Get our repo name
          IFS='/' read -r _ repo_dir <<< "${REPO_NAME}"

          # Clone the exampleSite and assets
          git clone --sparse --filter=blob:none --no-checkout --depth=1 https://github.com/${SOURCE_REPO}.git theme
          cd theme
          git sparse-checkout set exampleSite assets static
          git checkout
          cp -r exampleSite ../
          cp -r assets ../
          cp -r static ../
          cd ..

          # Move the whole 'repo_dir' directory to theme directory temporarily
          # Hugo requires the themesDir to have the same name as the config file
          cd ..
          mv "$repo_dir" "${SOURCE_REPO_NAME}"

          # Work inside the theme directory
          cd "${SOURCE_REPO_NAME}"
          hugo -E -F --minify --source exampleSite --themesDir ../.. --buildDrafts -b "https://preview-${{ needs.check-version.outputs.BRANCH_NAME }}.pages.dev/"

          # Rename 'theme_name' back to 'repo_dir'
          cd .. && mv ${SOURCE_REPO_NAME} "$repo_dir" && cd "$repo_dir"

      - name: Deploy to Cloudflare Pages
        id: deploy
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          command: pages deploy exampleSite/public --project-name=${{ env.DEPLOY_PROJECT_NAME }} --commit-dirty=true --branch=${{ needs.check-version.outputs.BRANCH_NAME }}
          gitHubToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Comment on PR with Preview URL
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const deploymentUrl = '${{ steps.deploy.outputs.deployment-url }}';
            const branchAliasUrl = '${{ steps.deploy.outputs.pages-deployment-alias-url }}';
            const previewUrl = deploymentUrl || branchAliasUrl;

            github.rest.issues.createComment({
              issue_number: ${{ needs.sync-release.outputs.PR_NUMBER }},
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `📋 Preview deployment available at: [${previewUrl}](${previewUrl})\n\nPlease review the changes before merging the PR.`
            });
```

這部份比較單純，就是執行 Hugo build 流程，不過要注意的是 Hugo 要求主題的名稱和資料夾名稱相同，所以 `Build site for preview` 要加上目錄處理。

Cloudflare 部署要加上他的 API Key，到 repository secret 設定加入即可，要記得在 Cloudflare 左側的 workers/pages 項目建立 project 並且設定和對應名稱就完成 Pages 部署了。

最後一步在 PR 留言就是找到 Actions 拿來用，唯一要注意的是**別忘了設定權限**。

## create-draft-release

```yaml {13-14,16-18}
  create-draft-release:
    needs: deploy-preview
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Get latest release info
        id: fetch_release
        run: |
          VERSION=$(gh release view --repo ${SOURCE_REPO} --json tagName -q '.tagName')
          BODY=$(gh release view --repo ${SOURCE_REPO} --json body -q .body)
          echo "VERSION=$VERSION" >> $GITHUB_OUTPUT
          echo "BODY<<EOF" >> $GITHUB_OUTPUT
          echo "$BODY" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Create a new tag
        run: |
          git config user.name "${SOURCE_REPO_AUTHOR}"
          git config user.email "${SOURCE_REPO_AUTHOR_EMAIL}"
          git tag -a "${{ steps.fetch_release.outputs.VERSION }}" -m "Release ${{ steps.fetch_release.outputs.VERSION }}"
          git push origin "${{ steps.fetch_release.outputs.VERSION }}"

      - name: Create release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ steps.fetch_release.outputs.VERSION }}
          name: Sync with ${{ env.SOURCE_REPO_NAME }} ${{ steps.fetch_release.outputs.VERSION }}
          body: ${{ steps.fetch_release.outputs.BODY }}
          draft: true
          prerelease: false
```

由於需要審核 PR 所以不要馬上發佈，只先寫一個 draft 草稿。

這個 step 使用 `gh` 指令調用 Github API，這裡我們要注意，如果 BODY 內容是多行的變數，需要用 EOF (heredoc) 才可傳遞給 Github_OUTPUT。由於我是要鏡像發佈 core 版本所以直接取得原本的 release 內文鏡像發佈，但是不建議這樣做，因為會發 email 給所有被標注的人很擾民，後來我就改成自動打 git tag 而已。

## 結語

Github Actions 免費版[額度](https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions)每月 2000 分鐘，Cloudflare Pages 每個月[額度](https://developers.cloudflare.com/pages/platform/limits/)可以 build 500 次，並且無限網站數量，無限 preview 數量，這個 quota 基本上用不完，又是乞丐超人的勝利。

你說設定這高達 250 行的 actions 值得嗎？我會說非常值得，放著就會自己跑完全不用顧很爽，執行起來像是[這種感覺](https://github.com/ZhenShuo2021/blowfish-core/pull/1)，但這是在別人給我整份檔案的前提下，因為我自己寫的時候全部設定都不知道，每一行都要自己找怎麼用，所有指令都要從零開始除錯有夠痛苦，於是把結果放上來讓有緣人看到就不需要重新摸索。
