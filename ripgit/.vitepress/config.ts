import { defineConfig } from 'vitepress'
import footnote from 'markdown-it-footnote'
import taskCheckbox from 'markdown-it-task-checkbox'

import { themeConfig } from './themeConfig'
import { generateRewrites, createContainer } from './configTools'
import mermaidPlugin from './script/mermaid'

const srcDir = './docs'
const srcExclude = ['**/README.md', '**/TODO.md', 'snippets/*.md']

export default defineConfig({
  title: 'Git 零到一百',
  description: '自學 Git 最大的阻礙就是錯誤的網路資訊，不要再看錯誤的文章，本站教你快速且正確的學會 Git！',
  sitemap: { hostname: 'https://zsl0621.cc/ripgit/' },
  rewrites: {
    ...generateRewrites('./docs/01-basic', srcDir),
    ...generateRewrites('./docs/02-core', srcDir),
    ...generateRewrites('./docs/03-pro', srcDir),
    ...generateRewrites('./docs/04-help', srcDir),
    // 'packages/pkg-a/src/index.md': 'pkg-a/index.md',
    // 'packages/pkg-b/src/bar.md': 'pkg-b/bar.md'
  },

  markdown: {
    theme: { light: 'github-light', dark: 'github-dark' },
    image: { lazyLoading: true },
    config: (md) => {
      md.use(footnote)
      md.use(taskCheckbox)
      md.use(mermaidPlugin)
      md.use(...createContainer('note', '備註'))
      md.use(...createContainer('info', '資訊'))
      md.use(...createContainer('tip', '提示'))
      md.use(...createContainer('warning', '警告'))
      md.use(...createContainer('danger', '危險'))
      md.renderer.rules.heading_close = (tokens, idx, options, env, slf) => {
        let htmlResult = slf.renderToken(tokens, idx, options)

        if (tokens[idx].tag === 'h1' && !env.__hasInsertedMetadata) {
          htmlResult += '<ArticleMetadata />'
          env.__hasInsertedMetadata = true
        }
        return htmlResult
      }
    },
    math: true,
  },

  vite: {
    // pagefind 中文結果不精確
    // https://github.com/ATQQ/sugar-blog/blob/master/packages/vitepress-plugin-pagefind/README-zh.md
    // plugins: [
    //   pagefindPlugin({
    //     btnPlaceholder: '搜尋',
    //     placeholder: '搜尋文檔',
    //     emptyText: '空空如也',
    //     heading: '找到 {{searchResult}} 個結果',
    //     excludeSelector: ['img', 'a.header-anchor'],
    //     forceLanguage: 'zh-TW'
    //   }),
    // ],
  },

  lang: 'zh-TW',
  // In `docs` repo, `vitepress-git` folder
  base: '/ripgit/',
  head: [['link', { rel: 'icon', href: '/ripgit/img/favicon.ico' }]],
  mpa: false, // No JS in MPA mode https://github.com/vuejs/vitepress/issues/2092
  lastUpdated: true, // Only watches git commit https://vitepress.dev/reference/default-theme-last-updated
  cleanUrls: true,
  srcDir: srcDir,
  srcExclude: srcExclude,
  themeConfig: themeConfig,
})

// code import
// https://github.com/vuejs/vitepress/issues/3349
