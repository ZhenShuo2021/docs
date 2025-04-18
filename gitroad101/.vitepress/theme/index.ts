import type { Theme } from 'vitepress'
import { useData, useRoute } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import giscusTalk from 'vitepress-plugin-comment-with-giscus'
import Mermaid from './components/Mermaid.vue'
import ArticleMetadata from "./components/ArticleMetadata.vue"
import LastUpdate from "./components/LastUpdate.vue"

import './style/index.css'

export default (<Theme>{
  extends: DefaultTheme,
  enhanceApp: async ({ app }) => {
    app.component('Mermaid', Mermaid)
    // app.component('LastUpdate' , LastUpdate)
    // app.component('ArticleMetadata' , ArticleMetadata)
  },

  setup() {
    const { frontmatter } = useData()
    const route = useRoute()

    giscusTalk(
      {
        repo: 'ZhenShuo2021/giscus',
        repoId: 'R_kgDOOARatw',
        category: 'General',
        categoryId: 'DIC_kwDOOARat84CnYlI',
        mapping: 'pathname',
        inputPosition: 'top',
        lang: 'zh-TW',
      },
      {
        frontmatter,
        route,
      },
      true,
    )
  },
})
