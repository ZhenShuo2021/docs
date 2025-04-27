import * as fs from 'node:fs'
import * as path from 'node:path'
import container from 'markdown-it-container'
import type { PluginSimple } from 'markdown-it'

export const createContainer = (type: string, defaultTitle: string): [PluginSimple, string, { render: (tokens: any[], idx: number) => string }] => [
  container,
  type,
  {
    render(tokens, idx) {
      const token = tokens[idx]
      const info = token.info.trim().slice(type.length).trim()
      if (token.nesting === 1) {
        return `<div class="custom-block ${type}"><p class="custom-block-title">${info || defaultTitle}</p>\n`
      }
      return '</div>\n'
    }
  }
]

function generateRewrites(sourceDir = './docs', srcDir = './docs') {
  const rewrites = {}
  const absoluteRoot = path.resolve(srcDir)

  function processDirectory(dir) {
    const files = fs.readdirSync(dir)

    for (const file of files) {
      const fullPath = path.join(dir, file)
      const stat = fs.statSync(fullPath)

      if (stat.isDirectory()) {
        processDirectory(fullPath)
      } else if (path.extname(file) === '.md') {
        const absolutePath = path.resolve(fullPath)
        const relativePath = path.relative(absoluteRoot, absolutePath)

        const pathParts = relativePath.split(path.sep)
        const cleanedParts = pathParts.map((part) => part.replace(/^\d+-/, ''))

        const cleanedPath = cleanedParts.join('/')
        if (relativePath !== cleanedPath) {
          rewrites[relativePath] = cleanedPath
        }
      }
    }
  }

  try {
    processDirectory(path.resolve(sourceDir))
  } catch (error) {
    console.error('生成重寫規則時出錯:', error)
  }

  return rewrites
}

export { generateRewrites }
