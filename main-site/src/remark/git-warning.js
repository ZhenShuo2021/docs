import { visit } from 'unist-util-visit';

const WARNING_NODE = {
  type: 'mdxJsxFlowElement',
  name: 'blockquote',
  attributes: [],
  children: [
    {
      type: 'text',
      value: 'Git 教學現在有獨立的網站了！請移駕 '
    },
    {
      type: 'mdxJsxTextElement',
      name: 'a',
      attributes: [
        {
          type: 'mdxJsxAttribute',
          name: 'href',
          value: 'https://zsl0621.cc/ripgit/'
        }
      ],
      children: [
        {
          type: 'text',
          value: 'Git 零到一百'
        }
      ]
    },
    {
      type: 'text',
      value: '。'
    }
  ]
};

const plugin = (options) => {
  const transformer = async (ast) => {
    let foundFirstH1 = false;

    visit(ast, 'heading', (node, index, parent) => {
      // 只處理第一個 h1 標題
      if (node.depth === 1 && !foundFirstH1) {
        foundFirstH1 = true;

        // 在 h1 後插入警告訊息
        if (parent && Array.isArray(parent.children) && index !== undefined) {
          const warningNodeCopy = JSON.parse(JSON.stringify(WARNING_NODE));
          parent.children.splice(index + 1, 0, warningNodeCopy);
        }
      }
    });

    // 如果沒有找到 h1 標題則加在最上方
    if (!foundFirstH1 && ast.children && ast.children.length >= 0) {
      const warningNodeCopy = JSON.parse(JSON.stringify(WARNING_NODE));
      ast.children.unshift(warningNodeCopy);
    }
  };

  return transformer;
};

export default plugin;
