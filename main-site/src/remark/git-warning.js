import { visit } from 'unist-util-visit';

const WARNING_NODE = {
  type: 'mdxJsxFlowElement',
  name: 'div',
  attributes: [
    {
      type: 'mdxJsxAttribute',
      name: 'style',
      value: {
        type: 'mdxJsxAttributeValueExpression',
        value: '{{"background":"#fff3cd","padding":"1em","borderRadius":"6px","border":"1px solid #ffeeba","marginBottom":"1.5em","color":"#000"}}',  // 加上文字顏色設定
        data: {
          estree: {
            type: 'Program',
            body: [
              {
                type: 'ExpressionStatement',
                expression: {
                  type: 'ObjectExpression',
                  properties: [
                    {
                      type: 'Property',
                      key: { type: 'Identifier', name: 'background' },
                      value: { type: 'Literal', value: '#fff3cd' },
                      kind: 'init',
                      computed: false,
                      method: false,
                      shorthand: false
                    },
                    {
                      type: 'Property',
                      key: { type: 'Identifier', name: 'padding' },
                      value: { type: 'Literal', value: '1em' },
                      kind: 'init',
                      computed: false,
                      method: false,
                      shorthand: false
                    },
                    {
                      type: 'Property',
                      key: { type: 'Identifier', name: 'borderRadius' },
                      value: { type: 'Literal', value: '6px' },
                      kind: 'init',
                      computed: false,
                      method: false,
                      shorthand: false
                    },
                    {
                      type: 'Property',
                      key: { type: 'Identifier', name: 'border' },
                      value: { type: 'Literal', value: '1px solid #ffeeba' },
                      kind: 'init',
                      computed: false,
                      method: false,
                      shorthand: false
                    },
                    {
                      type: 'Property',
                      key: { type: 'Identifier', name: 'marginBottom' },
                      value: { type: 'Literal', value: '1.5em' },
                      kind: 'init',
                      computed: false,
                      method: false,
                      shorthand: false
                    },
                    {
                      type: 'Property',
                      key: { type: 'Identifier', name: 'color' },  // 新增顏色屬性
                      value: { type: 'Literal', value: '#000' },  // 設定文字顏色為紅色
                      kind: 'init',
                      computed: false,
                      method: false,
                      shorthand: false
                    }
                  ]
                }
              }
            ]
          }
        }
      }
    }
  ],
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
          value: 'https://zsl0621.cc/gitroad101/'
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
