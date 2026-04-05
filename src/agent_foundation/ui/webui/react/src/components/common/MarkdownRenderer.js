/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Reusable Markdown renderer with syntax highlighting.
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * Auto-detect language from code content for common languages
 */
const detectLanguage = (code) => {
  const codeStr = String(code).trim();

  // Python patterns
  if (/^(import |from |def |class |if __name__|@|print\(|async def )/.test(codeStr) ||
      /:\s*$/.test(codeStr.split('\n')[0]) && !/[{;]/.test(codeStr.split('\n')[0])) {
    return 'python';
  }

  // JavaScript/TypeScript patterns
  if (/^(const |let |var |function |import |export |=>|async |await )/.test(codeStr) ||
      /\.(then|catch|map|filter|reduce)\(/.test(codeStr)) {
    return 'javascript';
  }

  // JSON patterns
  if (/^\s*[\[{]/.test(codeStr) && /[\]}]\s*$/.test(codeStr)) {
    try {
      JSON.parse(codeStr);
      return 'json';
    } catch (e) {
      // Not valid JSON
    }
  }

  // Bash/shell patterns
  if (/^(#!\/bin\/(ba)?sh|apt-get |npm |pip |buck |cd |ls |mkdir |echo |export |source )/.test(codeStr) ||
      /^\$\s/.test(codeStr)) {
    return 'bash';
  }

  // SQL patterns
  if (/^(SELECT |INSERT |UPDATE |DELETE |CREATE |DROP |ALTER |FROM |WHERE )/i.test(codeStr)) {
    return 'sql';
  }

  // YAML patterns
  if (/^[\w-]+:\s/.test(codeStr) && !/{/.test(codeStr.split('\n')[0])) {
    return 'yaml';
  }

  // Default to plaintext for syntax highlighting (still better than no highlighting)
  return 'text';
};

/**
 * Code component for ReactMarkdown with syntax highlighting.
 *
 * react-markdown v9 removed the `inline` prop. We detect block vs inline
 * by checking for a language class or newlines in the content.
 */
const CodeComponent = ({ node, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeContent = String(children).replace(/\n$/, '');

  const isBlock = match || codeContent.includes('\n');

  if (isBlock) {
    const language = match ? match[1] : detectLanguage(codeContent);

    // Detect ToolsToInvoke blocks: check AST meta or content pattern
    const meta = node?.data?.meta || node?.properties?.dataMeta || '';
    const isToolsToInvoke = meta.includes('ToolsToInvoke') ||
      (language === 'json' && codeContent.includes('"type"') && codeContent.includes('"name"')
        && (codeContent.includes('"action"') || codeContent.includes('"conversation"')));

    return (
      <>
        {isToolsToInvoke && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 12px',
            marginTop: 8,
            backgroundColor: 'var(--theme-surface-overlay-light)',
            borderRadius: '6px 6px 0 0',
            borderBottom: '1px solid var(--theme-surface-card-border)',
            fontSize: '0.75em',
            color: 'var(--theme-warning-main)',
            fontWeight: 600,
            letterSpacing: '0.02em',
          }}>
            {'⚡ Tools to Invoke'}
          </div>
        )}
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={language}
          PreTag="div"
          customStyle={{
            margin: isToolsToInvoke ? '0 0 8px 0' : '8px 0',
            borderRadius: isToolsToInvoke ? '0 0 6px 6px' : '6px',
            fontSize: '0.85em',
          }}
          {...props}
        >
          {codeContent}
        </SyntaxHighlighter>
      </>
    );
  }

  return (
    <code
      className={className}
      style={{
        backgroundColor: 'var(--theme-surface-overlay-active)',
        padding: '2px 6px',
        borderRadius: 4,
        fontSize: '0.85em',
        fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace",
      }}
      {...props}
    >
      {children}
    </code>
  );
};

/**
 * Pre component override — pass through children directly.
 *
 * ReactMarkdown wraps fenced code blocks in <pre>. Since SyntaxHighlighter
 * already uses PreTag="div" for block code, we strip the <pre> wrapper.
 */
const PreComponent = ({ children }) => <>{children}</>;

/**
 * Link component that opens external links in new tab
 */
const LinkComponent = ({ href, children }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    style={{
      color: 'var(--theme-primary-light)',
      textDecoration: 'underline',
    }}
  >
    {children}
  </a>
);

/**
 * Table components with styling for GitHub Flavored Markdown tables
 */
const TableComponents = {
  table: ({ children }) => (
    <table style={{
      borderCollapse: 'collapse',
      width: '100%',
      margin: '16px 0',
      fontSize: '0.9em',
    }}>
      {children}
    </table>
  ),
  thead: ({ children }) => (
    <thead style={{
      backgroundColor: 'var(--theme-surface-overlay-active)',
    }}>
      {children}
    </thead>
  ),
  th: ({ children }) => (
    <th style={{
      border: '1px solid var(--theme-surface-overlay-medium)',
      padding: '8px 12px',
      textAlign: 'left',
      fontWeight: 600,
    }}>
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td style={{
      border: '1px solid var(--theme-surface-overlay-medium)',
      padding: '8px 12px',
    }}>
      {children}
    </td>
  ),
};

/**
 * Convert CamelCase tag name to spaced words.
 * e.g. "WorkflowDescription" -> "Workflow Description"
 *      "WorkflowNextStepGuidance" -> "Workflow Next Step Guidance"
 */
const camelToSpaced = (name) =>
  name.replace(/([a-z])([A-Z])/g, '$1 $2');

/**
 * Escape structural XML tags so they render as bold markdown headings.
 *
 * react-markdown strips unrecognized HTML/XML tags. This function converts
 * declared structural tags into bold markdown labels so they remain visible.
 * e.g. `<WorkflowStatus>` becomes `**Workflow Status**`
 *
 * A tag is only escaped when ALL of:
 *   1. Its name is in the tagNames list
 *   2. It appears on its own line (only whitespace around it)
 *   3. The tag pair does not nest other XML tag pairs inside
 *
 * @param {string} content - Markdown content to process
 * @param {string[]} tagNames - Tag names to escape (e.g. ['WorkflowStatus'])
 * @returns {string} Content with structural tags converted to bold labels
 */
const escapeStructuralXmlTags = (content, tagNames) => {
  if (!content || !tagNames?.length) return content;

  const tagSet = new Set(tagNames);

  for (const tagName of tagSet) {
    // Check for nesting: find content between <Tag> and </Tag>
    const pairRe = new RegExp(
      '<' + tagName + '>([\\s\\S]*?)</' + tagName + '>',
      'g',
    );
    // If any occurrence nests another XML tag pair, skip this tag entirely
    let nested = false;
    let m;
    while ((m = pairRe.exec(content)) !== null) {
      if (/<[A-Za-z][A-Za-z0-9_]*>[\s\S]*?<\/[A-Za-z][A-Za-z0-9_]*>/.test(m[1])) {
        nested = true;
        break;
      }
    }
    if (nested) continue;

    const label = camelToSpaced(tagName);
    // Escape opening/closing tags that appear on their own line
    const openRe = new RegExp('^(\\s*)<' + tagName + '>\\s*$', 'gm');
    const closeRe = new RegExp('^(\\s*)</' + tagName + '>\\s*$', 'gm');
    content = content.replace(openRe, '$1**' + label + '**');
    content = content.replace(closeRe, '');  // Remove closing tags entirely
  }

  return content;
};

/**
 * Reusable Markdown renderer component
 * @param {object} props
 * @param {string} props.content - Markdown content to render
 * @param {object} props.components - Additional component overrides
 * @param {string[]} props.structuralXmlTags - Tag names to escape for display
 */
export function MarkdownRenderer({ content, components = {}, structuralXmlTags }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code: CodeComponent,
        pre: PreComponent,
        a: LinkComponent,
        ...TableComponents,
        ...components,
      }}
    >
      {escapeStructuralXmlTags(content, structuralXmlTags)}
    </ReactMarkdown>
  );
}

export { CodeComponent };
export default MarkdownRenderer;
