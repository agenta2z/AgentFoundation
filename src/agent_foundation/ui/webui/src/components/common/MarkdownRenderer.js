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
 * Code component for ReactMarkdown with syntax highlighting
 */
const CodeComponent = ({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeContent = String(children).replace(/\n$/, '');

  // For block code (not inline), use syntax highlighter
  if (!inline) {
    // Use specified language or auto-detect
    const language = match ? match[1] : detectLanguage(codeContent);

    return (
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={language}
        PreTag="div"
        customStyle={{
          margin: '8px 0',
          borderRadius: '6px',
          fontSize: '0.85em',
        }}
        {...props}
      >
        {codeContent}
      </SyntaxHighlighter>
    );
  }

  // Inline code
  return (
    <code
      className={className}
      style={{
        backgroundColor: 'rgba(0,0,0,0.3)',
        padding: '2px 6px',
        borderRadius: 4,
        fontSize: '0.9em',
      }}
      {...props}
    >
      {children}
    </code>
  );
};

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
      backgroundColor: 'rgba(255,255,255,0.1)',
    }}>
      {children}
    </thead>
  ),
  th: ({ children }) => (
    <th style={{
      border: '1px solid rgba(255,255,255,0.2)',
      padding: '8px 12px',
      textAlign: 'left',
      fontWeight: 600,
    }}>
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td style={{
      border: '1px solid rgba(255,255,255,0.2)',
      padding: '8px 12px',
    }}>
      {children}
    </td>
  ),
};

/**
 * Reusable Markdown renderer component
 * @param {object} props
 * @param {string} props.content - Markdown content to render
 * @param {object} props.components - Additional component overrides
 */
export function MarkdownRenderer({ content, components = {} }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code: CodeComponent,
        ...TableComponents,
        ...components,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export { CodeComponent };
export default MarkdownRenderer;
