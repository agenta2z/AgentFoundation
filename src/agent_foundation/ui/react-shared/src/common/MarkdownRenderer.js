/**
 * Reusable Markdown renderer with syntax highlighting.
 * Copied from rankevolve — adapted for OpenStartup.
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
  if (/^\s*[[\{]/.test(codeStr) && /[\]}]\s*$/.test(codeStr)) {
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

  return 'text';
};

/**
 * Code component for ReactMarkdown with syntax highlighting
 */
const CodeComponent = ({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeContent = String(children).replace(/\n$/, '');
  const isBlock = !inline && (node?.position?.start?.line !== node?.position?.end?.line || match || className);

  if (isBlock) {
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
 * Normalize LLM output before markdown parsing.
 *
 * Two transforms:
 * 1. Convert line-start Unicode bullet characters (•·‣⁃) to markdown list
 *    markers (-). LLMs frequently use • instead of - for bullet lists, which
 *    react-markdown does not recognize as a list marker.
 *    Example: "  • Item text" → "- Item text"
 *
 * 2. Insert a blank line before list blocks that immediately follow a paragraph.
 *    The CommonMark spec requires a blank line between a paragraph and a list;
 *    without it the list is treated as a continuation of the paragraph.
 *    Example: "Some text\n- item" → "Some text\n\n- item"
 *
 * Note: inline bullets on the same line (e.g. "Section • item1 • item2") are
 * not converted — they are not at line-start and should remain as plain text.
 */
function preprocessContent(content) {
  if (!content) return content;
  let text = content;

  // Convert line-start Unicode bullets to markdown list markers
  text = text.replace(/^[ \t]*[•·‣⁃]\s+/gm, '- ');

  // Ensure a blank line before list blocks that follow a non-list line.
  // Uses ^([^-\n].*) to match only lines that don't already start with '-',
  // preventing blank-line insertion between consecutive list items (which would
  // create CommonMark "loose lists" and add unwanted <p> wrappers inside <li>).
  text = text.replace(/^([^-\n].*)\n(- )/gm, '$1\n\n$2');

  return text;
}

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
      {preprocessContent(content)}
    </ReactMarkdown>
  );
}

export { CodeComponent };
export default MarkdownRenderer;
