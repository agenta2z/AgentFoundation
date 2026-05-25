/**
 * ThinkingFold — Collapsible "Thinking" section.
 * 
 * Shows pre-<Response> content (LLM reasoning, tool calls, intermediate output)
 * in a collapsed block that can be expanded. Adapted from rankevolve's
 * AgentStreamSection.js ThinkingFold component.
 */

import React, { useState } from 'react';
import { Box, Typography, Collapse } from '@mui/material';
import { Psychology as ThinkingIcon } from '@mui/icons-material';
import { MarkdownRenderer } from './MarkdownRenderer';

/**
 * Strip tool invocation blocks from content.
 * Handles all formats the LLM may produce:
 * 1. ```json ToolsToInvoke ... ``` (standard format)
 * 2. ```json ... ``` containing "type":"conversation"/"action" (no ToolsToInvoke label)
 * 3. Raw JSON lines with "type":"conversation"/"action" (no code fences)
 * 4. <ConversationTools>...</ConversationTools> (legacy)
 */
export function stripToolsToInvoke(text) {
  if (!text) return text;
  // 1. ToolsToInvoke code blocks (completed + unclosed)
  let result = text.replace(/```json\s+ToolsToInvoke\n[\s\S]*?```/g, '');
  result = result.replace(/```json\s+ToolsToInvoke\n[\s\S]*$/g, '');
  // 2. Any ```json...``` block containing conversation/action tool JSON
  result = result.replace(/```json\s*\n[\s\S]*?```/g, (match) => {
    if (/"type"\s*:\s*"(?:conversation|action)"/.test(match)) return '';
    return match;  // Keep normal code blocks
  });
  // 3. Unclosed ```json blocks with tool JSON (streaming)
  result = result.replace(/```json\s*\n[\s\S]*$/g, (match) => {
    if (/"type"\s*:\s*"(?:conversation|action)"/.test(match)) return '';
    return match;
  });
  // 4. Raw conversation/action tool JSON lines (no code fences)
  result = result.replace(/^\s*\{"type"\s*:\s*"(?:conversation|action)"[^\n]*$/gm, '');
  // 5. Legacy <ConversationTools> tags
  result = result.replace(/<ConversationTools>[\s\S]*?<\/ConversationTools>/g, '');
  // 6. Collapse excessive blank lines
  result = result.replace(/\n{3,}/g, '\n\n');
  return result.trim();
}

/**
 * Strip <Response>/<\/Response> tags and acli separator lines from content.
 */
export function stripResponseTags(text) {
  if (!text) return text;
  return text
    .replace(/<Response>/g, '')
    .replace(/<\/Response>/g, '')
    .replace(/─── Response ─+/g, '')
    .replace(/─{40,}/g, '')
    .trim();
}

/**
 * Parse raw LLM output into thinking + response phases.
 *
 * acli output uses "─── Response ───..." separator lines (not XML tags).
 * The output contains multiple Response separator pairs — each section is
 * a thinking/response cycle. The LAST "─── Response ───" separator marks
 * the start of the final user-facing answer.
 *
 * Also supports <Response>...</Response> XML tags as a fallback.
 *
 * Three phases:
 * - pre_response:  no separator/tag seen yet → everything is thinking
 * - in_response:   separator seen, no closing separator → split at last one
 * - post_response: both separators seen → clean split
 *
 * @param {string} rawContent - Raw LLM output
 * @returns {{ phase: string, thinkingContent: string, responseContent: string }}
 */
export function parseResponseTags(rawContent) {
  if (!rawContent) return { phase: 'pre_response', thinkingContent: '', responseContent: '' };

  // First try XML <Response> tags (template-based output)
  const xmlStart = rawContent.indexOf('<Response>');
  if (xmlStart !== -1) {
    const thinking = rawContent.slice(0, xmlStart).trim();
    const afterTag = rawContent.slice(xmlStart + '<Response>'.length);
    const xmlEnd = afterTag.indexOf('</Response>');
    if (xmlEnd === -1) {
      return { phase: 'in_response', thinkingContent: thinking, responseContent: afterTag };
    }
    return { phase: 'post_response', thinkingContent: thinking, responseContent: afterTag.slice(0, xmlEnd) };
  }

  // Try acli "─── Response ───" separators
  // Find all positions of "─── Response ─" separator lines
  const sepRegex = /─── Response ─+/g;
  const matches = [];
  let match;
  while ((match = sepRegex.exec(rawContent)) !== null) {
    matches.push(match.index);
  }

  if (matches.length === 0) {
    return { phase: 'pre_response', thinkingContent: rawContent, responseContent: '' };
  }

  // Everything before the last "─── Response ───" is thinking
  const lastSepStart = matches[matches.length - 1];
  const thinking = rawContent.slice(0, lastSepStart).trim();

  // Find the newline after the separator line
  const newlineAfter = rawContent.indexOf('\n', lastSepStart);
  if (newlineAfter === -1) {
    return { phase: 'in_response', thinkingContent: thinking, responseContent: '' };
  }

  const afterSep = rawContent.slice(newlineAfter + 1);

  // Find closing plain separator line (────...──── with 40+ dashes)
  const closeMatch = afterSep.match(/\n─{40,}(?:\n|$)/);
  if (!closeMatch) {
    // No closing separator yet — still streaming the response
    return { phase: 'in_response', thinkingContent: thinking, responseContent: afterSep };
  }

  // Clean response is between last Response separator and closing separator
  const responseContent = afterSep.slice(0, closeMatch.index);
  return { phase: 'post_response', thinkingContent: thinking, responseContent: responseContent.trim() };
}

/**
 * Parse session context line from acli output.
 * Example: "Session context: ▮▮▮▮▮▮▮▮▮▮ 33.8K/1M"
 *
 * @param {string} text - Raw content to search for context line
 * @returns {{ used: string, total: string, percentage: number } | null}
 */
export function parseSessionContext(text) {
  if (!text) return null;
  const match = text.match(/Session context:.*?(\d+\.?\d*[KMkm]?)\s*\/\s*(\d+\.?\d*[KMkm]?)/);
  if (!match) return null;

  const parseNum = (s) => {
    const num = parseFloat(s);
    if (s.toUpperCase().endsWith('K')) return num * 1000;
    if (s.toUpperCase().endsWith('M')) return num * 1000000;
    return num;
  };

  const used = parseNum(match[1]);
  const total = parseNum(match[2]);
  return {
    usedLabel: match[1],
    totalLabel: match[2],
    percentage: total > 0 ? (used / total) * 100 : 0,
  };
}

/**
 * Strip session context line from content for clean display.
 */
export function stripSessionContext(text) {
  if (!text) return text;
  return text.replace(/Session context:.*$/gm, '').trim();
}

/**
 * Strip ANSI escape codes from text.
 */
export function stripAnsi(text) {
  if (!text) return text;
  // eslint-disable-next-line no-control-regex
  return text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
}

/**
 * Strip acli header noise from streaming output.
 * Removes lines like: "Working in ...", "Jira projects: ...", "[?2004h...",
 * "INTERNAL USE: ...", "✔ Using model: ...", "✔ Started ...",
 * "─── Response ───", "────────────"
 */
export function stripAcliNoise(text) {
  if (!text) return text;
  const noisePatterns = [
    /^\[[\?0-9;]*[a-zA-Z].*/gm,                      // ANSI escape sequences at line start
    /^Working in .*/gm,                                // Working directory
    /^Jira projects: .*/gm,                            // Jira project refs
    /^INTERNAL USE: .*$/gm,                            // Internal use warning
    /^✔ Using model: .*/gm,                            // Model info
    /^✔ Started \d+ MCP.*/gm,                          // MCP server count
    /^─{3,}.*─{3,}$/gm,                               // Separator lines
    /^─── Response ─+$/gm,                             // Response header lines
  ];
  let result = text;
  for (const pattern of noisePatterns) {
    result = result.replace(pattern, '');
  }
  // Collapse multiple blank lines
  result = result.replace(/\n{3,}/g, '\n\n');
  return result.trim();
}


export function ThinkingFold({ thinkingContent }) {
  const [expanded, setExpanded] = useState(false);

  if (!thinkingContent) return null;

  // Clean up the thinking content
  const cleaned = stripAnsi(stripAcliNoise(stripToolsToInvoke(thinkingContent)));
  if (!cleaned) return null;

  const charCount = cleaned.length;

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box
        onClick={() => setExpanded(!expanded)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          cursor: 'pointer',
          py: 0.5,
          px: 1,
          borderRadius: 1,
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
          '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.06)' },
        }}
      >
        <ThinkingIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        <Typography
          variant="caption"
          sx={{ color: 'text.disabled', fontWeight: 500, userSelect: 'none' }}
        >
          {expanded ? '▾' : '▸'} Thinking ({charCount.toLocaleString()} chars)
        </Typography>
      </Box>
      <Collapse in={expanded}>
        <Box
          sx={{
            mt: 0.5,
            ml: 1,
            pl: 1.5,
            borderLeft: '2px solid rgba(255, 255, 255, 0.08)',
            opacity: 0.5,
            color: 'text.secondary',
            maxHeight: 200,
            overflow: 'auto',
            fontSize: '0.8rem',
            '& p': { m: 0 },
          }}
        >
          <MarkdownRenderer content={cleaned} />
        </Box>
      </Collapse>
    </Box>
  );
}

export default ThinkingFold;
