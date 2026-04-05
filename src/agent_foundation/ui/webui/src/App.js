/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * MRS RankEvolve - Main App Component
 *
 * Supports two modes:
 * - "demo": Original demo mode with pre-authored JSON flow scripts
 * - "real": Live agent mode with WebSocket-based LLM chat and /task execution
 *
 * Mode is determined by fetching GET /api/config on mount.
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Menu,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import {
  Description as FileIcon,
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Import utilities
import { API_BASE } from './utils/api';
import { addDebugLog } from './utils/debug';

// Import custom hooks
import {
  useChat,
  useProgress,
  useSectionVisibility,
  useProgressHeader,
  useInputFields,
  useContextMenu,
  useFileViewer,
} from './hooks';

// Import components
import {
  AppHeader,
  FileViewer,
  ChatMessage,
  ChatInput,
  SuggestedActions,
  PreMessage,
  ProgressSection,
  CompletedSection,
  LoadingIndicator,
  WelcomeScreen,
  MarkdownRenderer,
  EditableQueryList,
  PlanModeSelector,
} from './components';

// Import agent components
import { AgentChatPanel } from './components/agent/AgentChatPanel';

function DemoApp() {
  // ============================================
  // Custom Hooks
  // ============================================

  // Progress animation state
  const {
    progressState,
    isAnimating,
    collapsedSections,
    lastProgressPhase,
    animationPhase,
    currentPreMessage,
    revealedPostMessages,
    setProgressState,
    setIsAnimating,
    setCollapsedSections,
    setLastProgressPhase,
    setAnimationPhase,
    setCurrentPreMessage,
    setRevealedPostMessages,
    toggleSection,
    collapseSections,
    resetAnimationState,
  } = useProgress();

  // Section visibility (delayed appearance)
  const {
    visibleSections,
    sectionAppearTimes,
    hasJustAppeared,
  } = useSectionVisibility(progressState, animationPhase, lastProgressPhase);

  // Input fields state
  const {
    userInputFields,
    collapsedInputFields,
    handleUserInputChange,
    handleExpandInputField,
    getInputFieldValue,
    isInputFieldCollapsed,
  } = useInputFields();

  // Context menu state
  const {
    contextMenu,
    handleContextMenu,
    handleContextMenuClose,
  } = useContextMenu();

  // File viewer state
  const {
    fileViewerOpen,
    fileContent,
    fileName,
    isHtmlFile,
    htmlFilePath,
    openFileViewer,
    closeFileViewer,
  } = useFileViewer();

  // Refs
  const messagesEndRef = useRef(null);
  const pollingIntervalRef = useRef(null);

  // ============================================
  // HTTP Polling for Progress Updates
  // ============================================

  const startProgressPolling = useCallback(() => {
    addDebugLog('polling', 'startProgressPolling called', {
      existingInterval: !!pollingIntervalRef.current,
    });

    if (pollingIntervalRef.current) {
      addDebugLog('polling', 'Clearing existing interval', {});
      clearInterval(pollingIntervalRef.current);
    }

    addDebugLog('polling', 'Starting new polling interval (200ms)', {});
    pollingIntervalRef.current = setInterval(async () => {
      try {
        if (chatState.completionHandledRef.current) {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
          return;
        }

        const response = await fetch(`${API_BASE}/chat/progress`);
        if (!response.ok) {
          console.error('Progress polling failed:', response.status);
          return;
        }

        const progressData = await response.json();

        if (chatState.completionHandledRef.current) {
          addDebugLog('polling', 'IGNORING STRAY POLL (completion already handled)', {
            phase: progressData.phase,
          });
          return;
        }

        // Process completed steps from auto-advance chain
        if (progressData.completed_steps && progressData.completed_steps.length > 0) {
          addDebugLog('auto-advance', `PROCESSING ${progressData.completed_steps.length} COMPLETED STEPS`, {
            steps: progressData.completed_steps.map(s => s.step_id),
          });

          for (const completedStep of progressData.completed_steps) {
            const stepId = completedStep.step_id;
            if (completedStep.keep_progress_sections && completedStep.sections?.length > 0) {
              const userMessageForStep = chatState.pendingUserMessageRef.current;
              chatState.setCompletedSteps(prev => [...prev, {
                step_id: stepId,
                user_message: userMessageForStep,
                post_messages: completedStep.post_messages || [],
                progress_sections: completedStep.sections.map(s => ({
                  ...s,
                  revealed_count: completedStep.revealed_counts?.[s.slot] || s.messages.length,
                  is_completed: true,
                })),
              }]);

              if (chatState.pendingUserMessageRef.current) {
                chatState.clearPendingUserMessage();
              }

              collapseSections(completedStep.sections.map(s => s.slot));
            } else if (completedStep.post_messages?.length > 0) {
              const userMessageForStep = chatState.pendingUserMessageRef.current;
              chatState.setCompletedSteps(prev => [...prev, {
                step_id: completedStep.step_id,
                user_message: userMessageForStep,
                post_messages: completedStep.post_messages.map(msg => ({
                  role: msg.role || 'assistant',
                  content: msg.content,
                  file_path: msg.file_path,
                  message_type: msg.message_type || 'text',
                  input_field: msg.input_field,
                  editable_query_list: msg.editable_query_list,
                })),
                progress_sections: [],
              }]);

              if (chatState.pendingUserMessageRef.current) {
                chatState.clearPendingUserMessage();
              }
            }
          }
        }

        // Handle idle state
        if (progressData.phase === 'idle' && progressData.sections?.length === 0) {
          if (progressData.suggested_actions?.actions?.length > 0) {
            chatState.setSuggestedActions(progressData.suggested_actions);
          }

          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          setIsAnimating(false);
          resetAnimationState();
          chatState.markCompletionHandled();
          return;
        }

        // Update progress state
        setProgressState(progressData);
        setIsAnimating(progressData.is_animating);
        setAnimationPhase(progressData.phase);

        // Handle phases
        const phase = progressData.phase;

        if (phase === 'pre_delay') {
          setCurrentPreMessage(null);
          setRevealedPostMessages([]);
        } else if (phase === 'pre_messages') {
          const preIndex = progressData.current_pre_message_index;
          if (preIndex >= 0 && progressData.pre_messages?.[preIndex]) {
            setCurrentPreMessage(progressData.pre_messages[preIndex]);
          }
          setRevealedPostMessages([]);
        } else if (phase === 'progress_header') {
          const preIndex = progressData.current_pre_message_index;
          if (preIndex >= 0 && progressData.pre_messages?.[preIndex]) {
            setCurrentPreMessage(progressData.pre_messages[preIndex]);
          }
          setRevealedPostMessages([]);
          progressHeaderState.initializeProgressHeaderInput(
            progressData.progress_header?.input_field?.default_value
          );

          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        } else if (phase === 'progress') {
          const preIndex = progressData.current_pre_message_index;
          if (preIndex >= 0 && progressData.pre_messages?.[preIndex]) {
            setCurrentPreMessage(progressData.pre_messages[preIndex]);
          }
          setRevealedPostMessages([]);
        } else if (phase === 'post_delay') {
          const preIndex = progressData.current_pre_message_index;
          if (preIndex >= 0 && progressData.pre_messages?.[preIndex]) {
            setCurrentPreMessage(progressData.pre_messages[preIndex]);
          }
          setRevealedPostMessages([]);
        } else if (phase === 'post_messages') {
          setCurrentPreMessage(null);
          const postCount = progressData.current_post_message_count;
          if (progressData.post_messages) {
            setRevealedPostMessages(progressData.post_messages.slice(0, postCount));
          }
        } else if (phase === 'complete' && !chatState.completionHandledRef.current) {
          const stepId = progressData.step_id || 'unknown';
          addDebugLog('completion', `COMPLETION START (${stepId})`, {
            phase,
            keep_progress_sections: progressData.keep_progress_sections,
            sections_count: progressData.sections?.length || 0,
          });

          setCurrentPreMessage(null);
          setRevealedPostMessages([]);

          if (progressData.keep_progress_sections && progressData.sections?.length > 0) {
            const stepPostMessages = (progressData.step_messages || []).map(msg => ({
              role: msg.role || 'assistant',
              content: msg.content,
              file_path: msg.file_path,
              message_type: msg.message_type || 'text',
              input_field: msg.input_field,
              editable_query_list: msg.editable_query_list,
            }));
            console.log('[DEBUG] keep_progress_sections: mapped stepPostMessages:', stepPostMessages);
            const userMessageForStep = chatState.pendingUserMessageRef.current;

            chatState.setCompletedSteps(prev => [...prev, {
              step_id: stepId,
              user_message: userMessageForStep,
              post_messages: stepPostMessages,
              progress_sections: progressData.sections.map(s => ({
                ...s,
                revealed_count: progressData.revealed_counts?.[s.slot] || s.messages.length,
                is_completed: true,
              })),
            }]);

            chatState.clearPendingUserMessage();
            collapseSections(progressData.sections.map(s => s.slot));
          } else if (progressData.step_messages) {
            const newMessages = progressData.step_messages.map(msg => ({
              role: msg.role || 'assistant',
              content: msg.content,
              file_path: msg.file_path,
              message_type: msg.message_type || 'text',
              input_field: msg.input_field,
              editable_query_list: msg.editable_query_list,
            }));
            chatState.setMessages(prev => {
              const baseMessages = prev.slice(0, chatState.baseMessageCountRef.current);
              return [...baseMessages, ...newMessages];
            });
          }

          setProgressState(null);

          if (progressData.suggested_actions) {
            chatState.setSuggestedActions(progressData.suggested_actions);
          }

          chatState.markCompletionHandled();

          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }

        if (!progressData.is_animating && phase !== 'complete') {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }
      } catch (error) {
        console.error('Progress polling error:', error);
      }
    }, 200);
  }, [setProgressState, setIsAnimating, setAnimationPhase, setCurrentPreMessage,
      setRevealedPostMessages, resetAnimationState, collapseSections]);

  // Progress header state (needs startProgressPolling)
  const progressHeaderState = useProgressHeader(startProgressPolling);

  // Chat state (needs startProgressPolling and progressHeader reset)
  const chatState = useChat(
    startProgressPolling,
    progressHeaderState.resetProgressHeader,
    handleExpandInputField
  );

  // ============================================
  // Effects
  // ============================================

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatState.messages, progressState, currentPreMessage, revealedPostMessages]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Expose showState for debugging
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.showState = () => {
        const state = {
          completedSteps: chatState.completedSteps.map(s => ({
            step_id: s.step_id,
            post_messages_count: s.post_messages?.length || 0,
            progress_sections_count: s.progress_sections?.length || 0,
          })),
          progressState: progressState ? {
            phase: progressState.phase,
            sections: progressState.sections?.map(s => s.slot),
          } : null,
          animationPhase,
          isAnimating,
          messagesCount: chatState.messages.length,
        };
        console.log('[DEBUG] Current State:', state);
        return state;
      };
    }
  }, [chatState.completedSteps, progressState, animationPhase, isAnimating, chatState.messages.length]);

  // ============================================
  // Event Handlers
  // ============================================

  const handleSubmit = (e) => {
    e.preventDefault();
    const callbacks = {
      setProgressState,
      setIsAnimating,
      setCurrentPreMessage,
      setRevealedPostMessages,
      setAnimationPhase,
    };
    chatState.sendMessage(chatState.inputValue, callbacks);
  };

  const handleActionClick = (actionIndex) => {
    const callbacks = {
      setProgressState,
      setIsAnimating,
      setCurrentPreMessage,
      setRevealedPostMessages,
      setAnimationPhase,
    };
    chatState.handleAction(actionIndex, callbacks);
  };

  const handleBranchActionClick = (actionIndex, inputValue) => {
    const callbacks = {
      setProgressState,
      setIsAnimating,
      setCurrentPreMessage,
      setRevealedPostMessages,
      setAnimationPhase,
    };
    chatState.handleBranchAction(actionIndex, inputValue, callbacks);
  };

  const handleProgressHeaderContinue = () => {
    progressHeaderState.handleProgressHeaderContinue(progressState, {
      setProgressState,
      setAnimationPhase,
      setIsAnimating,
      pollingIntervalRef,
    });
  };

  // ============================================
  // Render Helper Functions
  // ============================================

  const renderInputField = (inputField, messageIndex) => {
    if (!inputField) return null;

    console.log('[DEBUG] renderInputField called with:', JSON.stringify(inputField, null, 2));

    const variableName = inputField.variable_name || `input_${messageIndex}`;
    const isCollapsible = inputField.collapsible || false;
    const initiallyCollapsed = inputField.initially_collapsed || false;
    const isCollapsed = isInputFieldCollapsed(variableName, initiallyCollapsed);
    const currentValue = getInputFieldValue(variableName, inputField.default_value || '');

    if (isCollapsible && isCollapsed) {
      return (
        <Box sx={{ mt: 2 }}>
          <Button
            variant="outlined"
            onClick={() => handleExpandInputField(variableName)}
            startIcon={<ExpandMoreIcon />}
            sx={{
              fontWeight: 400,
              textTransform: 'none',
              borderColor: 'rgba(255, 255, 255, 0.3)',
              color: 'text.secondary',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'rgba(74, 144, 217, 0.1)',
              },
            }}
          >
            ✏️ Click to customize
          </Button>
        </Box>
      );
    }

    if (inputField.mode_selector) {
      return (
        <PlanModeSelector
          modeSelector={inputField.mode_selector}
          variableName={variableName}
          placeholder={inputField.placeholder}
          onChange={handleUserInputChange}
          currentValue={currentValue}
        />
      );
    }

    return (
      <Box sx={{ mt: 2 }}>
        <TextField
          fullWidth
          multiline={inputField.multiline || false}
          minRows={inputField.multiline ? 3 : 1}
          maxRows={inputField.multiline ? 8 : 1}
          placeholder={inputField.placeholder || 'Enter your input...'}
          value={currentValue}
          onChange={(e) => handleUserInputChange(variableName, e.target.value)}
          variant="outlined"
          sx={{
            '& .MuiOutlinedInput-root': {
              backgroundColor: 'rgba(0, 0, 0, 0.2)',
              borderRadius: 1,
              '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.2)' },
              '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
              '&.Mui-focused fieldset': { borderColor: 'primary.main' },
            },
            '& .MuiInputBase-input': { color: 'text.primary', fontSize: '0.95rem' },
          }}
        />
        {inputField.optional && (
          <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
            Optional - you can leave this blank
          </Typography>
        )}
      </Box>
    );
  };

  const renderEditableQueryList = (queryListConfig, messageIndex) => {
    console.log('[DEBUG] renderEditableQueryList called:', {
      messageIndex,
      queryListConfig: queryListConfig ? JSON.stringify(queryListConfig, null, 2) : 'null/undefined',
    });
    if (!queryListConfig) return null;
    return (
      <EditableQueryList
        queries={queryListConfig.queries || []}
        additionalQueries={queryListConfig.additional_queries || []}
        allowCustomQuery={queryListConfig.allow_custom_query ?? true}
        variableName={queryListConfig.variable_name || 'selected_queries'}
        customQueryPlaceholder={queryListConfig.custom_query_placeholder || 'Enter your own research query...'}
        onChange={(queries) => handleUserInputChange(queryListConfig.variable_name || 'selected_queries', queries)}
      />
    );
  };

  const renderRevealedPostMessages = () => {
    if (revealedPostMessages.length === 0) return null;
    return revealedPostMessages.map((msg, index) => (
      <Box
        key={`post-${index}`}
        sx={{
          display: 'flex', justifyContent: 'flex-start', mb: 2,
          animation: 'fadeIn 0.3s ease-in-out',
          '@keyframes fadeIn': { '0%': { opacity: 0, transform: 'translateY(10px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        }}
      >
        <Paper elevation={0} sx={{ p: 2, maxWidth: '80%', backgroundColor: 'background.paper', borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
          <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
            <MarkdownRenderer content={msg.content} />
          </Box>
          {msg.file_path && (
            <Button size="small" startIcon={<FileIcon />} onClick={() => openFileViewer(msg.file_path)} sx={{ mt: 1 }}>
              📄 View File
            </Button>
          )}
          {msg.input_field && renderInputField(msg.input_field, `revealed-post-${index}`)}
          {msg.editable_query_list && renderEditableQueryList(msg.editable_query_list, `revealed-post-${index}`)}
        </Paper>
      </Box>
    ));
  };

  const renderProgressAnimation = () => {
    if (!progressState || !progressState.sections || progressState.sections.length === 0) return null;

    const showProgressPhases = ['progress', 'post_delay', 'post_messages', 'complete'];
    if (!showProgressPhases.includes(animationPhase)) return null;

    const sectionsToRender = progressState.sections.filter(section => {
      if (!section.appearance_delay_ms || section.appearance_delay_ms <= 0) return true;
      return visibleSections[section.slot] === true;
    });

    return (
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2, justifyContent: 'center' }} className="progress-sections-container">
        {sectionsToRender.map((section, sectionIndex) => {
          const revealedCount = progressState.revealed_counts[section.slot] || 0;
          const isComplete = revealedCount >= section.messages.length;

          return (
            <ProgressSection
              key={sectionIndex}
              section={section}
              revealedCount={revealedCount}
              isComplete={isComplete}
              isCollapsed={collapsedSections[section.slot]}
              isAnimating={isAnimating}
              justAppeared={hasJustAppeared(section.slot)}
              onToggle={toggleSection}
              onContextMenu={handleContextMenu}
              onOpenFile={openFileViewer}
            />
          );
        })}
      </Box>
    );
  };

  const renderProgressHeader = () => {
    if (progressHeaderState.progressHeaderSubmitted && progressHeaderState.submittedProgressHeader) {
      const header = progressHeaderState.submittedProgressHeader.header;
      const submittedValue = progressHeaderState.submittedProgressHeader.inputValue;

      return (
        <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
          <Paper
            elevation={0}
            sx={{
              p: 2, maxWidth: '80%', backgroundColor: 'rgba(74, 144, 217, 0.08)',
              borderRadius: 2, border: '1px solid', borderColor: 'rgba(74, 144, 217, 0.3)', opacity: 0.85,
            }}
          >
            <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
              <MarkdownRenderer content={header.content} />
            </Box>
            {header.input_field && (
              <Box sx={{ mt: 2 }}>
                <TextField
                  fullWidth
                  multiline={header.input_field.multiline || false}
                  minRows={header.input_field.multiline ? 3 : 1}
                  maxRows={header.input_field.multiline ? 8 : 1}
                  placeholder={header.input_field.placeholder || 'Enter your input...'}
                  value={submittedValue}
                  disabled={true}
                  variant="outlined"
                  sx={{
                    '& .MuiOutlinedInput-root': { backgroundColor: 'rgba(0, 0, 0, 0.1)', borderRadius: 1, '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.1)' } },
                    '& .MuiInputBase-input': { color: 'text.secondary', fontSize: '0.95rem' },
                    '& .Mui-disabled': { WebkitTextFillColor: 'rgba(255, 255, 255, 0.5)' },
                  }}
                />
              </Box>
            )}
            {header.continue_button && (
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={16} thickness={4} sx={{ color: 'primary.light' }} />
                <Button variant="outlined" disabled={true} sx={{ fontWeight: 400, px: 3, py: 1, opacity: 0.6 }}>
                  {header.continue_button.label}
                </Button>
              </Box>
            )}
          </Paper>
        </Box>
      );
    }

    if (!progressState || !progressState.progress_header || animationPhase !== 'progress_header') return null;

    const header = progressState.progress_header;

    return (
      <Box
        sx={{
          display: 'flex', justifyContent: 'flex-start', mb: 2,
          animation: 'fadeIn 0.3s ease-in-out',
          '@keyframes fadeIn': { '0%': { opacity: 0, transform: 'translateY(10px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 2, maxWidth: '80%', backgroundColor: 'rgba(74, 144, 217, 0.15)',
            borderRadius: 2, border: '1px solid', borderColor: 'primary.main',
          }}
        >
          <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
            <MarkdownRenderer content={header.content} />
          </Box>
          {header.input_field && (
            <Box sx={{ mt: 2 }}>
              <TextField
                fullWidth
                multiline={header.input_field.multiline || false}
                minRows={header.input_field.multiline ? 3 : 1}
                maxRows={header.input_field.multiline ? 8 : 1}
                placeholder={header.input_field.placeholder || 'Enter your input...'}
                value={progressHeaderState.progressHeaderInput}
                onChange={(e) => progressHeaderState.setProgressHeaderInput(e.target.value)}
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    backgroundColor: 'rgba(0, 0, 0, 0.2)', borderRadius: 1,
                    '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.2)' },
                    '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                    '&.Mui-focused fieldset': { borderColor: 'primary.main' },
                  },
                  '& .MuiInputBase-input': { color: 'text.primary', fontSize: '0.95rem' },
                }}
              />
              {header.input_field.optional && (
                <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
                  Optional - you can leave this blank
                </Typography>
              )}
            </Box>
          )}
          {header.continue_button && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant={header.continue_button.style === 'primary' ? 'contained' : 'outlined'}
                onClick={handleProgressHeaderContinue}
                sx={{ fontWeight: header.continue_button.style === 'primary' ? 600 : 400, px: 3, py: 1 }}
              >
                {header.continue_button.label}
              </Button>
            </Box>
          )}
        </Paper>
      </Box>
    );
  };

  const renderAllSteps = () => {
    return (
      <>
        {chatState.completedSteps?.length > 0 && chatState.completedSteps.map((step, stepIndex) => (
          <Box key={`completed-step-${stepIndex}`} sx={{ mb: 3 }}>
            {step.user_message && (
              <ChatMessage message={step.user_message} onOpenFile={openFileViewer} />
            )}
            {step.post_messages?.map((msg, msgIndex) => {
              console.log('[DEBUG] renderAllSteps - rendering post_message:', {
                stepIndex,
                msgIndex,
                hasEditableQueryList: !!msg.editable_query_list,
                editableQueryList: msg.editable_query_list ? JSON.stringify(msg.editable_query_list, null, 2) : 'null/undefined',
                msgKeys: Object.keys(msg),
              });
              return (
              <Box key={`step-${stepIndex}-msg-${msgIndex}`} sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                <Paper elevation={0} sx={{ p: 2, maxWidth: '80%', backgroundColor: 'background.paper', borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                  <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
                    <MarkdownRenderer content={msg.content} />
                  </Box>
                  {msg.file_path && (
                    <Button size="small" startIcon={<FileIcon />} onClick={() => openFileViewer(msg.file_path)} sx={{ mt: 1 }}>📄 View File</Button>
                  )}
                  {msg.input_field && renderInputField(msg.input_field, `step-${stepIndex}-msg-${msgIndex}`)}
                  {msg.editable_query_list && renderEditableQueryList(msg.editable_query_list, `step-${stepIndex}-msg-${msgIndex}`)}
                </Paper>
              </Box>
              );
            })}

            {step.progress_sections?.length > 0 && (
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center', mb: 2 }}>
                {step.progress_sections.map((section, sectionIndex) => (
                  <CompletedSection
                    key={`completed-${stepIndex}-${sectionIndex}`}
                    section={section}
                    isCollapsed={collapsedSections[section.slot]}
                    onToggle={toggleSection}
                    onContextMenu={handleContextMenu}
                    onOpenFile={openFileViewer}
                  />
                ))}
              </Box>
            )}
          </Box>
        ))}

        {(chatState.pendingUserMessage || isAnimating) && (
          <Box sx={{ mb: 3 }}>
            {chatState.pendingUserMessage && (
              <ChatMessage message={chatState.pendingUserMessage} onOpenFile={openFileViewer} />
            )}
            {currentPreMessage && (
              <PreMessage message={currentPreMessage} />
            )}
            {renderProgressHeader()}
            {renderProgressAnimation()}
            {renderRevealedPostMessages()}
          </Box>
        )}
      </>
    );
  };

  // ============================================
  // Main Render
  // ============================================

  const showWelcomeScreen = chatState.messages.length === 0 &&
    !isAnimating &&
    !chatState.pendingUserMessage &&
    chatState.completedSteps.length === 0;

  return (
    <>
      {/* Main Chat Area */}
      <Container maxWidth="md" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 2, overflow: 'hidden' }}>
        <Box sx={{ flex: 1, overflow: 'auto', mb: 2, px: 1 }}>
          {showWelcomeScreen && <WelcomeScreen />}

          {chatState.messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} onOpenFile={openFileViewer} />
          ))}
          {renderAllSteps()}

          <SuggestedActions
            suggestedActions={chatState.suggestedActions}
            onAction={handleActionClick}
            onBranchAction={handleBranchActionClick}
            disabled={chatState.isLoading || isAnimating}
          />

          {chatState.isLoading && !isAnimating && <LoadingIndicator />}

          <div ref={messagesEndRef} />
        </Box>

        <ChatInput
          value={chatState.inputValue}
          onChange={chatState.setInputValue}
          onSubmit={handleSubmit}
          disabled={chatState.isLoading}
        />
      </Container>

      <FileViewer
        open={fileViewerOpen}
        onClose={closeFileViewer}
        fileName={fileName}
        fileContent={fileContent}
        isHtmlFile={isHtmlFile}
        htmlFilePath={htmlFilePath}
      />

      <Menu
        open={contextMenu !== null}
        onClose={handleContextMenuClose}
        anchorReference="anchorPosition"
        anchorPosition={contextMenu !== null ? { top: contextMenu.mouseY, left: contextMenu.mouseX } : undefined}
      >
        <MenuItem
          onClick={() => {
            if (contextMenu?.section?.prompt_file) openFileViewer(contextMenu.section.prompt_file);
            handleContextMenuClose();
          }}
          disabled={!contextMenu?.section?.prompt_file}
        >
          <FileIcon sx={{ mr: 1, fontSize: 18 }} />View Prompt
        </MenuItem>
      </Menu>
    </>
  );
}

function App() {
  const [mode, setMode] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then(res => res.json())
      .then(data => {
        setMode(data.mode || 'demo');
        setLoading(false);
      })
      .catch(() => {
        setMode('demo');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <AppHeader />
        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <CircularProgress />
        </Box>
      </Box>
    );
  }

  if (mode === 'real') {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <AppHeader title="RankEvolve Agent" />
        <AgentChatPanel />
      </Box>
    );
  }

  // Demo mode — original behavior
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <AppHeader />
      <DemoApp />
    </Box>
  );
}

export default App;
