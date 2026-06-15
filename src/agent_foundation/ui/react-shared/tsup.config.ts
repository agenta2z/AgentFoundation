import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index:    'src/index.js',
    theme:    'src/theme/index.js',
    protocol: 'src/protocol/index.js',
  },
  format: ['esm', 'cjs'],
  outDir: 'dist',
  clean: true,
  sourcemap: true,
  treeshake: true,
  external: [
    'react', 'react-dom', 'react/jsx-runtime', 'react/jsx-dev-runtime',
    '@mui/material', '@mui/icons-material',
    '@emotion/react', '@emotion/styled',
    'react-markdown', 'remark-gfm',
    'react-syntax-highlighter', 'react-syntax-highlighter/dist/esm/styles/prism',
  ],
  jsx: 'automatic',
  loader: { '.js': 'jsx' },
  outExtension: ({ format }) => ({ js: format === 'esm' ? '.mjs' : '.cjs' }),
});
