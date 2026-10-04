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
  // Self-contained bundles per entry — NO code splitting. tsup defaults
  // `splitting: true` for multi-entry ESM, which emits shared `chunk-*.mjs`
  // files that `index.mjs` re-exports via RELATIVE `./chunk-*.mjs` imports.
  // This package is consumed by Create React App (react-scripts / webpack 5),
  // whose module analysis chokes on split `.mjs` chunks inside node_modules and
  // reports every re-exporting shim as "module has no exports". A non-split
  // build keeps each entry (index/theme/protocol) fully self-contained and
  // CRA-consumable. Do NOT re-enable splitting while CRA is the consumer.
  splitting: false,
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
