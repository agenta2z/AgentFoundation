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
    'react', 'react-dom',
    '@mui/material', '@mui/icons-material',
    '@emotion/react', '@emotion/styled',
  ],
  jsx: 'automatic',
  outExtension: ({ format }) => ({ js: format === 'esm' ? '.mjs' : '.cjs' }),
});
