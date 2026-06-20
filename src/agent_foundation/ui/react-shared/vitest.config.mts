import { defineConfig } from 'vitest/config';

// Source files use the .js extension but contain JSX (matching tsup's loader config).
// Vite's built-in esbuild transform only treats .ts/.tsx/.jsx as JSX by default, so widen
// `esbuild.include` to cover plain .js and force the jsx loader. This lets component tests
// render real DOM under jsdom without renaming every source file.
export default defineConfig({
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.jsx?$/,
    exclude: [],
    jsx: 'automatic',
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: { '.js': 'jsx' },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
  },
});
