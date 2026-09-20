#!/usr/bin/env node
/**
 * Invalidate consumer CRA/webpack persistent caches after a shared-ui rebuild.
 *
 * Webpack 5 treats `node_modules/*` as `managedPaths` and snapshots them by
 * `package.json` content rather than by transitive bundle-file mtimes. A tsup
 * rebuild that changes `dist/index.{mjs,cjs}` but leaves `package.json`
 * untouched does NOT invalidate the consumer's `.cache/default-development`.
 * The next CRA compile then loads the pre-rebuild module analysis (an older
 * export table) and reports every re-exporting shim as `module has no exports`.
 *
 * This script deletes any known consumer's persistent cache after every
 * shared-ui build, forcing webpack to re-analyze on the next compile. It is
 * safe when no consumer cache is present (no-op).
 *
 * Extend `CONSUMER_UI_ROOTS` when new consumers appear.
 */
const fs = require('fs');
const path = require('path');

const HERE = __dirname;

const CONSUMER_UI_ROOTS = [
  path.resolve(HERE, '../../../../../../OpenStartup/src/openteam/ui'),
];

const CACHE_SUBPATHS = [
  'node_modules/.cache/default-development',
  'node_modules/.cache/babel-loader',
  'node_modules/.cache/.eslintcache',
];

let cleared = 0;
for (const consumerRoot of CONSUMER_UI_ROOTS) {
  for (const sub of CACHE_SUBPATHS) {
    const p = path.join(consumerRoot, sub);
    if (fs.existsSync(p)) {
      fs.rmSync(p, { recursive: true, force: true });
      console.log(`[postbuild] cleared ${p}`);
      cleared += 1;
    }
  }
}

if (cleared === 0) {
  console.log('[postbuild] no consumer caches found (OK).');
}
