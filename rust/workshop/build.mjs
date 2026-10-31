// Bundle the relocated routines once; the workshop has no CDN dependency.
import { createRequire } from 'node:module';
import { mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
// Resolve the exported workshop's own dependencies outside the Hailstorm tree.
const root = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(root, 'package.json'));
const { build } = require('esbuild');
const outdir = path.resolve(process.argv[2]);
await mkdir(outdir, { recursive: true });
for (const name of ['reference-workshop', 'reference-isolation.worker', 'streaming-workshop']) {
 await build({ entryPoints: [fileURLToPath(new URL(name + '.ts', import.meta.url))],
  outfile: path.join(outdir, name + '.js'), bundle: true, format: 'esm', target: 'es2022',
  nodePaths: [path.join(root, 'node_modules')] });
}
await copyFile(new URL('reference-isolation.LICENSE.txt', import.meta.url), path.join(outdir, 'reference-isolation.LICENSE.txt'));
