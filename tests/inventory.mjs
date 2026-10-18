#!/usr/bin/env node
// Lists the source lines of one committed revision that match a pattern, for
// a defect register's revision-bound completion probe of a construct that
// must not remain:
//
//   node tests/inventory.mjs <revision> <extended-regex> <path>...
//
// Standard output carries one `revision:path:line:text` per remaining
// instance and is empty when none remains. A revision git cannot read, or a
// pattern git refuses, fails the command with git's reason; git grep's own
// "nothing matched" is the empty answer, not a failure.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
// argv holds node and this script before the caller's words.
const [, , revision, pattern, ...paths] = process.argv;
assert.ok(revision && pattern && paths.length, 'usage: node tests/inventory.mjs <revision> <extended-regex> <path>...');

const found = spawnSync('git', ['-C', root, 'grep', '-n', '-E', '-e', pattern, revision, '--', ...paths], { encoding: 'utf8' });
// git grep answers 1 when nothing matched and above 1 when it failed
// (https://git-scm.com/docs/git-grep#_exit_status).
const matched = spawnSync('true').status;
const nothing = spawnSync('false').status;
assert.ok(found.status === matched || found.status === nothing,
  `git grep ${pattern} at ${revision} failed (${found.status}): ${found.stderr}`);
process.stdout.write(found.stdout);
