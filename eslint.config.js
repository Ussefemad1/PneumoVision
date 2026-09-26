// @ts-check
import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';
import prettier from 'eslint-config-prettier';

export default tseslint.config(
  {
    // The Python ML training code, the inference service and build output are
    // out of scope for ESLint.
    ignores: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/coverage/**',
      '.venv/**',
      'medpatch/**',
      'services/**',
      'data/**',
      'checkpoints/**',
      'results/**',
    ],
  },

  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,

  // ── TypeScript sources ─────────────────────────────────────────────────────
  {
    files: ['**/*.ts', '**/*.tsx'],
    languageOptions: {
      parserOptions: {
        // A single lint-only project: the per-package build tsconfigs exclude
        // tests and config files, which typed rules still need to see.
        project: ['./tsconfig.eslint.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      '@typescript-eslint/consistent-type-imports': 'error',
      // Patient data must never be written to stdout; use the pino logger.
      'no-console': ['error', { allow: ['warn', 'error'] }],
      eqeqeq: ['error', 'always'],
    },
  },

  // ── Plain JS / MJS tooling: no type information available ─────────────────
  {
    files: ['**/*.js', '**/*.mjs', '**/*.cjs'],
    extends: [tseslint.configs.disableTypeChecked],
    languageOptions: {
      sourceType: 'module',
      globals: { ...globals.node },
    },
    rules: {
      // Build scripts legitimately report progress on stdout.
      'no-console': 'off',
    },
  },

  // ── Tests and build configs ───────────────────────────────────────────────
  {
    files: [
      '**/*.test.ts',
      '**/*.test.tsx',
      '**/tests/**/*.ts',
      '**/*.config.ts',
      '**/test-setup.ts',
    ],
    rules: {
      'no-console': 'off',
      // Assertion helpers (supertest response bodies, JSON fixtures) are
      // untyped by nature; demanding narrowing here adds noise, not safety.
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
    },
  },

  prettier,
);
