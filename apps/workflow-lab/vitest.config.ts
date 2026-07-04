import { defineConfig } from 'vitest/config'
import { workflow } from '@workflow/vitest'

export default defineConfig({ plugins: [workflow()] })
