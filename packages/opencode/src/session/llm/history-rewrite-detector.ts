import type { ModelMessage } from "ai"
import { diffLines } from "diff"
import * as Log from "@opencode-ai/core/util/log"

const log = Log.create({ service: "session.llm.history-rewrite-detector" })

// In-memory store of previous messages per session (deep cloned to avoid mutation issues)
const messageSnapshots = new Map<string, ModelMessage[]>()

const MAX_SESSIONS = 5

/**
 * Creates a stable string representation of a message for comparison.
 */
function messageFingerprint(msg: ModelMessage): string {
  const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
  return JSON.stringify({
    role: msg.role,
    content,
  })
}

/**
 * Truncates long strings for display in logs.
 */
function truncate(str: string, maxLen = 200): string {
  if (str.length <= maxLen) return str
  return str.slice(0, maxLen) + `... (+${str.length - maxLen} chars)`
}

/**
 * Detects when historical messages have been rewritten between requests.
 * Compares the current messages against the previous request for the same session.
 * Returns a report of changes if any were detected.
 */
export function detectMessageHistoryRewrite(sessionID: string, messages: ModelMessage[]): string | undefined {
  const prev = messageSnapshots.get(sessionID)
  if (!prev) {
    messageSnapshots.set(sessionID, structuredClone(messages))
    return undefined
  }

  // Compare message counts first
  if (prev.length === messages.length) {
    let hasChanges = false
    for (let i = 0; i < prev.length; i++) {
      if (messageFingerprint(prev[i]) !== messageFingerprint(messages[i])) {
        hasChanges = true
        break
      }
    }
    if (!hasChanges) {
      messageSnapshots.set(sessionID, structuredClone(messages))
      return undefined
    }
  }

  // Find which messages changed
  const changes: {
    index: number
    role: string
    diff: string
    oldPreview: string
    newPreview: string
  }[] = []

  const minLen = Math.min(prev.length, messages.length)
  for (let i = 0; i < minLen; i++) {
    const oldFp = messageFingerprint(prev[i])
    const newFp = messageFingerprint(messages[i])
    if (oldFp !== newFp) {
      const prevContent = prev[i].content
      const msgContent = messages[i].content
      const oldContent = typeof prevContent === "string" ? prevContent : JSON.stringify(prevContent)
      const newContent = typeof msgContent === "string" ? msgContent : JSON.stringify(msgContent)
      changes.push({
        index: i,
        role: prev[i].role,
        diff: diffLines(oldContent, newContent).map((h) => {
          const line = h.value.replace(/\n$/, "")
          if (h.added) return `+ ${line}`
          if (h.removed) return `- ${line}`
          return line
        }).join("\n"),
        oldPreview: truncate(oldContent, 150),
        newPreview: truncate(newContent, 150),
      })
    }
  }

  // Detect added/removed messages
  if (messages.length > prev.length) {
    for (let i = prev.length; i < messages.length; i++) {
      const msgContent = messages[i].content
      const content = typeof msgContent === "string" ? msgContent : JSON.stringify(msgContent)
      changes.push({
        index: i,
        role: messages[i].role,
        diff: "(message added)",
        oldPreview: "(none)",
        newPreview: truncate(content, 150),
      })
    }
  } else if (prev.length > messages.length) {
    for (let i = messages.length; i < prev.length; i++) {
      const prevContent = prev[i].content
      const content = typeof prevContent === "string" ? prevContent : JSON.stringify(prevContent)
      changes.push({
        index: i,
        role: prev[i].role,
        diff: "(message removed)",
        oldPreview: truncate(content, 150),
        newPreview: "(none)",
      })
    }
  }

  // Store current messages (bounded)
  if (messageSnapshots.size >= MAX_SESSIONS) {
    const keys = [...messageSnapshots.keys()]
    for (const key of keys.slice(0, keys.length - MAX_SESSIONS + 1)) {
      messageSnapshots.delete(key)
    }
  }
  messageSnapshots.set(sessionID, structuredClone(messages))

  if (changes.length === 0) return undefined

  // Build report
  const lines: string[] = [
    `History rewrite detected (session: ${sessionID})`,
    `${changes.length} message(s) changed:`,
    "",
  ]

  for (const change of changes) {
    lines.push(`--- Message [${change.index}] (${change.role}) ---`)
    lines.push(`Old: ${change.oldPreview}`)
    lines.push(`New: ${change.newPreview}`)
    if (change.diff) {
      lines.push("Diff:")
      lines.push(change.diff)
    }
    lines.push("")
  }

  return lines.join("\n")
}

/**
 * Clears the snapshot for a session (e.g., after compaction).
 */
export function clearHistory(sessionID: string): void {
  messageSnapshots.delete(sessionID)
}
