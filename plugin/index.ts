import type { OpenClawPluginApi } from "openclaw/plugin-sdk"

interface Mem0Config {
  apiUrl: string
  userId: string
  agentId: string
  autoRecall: boolean
  autoCapture: boolean
  maxRecallResults: number
  debug: boolean
}

const SKIPPED_PROVIDERS = ["exec-event", "cron-event", "heartbeat"]

// --- mem0 API client ---

async function mem0Search(cfg: Mem0Config, query: string, limit?: number) {
  const res = await fetch(`${cfg.apiUrl}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      user_id: cfg.userId,
      agent_id: cfg.agentId,
      limit: limit ?? cfg.maxRecallResults,
    }),
  })
  if (!res.ok) throw new Error(`mem0 search failed: ${res.status}`)
  return res.json()
}

async function mem0Add(cfg: Mem0Config, messages: Array<{role: string, content: string}>) {
  const res = await fetch(`${cfg.apiUrl}/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      user_id: cfg.userId,
      agent_id: cfg.agentId,
    }),
  })
  if (!res.ok) throw new Error(`mem0 add failed: ${res.status}`)
  return res.json()
}

async function mem0Profile(cfg: Mem0Config) {
  const res = await fetch(`${cfg.apiUrl}/profile/${cfg.userId}?agent_id=${cfg.agentId}`)
  if (!res.ok) throw new Error(`mem0 profile failed: ${res.status}`)
  return res.json()
}

async function mem0Forget(cfg: Mem0Config, memoryId: string) {
  const res = await fetch(`${cfg.apiUrl}/forget`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memory_id: memoryId }),
  })
  if (!res.ok) throw new Error(`mem0 forget failed: ${res.status}`)
  return res.json()
}

// --- Format recall context ---

function formatRecallContext(results: any[]): string | null {
  if (!results || results.length === 0) return null

  const lines = results.map((r: any) => {
    const memory = r.memory ?? ""
    const score = r.score != null ? `[${Math.round(r.score * 100)}%]` : ""
    const time = r.created_at ? formatTime(r.created_at) : ""
    return `- ${time ? `[${time}] ` : ""}${memory} ${score}`.trim()
  })

  return `<mem0-context>
The following memories were recalled from long-term memory (powered by mem0). Use them silently to inform your responses — only reference them when directly relevant.

## Recalled Memories
${lines.join("\n")}

Do not proactively bring up these memories unless the conversation naturally calls for it.
</mem0-context>`
}

function formatTime(iso: string): string {
  try {
    const dt = new Date(iso)
    const now = new Date()
    const hours = (now.getTime() - dt.getTime()) / 3600000
    if (hours < 1) return "just now"
    if (hours < 24) return `${Math.floor(hours)}h ago`
    const days = hours / 24
    if (days < 7) return `${Math.floor(days)}d ago`
    return dt.toLocaleDateString("en", { month: "short", day: "numeric" })
  } catch { return "" }
}

// --- Extract last turn from messages ---

function getLastTurn(messages: unknown[]): Array<{role: string, content: string}> {
  let lastUserIdx = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i] as Record<string, unknown>
    if (msg?.role === "user") { lastUserIdx = i; break }
  }
  
  const slice = lastUserIdx >= 0 ? messages.slice(lastUserIdx) : messages
  const result: Array<{role: string, content: string}> = []
  
  for (const msg of slice) {
    if (!msg || typeof msg !== "object") continue
    const m = msg as Record<string, unknown>
    if (m.role !== "user" && m.role !== "assistant") continue
    
    let text = ""
    if (typeof m.content === "string") {
      text = m.content
    } else if (Array.isArray(m.content)) {
      text = (m.content as any[])
        .filter(b => b?.type === "text" && typeof b.text === "string")
        .map(b => b.text)
        .join("\n")
    }
    
    // Strip injected mem0 context from previous turns
    text = text.replace(/<mem0-context>[\s\S]*?<\/mem0-context>\s*/g, "").trim()
    
    if (text.length >= 10) {
      result.push({ role: m.role as string, content: text })
    }
  }
  return result
}

// --- Plugin entry ---

const configSchema = {
  type: "object" as const,
  properties: {
    apiUrl: { type: "string" as const, default: "http://127.0.0.1:8765" },
    userId: { type: "string" as const, default: "omar" },
    agentId: { type: "string" as const, default: "zeno" },
    autoRecall: { type: "boolean" as const, default: true },
    autoCapture: { type: "boolean" as const, default: true },
    maxRecallResults: { type: "number" as const, default: 10 },
    debug: { type: "boolean" as const, default: false },
  },
}

export default {
  id: "openclaw-mem0",
  name: "mem0 Memory",
  description: "Self-hosted LLM-powered memory with auto-recall, auto-capture, contradiction resolution via mem0",
  kind: "memory" as const,
  configSchema,

  register(api: OpenClawPluginApi) {
    const raw = api.pluginConfig ?? {}
    const cfg: Mem0Config = {
      apiUrl: (raw.apiUrl as string) || "http://127.0.0.1:8765",
      userId: (raw.userId as string) || "omar",
      agentId: (raw.agentId as string) || "zeno",
      autoRecall: raw.autoRecall !== false,
      autoCapture: raw.autoCapture !== false,
      maxRecallResults: (raw.maxRecallResults as number) || 10,
      debug: !!raw.debug,
    }

    const log = (msg: string, ...args: any[]) => {
      if (cfg.debug) api.logger.info(`mem0: ${msg}`, ...args)
    }

    // --- Register tools ---

    api.registerTool({
      name: "mem0_store",
      description: "Save important information to long-term memory (mem0). Use for preferences, facts, decisions.",
      parameters: {
        type: "object",
        properties: {
          text: { type: "string", description: "Information to remember" },
        },
        required: ["text"],
      },
      execute: async (params: Record<string, unknown>) => {
        const text = params.text as string
        try {
          const result = await mem0Add(cfg, [
            { role: "user", content: text },
            { role: "assistant", content: "Noted and stored in long-term memory." },
          ])
          return { ok: true, result }
        } catch (err: any) {
          return { ok: false, error: err.message }
        }
      },
    })

    api.registerTool({
      name: "mem0_recall",
      description: "Search long-term memory (mem0) for relevant information. Use when you need context about user preferences, past decisions, or previously discussed topics.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", description: "Max results (default 10)" },
        },
        required: ["query"],
      },
      execute: async (params: Record<string, unknown>) => {
        try {
          const result = await mem0Search(cfg, params.query as string, params.limit as number)
          return result
        } catch (err: any) {
          return { ok: false, error: err.message }
        }
      },
    })

    api.registerTool({
      name: "mem0_forget",
      description: "Delete a specific memory from mem0 by ID.",
      parameters: {
        type: "object",
        properties: {
          memoryId: { type: "string", description: "Memory ID to delete" },
        },
        required: ["memoryId"],
      },
      execute: async (params: Record<string, unknown>) => {
        try {
          return await mem0Forget(cfg, params.memoryId as string)
        } catch (err: any) {
          return { ok: false, error: err.message }
        }
      },
    })

    api.registerTool({
      name: "mem0_profile",
      description: "View all stored memories / user profile from mem0.",
      parameters: { type: "object", properties: {} },
      execute: async () => {
        try {
          return await mem0Profile(cfg)
        } catch (err: any) {
          return { ok: false, error: err.message }
        }
      },
    })

    // --- Auto-recall hook ---

    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event: Record<string, unknown>) => {
        const prompt = event.prompt as string | undefined
        if (!prompt || prompt.length < 5) return

        try {
          const result = await mem0Search(cfg, prompt)
          const memories = result?.results?.results ?? result?.results ?? []
          const context = formatRecallContext(memories)
          
          if (context) {
            log(`injecting ${memories.length} memories (${context.length} chars)`)
            return { prependContext: context }
          }
          log("no relevant memories found")
        } catch (err: any) {
          api.logger.warn(`mem0: recall failed: ${err.message}`)
        }
        return
      })
    }

    // --- Auto-capture hook ---

    if (cfg.autoCapture) {
      api.on("agent_end", async (event: Record<string, unknown>, ctx: Record<string, unknown>) => {
        const provider = ctx.messageProvider as string
        if (SKIPPED_PROVIDERS.includes(provider)) return
        if (!event.success || !Array.isArray(event.messages) || event.messages.length === 0) return

        const lastTurn = getLastTurn(event.messages)
        if (lastTurn.length === 0) return

        log(`capturing ${lastTurn.length} messages`)

        try {
          await mem0Add(cfg, lastTurn)
        } catch (err: any) {
          api.logger.warn(`mem0: capture failed: ${err.message}`)
        }
      })
    }

    // --- Service registration ---

    api.registerService({
      id: "openclaw-mem0",
      start: () => {
        api.logger.info("mem0: connected to " + cfg.apiUrl)
      },
      stop: () => {
        api.logger.info("mem0: stopped")
      },
    })
  },
}
