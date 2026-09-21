# hermes-antigravity

Google Antigravity / Cloud Code Assist provider for **Hermes Agent**.

This repository is a Hermes-native port of
[pi-antigravity](https://github.com/Rahularya01/pi-antigravity). The original
TypeScript implementation under `src/` is kept as an upstream reference and
regression target. Hermes itself runs the Python implementation under
`hermes_antigravity/`; Pi, Node, and Yarn are **not required at runtime**.

> [!CAUTION]
> This is an unofficial integration and is not endorsed by Google. It uses
> Antigravity access outside Google's documented client surface. Accounts may
> be restricted or lose access. Use a separate account if that risk matters.

## What is preserved from pi-antigravity

The Hermes port keeps the protocol behavior that matters for long-running
agents:

- authenticated `fetchAvailableModels` model discovery;
- automatic grouping of runtime thinking variants into public model families;
- Gemini 3.8 / 3.7 / 3.6 / 3.5 Flash routing;
- Gemini 3.1 Pro, Claude 4.6, and GPT-OSS routing;
- `model_enum` discovery with static fallbacks;
- Gemini thinking budgets and Hermes reasoning-level mapping;
- Gemini thought-signature preservation for tool-call history;
- runtime fallback: Gemini 3.8 → 3.7 → 3.6 on model-not-found responses;
- endpoint failover across Antigravity production/sandbox endpoints;
- OAuth refresh;
- multiple Google accounts with automatic failover on hard quota/429 errors;
- Cloud Code project discovery/onboarding;
- quota probing;
- image data URLs and OpenAI/Hermes tool schemas.

## Install

```bash
hermes plugins install naive-one/hermes-antigravity --enable
```

Restart a running Hermes/Desktop session after the first install so both the
general plugin and generated model-provider shim are discovered.

## Login

On a desktop machine:

```bash
hermes agy login
```

On a VPS/headless server, the most reliable flow is:

```bash
hermes agy login --manual --no-browser
```

Open the printed Google authorization URL on your local computer. Google will
eventually redirect the browser to a localhost URL that may fail to load because
Hermes is running on the VPS. Copy that **entire callback URL** from the browser
address bar and paste it into the VPS prompt.

The OAuth request includes the current Antigravity `aicode` scope.

## Select a model

```bash
hermes agy select gemini-3.8-flash
```

Bare model names are normalized automatically, so the command above stores:

```text
google-antigravity/gemini-3.8-flash
```

You can also use Hermes' normal model picker after the provider is installed.

Current conservative fallback catalog:

```text
google-antigravity/gemini-3.8-flash
google-antigravity/gemini-3.7-flash
google-antigravity/gemini-3.6-flash
google-antigravity/gemini-3.5-flash
google-antigravity/gemini-3.1-pro
google-antigravity/claude-sonnet-4-6
google-antigravity/claude-opus-4-6
google-antigravity/gpt-oss-120b
```

The authenticated catalog is also inspected at runtime. Newly exposed Gemini,
Claude, or GPT-OSS families can therefore appear without waiting for the static
fallback table to be updated.

## Reasoning

Hermes reasoning levels are mapped to the runtime variants currently used by
Antigravity.

| Public model | Hermes reasoning | Runtime behavior |
| --- | --- | --- |
| Gemini 3.8/3.7/3.6 Flash | off | low runtime, thoughts disabled |
| Gemini 3.8/3.7/3.6 Flash | low | low runtime, budget 1000 |
| Gemini 3.8/3.7/3.6 Flash | medium | medium runtime, budget 4000 |
| Gemini 3.8/3.7/3.6 Flash | high | high runtime, dynamic budget -1 |
| Gemini 3.5 Flash | low / medium / high | extra-low / low / agent runtime |
| Gemini 3.1 Pro | low / high | low / agent runtime |
| Claude 4.6 | high | thinking-enabled runtime |
| GPT-OSS 120B | medium | medium runtime |

If Hermes does not explicitly enable reasoning, the provider keeps thoughts
disabled rather than silently spending thinking tokens.

## Commands

```bash
hermes agy status
hermes agy accounts
hermes agy use <account-or-email>
hermes agy remove-account <account>
hermes agy models
hermes agy models --refresh
hermes agy quota
hermes agy select <model>
hermes agy logout
```

### Multiple accounts

Each Hermes profile stores its own account pool in:

```text
$HERMES_HOME/.antigravity_accounts.json
```

The file is written with owner-only permissions where the OS supports them.

Run `hermes agy login --no-keychain` repeatedly to add accounts. The active
account is tried first. When Antigravity returns a hard quota wall (for example
HTTP 429), the current request can continue with the next stored account.

Existing single-account `$HERMES_HOME/.antigravity_oauth.json` credentials are
migrated automatically.

On macOS, an existing `agy` Keychain login can be reused unless
`--no-keychain` is supplied.

## Model discovery and fallback

For a request such as Gemini 3.8 Flash High, the port behaves approximately as:

```text
Hermes
  ↓
gemini-3.8-flash-high
  ↓ 404 / unavailable
gemini-3.7-flash-high
  ↓ 404 / unavailable
gemini-3.6-flash-high
```

Before inference, `fetchAvailableModels` is queried and cached. Live
`model_enum` values override static fallback enums.

Model discovery failure does not block inference; the conservative routing table
is still usable.

## Streaming status

The Antigravity transport itself consumes Google's SSE stream, including
thinking/tool-call parts, but **v1 currently aggregates those events into one
OpenAI-shaped completion before returning control to Hermes**. In other words,
protocol streaming is used internally, but Hermes does not yet receive native
incremental text/thinking deltas from this port.

This does not change model quality, tool calling, fallback, or token accounting;
it mainly affects time-to-visible-first-token and live thinking display. A
future version can move this integration to Hermes' provider-specific
`create_client()` transport hook to expose native streaming without changing
the Antigravity protocol modules.

## Architecture

```text
Hermes Agent
   │
   ├─ ProviderProfile
   ├─ llm_execution middleware
   └─ hermes agy CLI
          │
          ▼
hermes_antigravity/
   ├─ credentials.py   account pool
   ├─ oauth.py         Google OAuth / refresh
   ├─ cloudcode.py     project, catalog, quota
   ├─ models.py        model grouping and routing
   ├─ transform.py     Hermes/OpenAI → Antigravity wire
   ├─ client.py        SSE + endpoint failover
   ├─ runtime.py       model/account recovery
   └─ openai_compat.py Antigravity → Hermes response
          │
          ▼
Google Antigravity / Cloud Code Assist
```

The original TypeScript `src/` remains in the fork so upstream
`pi-antigravity` changes can be compared and ported without reverse
engineering the protocol again.

## Development

Python port:

```bash
python -m pip install -e .
python -m compileall -q hermes_antigravity tests
python -m unittest discover -v
```

The repository also retains upstream TypeScript tests as a protocol-reference
regression suite.

## Environment overrides

```bash
ANTIGRAVITY_USER_AGENT=...
ANTIGRAVITY_CLI_VERSION=...
ANTIGRAVITY_CLIENT_CL=...
ANTIGRAVITY_OAUTH_PORT=51121
ANTIGRAVITY_OAUTH_BIND_HOST=127.0.0.1
ANTIGRAVITY_OAUTH_REDIRECT_HOST=localhost
```

Normally none of these need to be set.

## Upstream

Protocol/reference implementation:

- https://github.com/Rahularya01/pi-antigravity

Hermes Agent:

- https://github.com/NousResearch/hermes-agent
