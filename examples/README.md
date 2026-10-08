# HEPTAPOD examples

These examples support coding-agent workflows through toolbase and standalone API-based workflows through Orchestral. For a coding agent, follow the [main quickstart](../README.md#quickstart) and the example's launcher instructions. The provider configuration below is for standalone Orchestral demos.

| Example | Workflow |
|---------|----------|
| [Primer](primer/) | Build and package your first tools |
| [EDA](eda/) | Exact symbolic amplitude calculations |
| [NDA](nda/) | Diagram enumeration and dimensional estimates |
| [Leptoquark simulation](sim/s1_lq_rr/) | Monte Carlo event generation and analysis |

## API and provider configuration

Run the following commands from the root of a [source checkout](../docs/usage.md#installing-from-source). The demos run in your Python environment; see each example's README for dependencies and external software requirements.

Copy the API-key template:

```bash
cp .env.example .env
```

Copy the provider and external-software configuration template:

```bash
cp config.example.py config.py
```

Edit these local, gitignored files for the providers you use. Select the provider and model in the demo script's `LLM` configuration, leaving one selection active.

### Hosted providers

Set the relevant key in `.env`:

```dotenv
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

The [.env template](../.env.example) includes links to each provider's key-management page. Only the selected provider needs a key.

### Ollama

For local inference through Ollama, set `ollama_host` and `ollama_model` in `config.py`, then select `get_ollama()` in the demo script. `ollama_host = None` uses the local server at `localhost:11434`; for a remote server, supply its URL. Ensure the selected model is available on that server. No hosted-provider API key is needed.

### vLLM

For a vLLM server, set `vllm_host` (including the `/v1` suffix) and `vllm_model` in `config.py`, then select `get_vllm()` in the demo script. Set `VLLM_API_KEY` in `.env` to the server's bearer token; for servers launched without authentication, use any nonempty value. Choose a model served by that endpoint.

### LiteLLM

For a LiteLLM proxy, set `litellm_host` and `litellm_model` in `config.py`, then select `get_litellm()` in the demo script. Set `LITELLM_API_KEY` in `.env` to your proxy's virtual key. Use the model name registered by the proxy administrator.

### External physics software

Set `mg5_path`, `wolframscript_path`, and `feynrules_path` in `config.py` as needed by the demo. The scripts pass these settings to toolbase as configuration overrides. See the [external dependency guide](../docs/usage.md#external-dependencies) and each example's README for the required software.

## Run a standalone demo

After configuring its provider and dependencies, run the example from the repository root:

**Symbolic calculations:**

```bash
python examples/eda/eda_demo.py
```

**Diagram enumeration and NDA:**

```bash
python examples/nda/nda_demo.py
```

**Leptoquark simulation:**

```bash
python examples/sim/s1_lq_rr/s1_lq_rr_demo.py
```

Each demo opens an Orchestral web UI at `http://127.0.0.1:8000`. See the individual example READMEs for prompts, launchers, and worked results.
