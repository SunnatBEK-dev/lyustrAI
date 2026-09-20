# Adaptive multi-model routing

Adaptive routing chooses a workflow without spending another model request.
`CapabilityRouter` inspects bounded signals such as retrieval context,
context-related keywords, reasoning terms, multi-part structure, and request
length. It returns an explainable `RoutingDecision` containing the selected
route, required capabilities, and safe signal names.

The web UI exposes two Adaptive effort choices. `Auto` uses those deterministic
signals to select the smallest suitable route. `Maximum` is an explicit user
override that always selects `FULL`, regardless of prompt wording. The override
is recorded as a safe signal and does not require another model call.

Four routes are available. `FAST` performs one OpenAI synthesis stage.
`CONTEXT` uses Gemini for context extraction followed by OpenAI synthesis.
`REASONING` uses Anthropic for analysis followed by OpenAI synthesis. `FULL`
runs Gemini context extraction, Anthropic reasoning, and OpenAI synthesis.
Stages exchange validated, bounded handoff payloads rather than arbitrary
provider objects.

The default offline routing benchmark contains 24 balanced English and Uzbek
cases: six cases for each route. At the current rules it selects 24 of 24
expected routes. Those decisions estimate 48 provider requests, compared with
72 requests if every case always used the three-stage FULL route. The reported
24-request reduction is an internal workflow estimate, not a claim about token
cost or answer quality.

`Maximum` intentionally gives up those request savings for one turn. It always
uses three provider requests and will normally have higher cost and latency.

Route selection is deliberately deterministic. This makes behavior cheap,
auditable, and regression-testable. It also means ambiguous natural-language
requests can be misclassified. The evaluation report discloses this limitation
and keeps route quality separate from generation quality.
