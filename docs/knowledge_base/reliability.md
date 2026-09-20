# Reliability, progress, and cancellation

Adaptive provider calls have one explicit retry owner. Provider SDK retries are
disabled, and `RetryPolicy` retries only known transient connection, timeout,
rate-limit, conflict, and server failures. Permanent configuration and request
errors are returned immediately. Attempts and delays are bounded by environment
configuration, with three total attempts by default.

Every adaptive workflow emits ordered, content-free progress events. Events
report the route, stage identifier, terminal status, completed stage count, and
expected stage count. They never include prompts, provider responses, tool
arguments, document text, or handoff payloads. The web layer maps these events
to Server-Sent Events for a live execution timeline.

Cancellation is cooperative and thread-safe. A cancellation token is checked
before provider attempts, retry delays, tool execution, and handoff stages. The
system stops only at a safe boundary; it does not terminate a Python thread or
pretend a blocking provider request was interrupted. If cancellation wins, the
unfinished conversation turn is rolled back and a `cancelled` event is sent.

The local demo permits one active chat run. A second concurrent request returns
a conflict instead of mutating shared conversation state. Rename and delete are
also rejected while that run is active. A first failed or cancelled request
removes its newly allocated chat when no messages were saved; a provider-saved
turn remains recoverable. Provider failures are contained: browser responses
receive a safe error category and never echo raw exception messages that could
contain sensitive request details.

Runtime metrics are bounded and content-free. They record route, stage IDs,
success or failure, exception class, and duration. They do not persist user
messages or API keys.
