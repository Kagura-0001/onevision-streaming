# Road inspection verification

The opt-in `road_inspection` policy uses the serving model to recheck a
response trigger before publishing any text. It uses the existing native
`add_branch_request(..., suffix_token_ids=...)` API; no second model or vLLM
changes are required. This is a model judgment, not a guarantee of factual
accuracy. Evaluate both unsupported alerts and missed events on your footage.

```python
async with client.session(
    verification="road_inspection",
    verification_timeout=10.0,
    max_visual_frames=120,
    compact_to_visual_frames=60,
) as session:
    # Send frames and consume events as usual.
    ...
```

The source still samples one control per input. Silence creates no verifier.
On a response trigger, the session forks the source's computed prefix and
appends a short text-only verification turn instead of generating a first
draft. The verifier inherits the frames and memories already in that prefix;
it receives no future frames or separately selected images. Its instruction
names the triggering frame's timestamp and asks it to ground any alert in
that frame. The source resumes input after branch admission, while the branch
generates. Native branching shares full attention blocks and copies the
partial tail and recurrent state needed to keep the two requests independent.

The branch can emit `<silence>` or `<response>` followed by an evidence-first
alert. The latter is published only when complete and syntactically valid.
Rejected, malformed, empty, truncated and timed-out results publish no alert
and write no answer memory. Accepted text uses the usual `answer_commit`
behavior: automatically on completion, or only through `commit_answer()` in
client mode. The verification suffix itself never enters source history.
Caption branches retain their existing behavior; disabling captions is a
renderer configuration, independent of verification.

Only one verification branch runs per session. While it is pending, additional
response triggers are skipped and recorded in `input.done` metadata as
`verification_skipped="verification_busy"`. This limits cost; it does not
identify whether two triggers refer to the same physical event, and it can
miss a distinct brief event. After completion, later triggers can create new
branches. Repeated-event suppression in the policy is a prompt instruction,
not a deterministic event tracker.

## Events and latency

Use `response.output_text.done` (or `session.responses()`) for user-visible
alerts. `verification.done` is an audit event: `metadata.raw` holds unfiltered
model output, `metadata.accepted` says whether it passed the policy parser,
and `reason` explains rejection. Do not display audit text as an alert.
`response.done` still terminates each branch. A model rejection is completed
with a `verification_*` reason; timeout is cancelled with
`verification_timeout`. Backend failures are failed responses.
The Python SDK exposes these extra fields through `event.metadata`; the
HTTP/SSE JSON representation places them directly at the event's top level.

`metadata.timing` reports milliseconds, measured on the server:

| Field | Interval |
|---|---|
| `server_preprocess_ms` | Session entry to input queue insertion |
| `input_queue_ms` | Queue insertion to processing start |
| `control_ms` | Processing start to source control result, including any compaction |
| `candidate_to_fork_ms` | Control result to branch admission |
| `branch_first_output_ms` | Admission to first nonempty raw output chunk |
| `branch_complete_ms` | Admission to complete raw output |
| `response_ready_ms` | Session entry to this event's creation |

The first raw chunk may be a control marker and is not a user-visible first
word. Compare `response_ready_ms` on accepted text events for server delivery
latency. These numbers exclude HTTP image decoding before session entry,
transport and client rendering. Input queue time can grow if the producer is
faster than inference; report it separately from branch generation time.

The timeout bounds branch admission and generation under normal engine
operation. Cleanup must preserve ADD-before-ABORT ordering, so a hung engine
admission can also delay cleanup; it is not a process-level watchdog.

## Compatibility

Use the V1 runner and the fork's supported execution settings from the main
README. The verifier needs visible silence/response control strings in decoded
output. Checkpoints that mark these controls as tokenizer special tokens may
require a backend decoding change; the tested OV step6000 tokenizer stores
them as ordinary added tokens. CPU tests validate routing, publication,
memory isolation, cancellation, compaction and client options; they do not
establish visual accuracy or GPU performance.
