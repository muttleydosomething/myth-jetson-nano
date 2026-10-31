//! Buffered host timeline at existing request boundaries; no GPU events/fences.
//! Private shared collector keeps diagnostic state out of numerical owners.
use serde::Serialize;
use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
const CAPACITY: usize = 32768;
thread_local! { static ACTIVE: RefCell<Option<Trace>> = const { RefCell::new(None) }; }
#[derive(Serialize)]
struct Event {
    id: usize,
    parent: Option<usize>,
    name: &'static str,
    detail: Option<String>,
    index: Option<usize>,
    start_ns: u128,
    duration_ns: u128,
    complete: bool,
}
struct Trace {
    began: Instant,
    started_unix_ms: u128,
    events: Vec<Event>,
    current: Option<usize>,
    dropped: usize,
    capacity: usize,
}
// Guards remain on their originating synchronous inference worker.
pub(crate) struct Span {
    id: Option<usize>,
    _local: PhantomData<Rc<()>>,
}
pub(crate) struct Root {
    active: bool,
    success: bool,
    route: &'static str,
    _local: PhantomData<Rc<()>>,
}
impl Root {
    pub fn begin(route: &'static str) -> Self {
        Self::start(
            route,
            std::env::var_os("HAILSTORM_NATIVE_TRACE").is_some_and(|v| v == "1"),
            CAPACITY,
        )
    }
    fn start(route: &'static str, enabled: bool, capacity: usize) -> Self {
        let active = enabled
            && ACTIVE.with(|slot| {
                let mut slot = slot.borrow_mut();
                if slot.is_some() {
                    return false;
                }
                *slot = Some(Trace {
                    began: Instant::now(),
                    started_unix_ms: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis(),
                    events: Vec::with_capacity(capacity),
                    current: None,
                    dropped: 0,
                    capacity,
                });
                true
            });
        Self {
            active,
            success: false,
            route,
            _local: PhantomData,
        }
    }
    pub fn success(&mut self) {
        self.success = true;
    }
    fn take(&mut self) -> Option<Trace> {
        if !self.active {
            return None;
        }
        self.active = false;
        ACTIVE.with(|slot| slot.borrow_mut().take())
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        if let Some(trace) = self.take() {
            // Retire TLS before output. Failed diagnostic I/O cannot fail speech.
            let duration_ns = trace.began.elapsed().as_nanos();
            let receipt = serde_json::json!({"operation":"native_call_trace", "schema_version":1,
                "route":self.route,"success":self.success && !std::thread::panicking(),
                "started_unix_ms":trace.started_unix_ms,
                "finished_unix_ms":SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
                "duration_ns":duration_ns,"clock":"host-monotonic",
                "capacity":trace.capacity,"dropped_spans":trace.dropped,"events":trace.events});
            use std::io::Write;
            let _ = writeln!(std::io::stderr().lock(), "{receipt}");
        }
    }
}
/// Disabled spans do not read a clock, allocate labels, or acquire a lock.
pub(crate) fn span(name: &'static str) -> Span {
    begin(name, None, None)
}
pub(crate) fn named(name: &'static str, detail: &str) -> Span {
    begin(name, Some(detail), None)
}
pub(crate) fn iteration(name: &'static str, index: usize) -> Span {
    begin(name, None, Some(index))
}
fn begin(name: &'static str, detail: Option<&str>, index: Option<usize>) -> Span {
    let id = ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let trace = slot.as_mut()?;
        if trace.events.len() == trace.capacity {
            trace.dropped += 1;
            return None;
        }
        let id = trace.events.len();
        trace.events.push(Event {
            id,
            parent: trace.current,
            name,
            detail: detail.map(str::to_owned),
            index,
            start_ns: trace.began.elapsed().as_nanos(),
            duration_ns: 0,
            complete: false,
        });
        trace.current = Some(id);
        Some(id)
    });
    Span {
        id,
        _local: PhantomData,
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            ACTIVE.with(|slot| {
                let mut slot = slot.borrow_mut();
                if let Some(trace) = slot.as_mut() {
                    let end = trace.began.elapsed().as_nanos();
                    let event = &mut trace.events[id];
                    event.duration_ns = end.saturating_sub(event.start_ns);
                    event.complete = true;
                    trace.current = event.parent;
                }
            });
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_has_no_trace_or_event() {
        let mut root = Root::start("disabled", false, 4);
        let span = named("ignored", "no allocation");
        assert!(span.id.is_none());
        assert!(root.take().is_none());
    }
    #[test]
    fn nesting_preserves_accounting_and_root_ownership() {
        let mut root = Root::start("outer", true, 8);
        {
            let _parent = span("parent");
            let mut nested = Root::start("nested", true, 8);
            assert!(nested.take().is_none());
            {
                let _child = iteration("child", 2);
            }
        }
        let trace = root.take().unwrap();
        assert_eq!(trace.events.len(), 2);
        let a = &trace.events[0];
        let b = &trace.events[1];
        assert!(a.complete && b.complete);
        assert_eq!(b.parent, Some(0));
        assert_eq!(b.index, Some(2));
        assert!(a.duration_ns >= b.duration_ns);
        assert!(b.start_ns >= a.start_ns);
        assert!(b.start_ns + b.duration_ns <= a.start_ns + a.duration_ns);
        assert!(trace.current.is_none());
    }
    #[test]
    fn overflow_is_bounded_and_preserves_parent() {
        let mut root = Root::start("overflow", true, 2);
        {
            let _a = span("a");
            {
                let _b = span("b");
            }
            {
                let _c = span("c");
            }
            {
                let _d = span("d");
            }
        }
        let trace = root.take().unwrap();
        assert_eq!(trace.events.len(), 2);
        assert_eq!(trace.dropped, 2);
        assert!(trace.events.iter().all(|e| e.complete));
        assert!(trace.current.is_none());
    }
    #[test]
    fn error_and_panic_retire_trace_for_next_request() {
        fn failed() -> Result<(), ()> {
            let _root = Root::start("error", true, 8);
            let _span = span("failing");
            Err(())?
        }
        assert!(failed().is_err());
        assert!(ACTIVE.with(|s| s.borrow().is_none()));
        let _ = std::panic::catch_unwind(|| {
            let _root = Root::start("panic", true, 8);
            let _span = span("unwinding");
            panic!("diagnostic test");
        });
        assert!(ACTIVE.with(|s| s.borrow().is_none()));
        let mut next = Root::start("next", true, 8);
        assert!(next.take().is_some());
    }
}
