use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
pub struct IDGenerator {
    next_id : AtomicUsize,
}

impl Default for IDGenerator {
    fn default() -> Self {
        IDGenerator {
            next_id: AtomicUsize::new(1),
        }
    }
}

impl IDGenerator {
    pub fn generate_id(&self) -> usize {
        self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
}