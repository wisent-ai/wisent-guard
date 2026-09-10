//! The `ster` binary. Everything it owns lives in `cli`; the work itself is
//! the library's.

mod cli;
mod onboarding;

use anyhow::Result;

fn main() -> Result<()> {
    cli::run()
}
