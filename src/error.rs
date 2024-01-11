use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    ParseFloatError(#[from] std::num::ParseFloatError),
}

pub type Result<T> = core::result::Result<T, Error>;
